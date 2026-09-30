"""Gene-Query JEPA trainer (Lightning). KEEP THIS FILE.

Train:       docs/examples/train_gene_query_jepa.py (--data toy|full)
HPO:         docs/examples/run_gene_query_toy_sweep.sh
Honesty:     gene_gap_vs_copy_src must be > 0 (prediction beats copy-source).
Index:       docs/examples/GENE_QUERY_JEPA.md

WHAT HAPPENS EVERY TRAINING STEP (plain language)
-------------------------------------------------
1. The dataloader hands us a batch: source cells (normal / resting tokens)
   and their pseudo-paired target cells at each later timepoint (t1, t2, t3).
2. For every cell and timepoint we SAMPLE GENE QUERIES — a quiz sheet of
   ``n_queries`` genes per cell (three parts):
      ~40%  genes present in BOTH source and target   (shared — shift)
      ~50%  genes present ONLY in the target           (induced)
      ~10%  genes absent from the target               (placeholder pad; no loss)
3. The model predicts each queried gene's embedding at the target time.
   Timepoints are independent (no autoregressive context).
4. Losses:  gene loss   = 1 - cosine(predicted, true) on PRESENT queries only
            cell loss   = 1 - cosine(learned cell-query pred, encoder CLS)
            contrastive = InfoNCE on present queries (anti-collapse; default 0)
            VICReg      = var/cov on z_hat_cell only (no gene bag)
            gene MSE    = logged only (train/val), not in the objective
   total = lambda_gene * gene_loss + lambda_cell * cell_loss
           + lambda_contr * contrastive + vic_var * var + vic_cov * cov
5. After the optimiser step, the EMA target encoder drifts a tiny step
   towards the online encoder.

THE TWO TOKEN-ID SPACES (important!)
------------------------------------
* GLOBAL ids: the pretrained vocabulary (~19 000 genes). Source sequences
  arrive in this space, and the pretrained scmaskgit encoder expects it.
* LOCAL ids: the 2 000-HVG vocabulary of this dataset (ids 0..1859).
  Target sequences arrive in this space.
* ids 0..3 are special in the LOCAL space (0=pad, 1=mask, 2=cls, 3=eos),
  so REAL GENES ARE LOCAL IDS >= 4. In the GLOBAL space only 0 (pad)
  matters for us.
* ``tokenid_to_rowid_path`` gives the dictionary between the two spaces.

The sampler thinks entirely in LOCAL ids (small and simple). Right before
tensors enter an encoder we convert to whatever space that encoder expects.

HOW TO READ THE METRICS
-----------------------
The question we must answer honestly: did the model learn dynamics, or does
it just copy? So next to the prediction quality we always log two baselines:
* copy-source baseline: cosine(source gene embedding, target gene embedding)
  — what you get by assuming nothing changes.
* static baseline: cosine(gene identity vector, target gene embedding)
  — what you get knowing only the gene's name, no cell at all.
``gene_gap_vs_copy_src`` and ``gene_gap_vs_static`` are (prediction minus
baseline): POSITIVE means the model beats copying. If those gaps hover at
zero, the model has not learned anything useful — stop and rethink.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule

from perturbgen.Modules.gene_query_jepa import GeneQueryJEPA, GeneQueryPredictor
from perturbgen.src.jepa_metrics import vicreg_var_cov
from perturbgen.src.jepa_token_maps import apply_id_lookup, maybe_load_maps

# In the LOCAL id space: 0=pad, 1=mask, 2=cls, 3=eos. Genes start here.
FIRST_REAL_GENE_LOCAL_ID = 4
# <mask> is id 1 in both LOCAL and GLOBAL LPS vocabs.
SRC_MASK_TOKEN_ID = 1


# ==========================================================================
# The query sampler. Plain Python on purpose: per cell we do set operations
# on ~100-2000 gene ids, which costs well under a millisecond per cell.
# ==========================================================================
def sample_query_batch(
    src_input_ids_global: torch.Tensor,   # (B, Ls) source tokens, GLOBAL ids
    tgt_input_ids_local: torch.Tensor,    # (B, Lt) target tokens, LOCAL ids
    global_to_local: torch.Tensor,        # lookup: GLOBAL id -> LOCAL id (0 if none)
    all_gene_local_ids: List[int],        # every real gene in the LOCAL vocab
    n_queries: int,
    frac_shared: float,
    frac_tgt_only: float,
    rng: random.Random,
    query_mode: str = 'mixed',
    shared_max_queries: int = 0,
    mask_frac_of_shared: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Build the per-cell quiz sheet. Returns CPU tensors, all shaped (B, Q).

    Three pools: shared (~40%), target-only (~50%), absent placeholders (rest).
    query_mode / shared_max_queries are accepted for old CLI flags and ignored.
    ``mask_frac_of_shared`` marks that fraction of *chosen shared* slots for
    mask-CE (no gene identity). Default 0 keeps the old quiz.
    """
    del query_mode, shared_max_queries
    batch_size = tgt_input_ids_local.size(0)

    src_rows = src_input_ids_global.detach().cpu().tolist()
    tgt_rows = tgt_input_ids_local.detach().cpu().tolist()
    global_to_local_list = global_to_local.detach().cpu().tolist()

    all_genes_set = set(all_gene_local_ids)
    n_shared_wanted = round(frac_shared * n_queries)
    n_tgt_only_wanted = round(frac_tgt_only * n_queries)
    mask_frac = max(0.0, float(mask_frac_of_shared))

    out_gene_ids: List[List[int]] = []
    out_is_present: List[List[bool]] = []
    out_tgt_position: List[List[int]] = []
    out_src_position: List[List[int]] = []
    out_is_mask_ce: List[List[bool]] = []

    for cell in range(batch_size):
        tgt_gene_position: Dict[int, int] = {}
        for position, token in enumerate(tgt_rows[cell]):
            if token >= FIRST_REAL_GENE_LOCAL_ID and token not in tgt_gene_position:
                tgt_gene_position[token] = position

        src_gene_position: Dict[int, int] = {}
        for position, token in enumerate(src_rows[cell]):
            if 0 <= token < len(global_to_local_list):
                local_id = global_to_local_list[token]
            else:
                local_id = 0
            if local_id >= FIRST_REAL_GENE_LOCAL_ID and local_id not in src_gene_position:
                src_gene_position[local_id] = position

        shared_pool = [g for g in tgt_gene_position if g in src_gene_position]
        tgt_only_pool = [g for g in tgt_gene_position if g not in src_gene_position]

        take_shared = min(len(shared_pool), n_shared_wanted)
        take_tgt_only = min(len(tgt_only_pool), n_tgt_only_wanted)
        present_quota = n_shared_wanted + n_tgt_only_wanted
        missing = present_quota - (take_shared + take_tgt_only)
        if missing > 0:
            extra = min(missing, len(shared_pool) - take_shared)
            take_shared += extra
            missing -= extra
        if missing > 0:
            extra = min(missing, len(tgt_only_pool) - take_tgt_only)
            take_tgt_only += extra
        n_absent = n_queries - take_shared - take_tgt_only

        chosen_shared = rng.sample(shared_pool, take_shared) if take_shared else []
        n_mask = int(round(mask_frac * len(chosen_shared))) if mask_frac > 0 else 0
        n_mask = min(n_mask, len(chosen_shared))
        mask_shared = set(rng.sample(chosen_shared, n_mask)) if n_mask else set()
        chosen_tgt_only = (
            rng.sample(tgt_only_pool, take_tgt_only) if take_tgt_only else []
        )
        absent_pool = list(all_genes_set - set(tgt_gene_position))
        chosen_absent = rng.sample(
            absent_pool, min(n_absent, len(absent_pool))
        ) if n_absent and absent_pool else []

        row_gene_ids: List[int] = []
        row_is_present: List[bool] = []
        row_tgt_position: List[int] = []
        row_src_position: List[int] = []
        row_is_mask_ce: List[bool] = []
        for gene in chosen_shared + chosen_tgt_only:
            row_gene_ids.append(gene)
            row_is_present.append(True)
            row_tgt_position.append(tgt_gene_position[gene])
            row_src_position.append(src_gene_position.get(gene, -1))
            row_is_mask_ce.append(gene in mask_shared)
        for gene in chosen_absent:
            row_gene_ids.append(gene)
            row_is_present.append(False)
            row_tgt_position.append(0)
            row_src_position.append(-1)
            row_is_mask_ce.append(False)
        while len(row_gene_ids) < n_queries:
            row_gene_ids.append(all_gene_local_ids[0])
            row_is_present.append(False)
            row_tgt_position.append(0)
            row_src_position.append(-1)
            row_is_mask_ce.append(False)

        out_gene_ids.append(row_gene_ids)
        out_is_present.append(row_is_present)
        out_tgt_position.append(row_tgt_position)
        out_src_position.append(row_src_position)
        out_is_mask_ce.append(row_is_mask_ce)

    return {
        'gene_ids_local': torch.tensor(out_gene_ids, dtype=torch.long),
        'is_present': torch.tensor(out_is_present, dtype=torch.bool),
        'tgt_position': torch.tensor(out_tgt_position, dtype=torch.long),
        'src_position': torch.tensor(out_src_position, dtype=torch.long),
        'is_mask_ce': torch.tensor(out_is_mask_ce, dtype=torch.bool),
    }


def apply_src_masks(
    src_input_ids: torch.Tensor,
    src_position: torch.Tensor,
    is_mask_ce: torch.Tensor,
    mask_token_id: int = SRC_MASK_TOKEN_ID,
) -> torch.Tensor:
    """Replace shared-mask slots in the source sequence with <mask>."""
    masked = src_input_ids.clone()
    if not bool(is_mask_ce.any()):
        return masked
    batch_idx, query_idx = torch.where(is_mask_ce)
    pos = src_position[batch_idx, query_idx]
    valid = pos >= 0
    if not bool(valid.any()):
        return masked
    batch_idx = batch_idx[valid]
    pos = pos[valid]
    seq_len = masked.size(1)
    in_range = pos < seq_len
    masked[batch_idx[in_range], pos[in_range]] = int(mask_token_id)
    return masked


# ==========================================================================
# Two tiny math helpers, named for what they mean.
# ==========================================================================
def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, per row. 0 = same direction, 2 = opposite."""
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average of ``values`` where ``mask`` is True (safe when all-False)."""
    mask = mask.to(values.dtype)
    total = (values * mask).sum()
    count = mask.sum().clamp(min=1.0)
    return total / count


def info_nce_present_genes(
    z_hat: torch.Tensor,          # (B, Q, D) predicted gene embeddings
    z_tgt: torch.Tensor,          # (B, Q, D) true EMA gene embeddings
    is_present: torch.Tensor,     # (B, Q)    True for genes present in target
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE over present gene queries only (shared + tgt-only).

    Anchor = predicted embedding for gene g.
    Positive = EMA target embedding for the same gene in the same cell.
    Negatives = all other present-query target embeddings in the batch.

    Absent decoys are excluded: they already use the learned-null cosine term.
    """
    anchors = z_hat[is_present]       # (N, D)
    positives = z_tgt[is_present]     # (N, D)
    n = anchors.size(0)
    if n < 2:
        return z_hat.new_zeros(())

    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)
    logits = (anchors @ positives.T) / temperature          # (N, N)
    labels = torch.arange(n, device=anchors.device)
    return F.cross_entropy(logits, labels)


def parse_encoder_lr_schedule(
    spec: Optional[str],
) -> List[tuple]:
    """Parse ``freeze:5,5e-7:5,1e-6:5`` into ``[(5, None), (5, 5e-7), ...]``.

    ``None`` lr means the online encoder is frozen for that many epochs.
    After the last stage the last action is held.
    """
    if not spec:
        return []
    stages: List[tuple] = []
    for part in spec.split(','):
        token = part.strip()
        if not token:
            continue
        if ':' not in token:
            raise ValueError(
                f'encoder_lr_schedule entries need duration, got {token!r}'
            )
        kind, duration_s = token.rsplit(':', 1)
        duration = int(duration_s)
        if duration < 1:
            raise ValueError(f'schedule duration must be >= 1, got {token!r}')
        kind = kind.strip().lower()
        if kind in ('freeze', 'off'):
            stages.append((duration, None))
        else:
            stages.append((duration, float(kind)))
    return stages


def encoder_stage_at_epoch(
    stages: List[tuple], epoch: int
) -> tuple:
    """Return ``(frozen, lr)`` for a 0-based epoch. Empty stages → (False, None)."""
    if not stages:
        return False, None
    cursor = 0
    last = stages[-1]
    for duration, lr in stages:
        if epoch < cursor + duration:
            return lr is None, lr
        cursor += duration
    return last[1] is None, last[1]


class GeneQueryJEPATrainer(LightningModule):
    """PyTorch-Lightning wrapper: batches in, losses/metrics/checkpoints out."""

    def __init__(
        self,
        # ---- encoder ----
        jepa_encoder: str = 'scmaskgit',        # 'scmaskgit' or 'cell'
        encoder_path: Optional[str] = None,     # pretrained ckpt (scmaskgit)
        jepa_encoder_layers: int = 6,           # early-exit depth (scmaskgit)
        freeze_jepa_encoder: bool = False,
        freeze_encoder_epochs: int = 0,
        encoder_lr_schedule: Optional[str] = None,
        # ---- query sampling ----
        n_queries: int = 64,
        frac_shared: float = 0.4,
        frac_tgt_only: float = 0.5,
        query_mode: str = 'mixed',
        shared_max_queries: int = 0,
        predictor_layers: int = 2,
        predictor_arch: str = 'torch',
        # ---- loss weights (gene-primary, cell-auxiliary) ----
        lambda_gene: float = 1.0,
        lambda_cell: float = 0.0,
        lambda_contrastive: float = 0.0,
        contrastive_temperature: float = 0.1,
        vicreg_var_coeff: float = 0.0,
        vicreg_cov_coeff: float = 0.0,
        vicreg_gamma: Optional[float] = None,
        gene_loss: str = 'cosine',
        lambda_mask_ce: float = 0.0,
        mask_frac_of_shared: float = 0.0,
        use_query_src_pos: bool = True,
        dump_raw_latents: bool = False,
        src_mask_token_id: int = SRC_MASK_TOKEN_ID,
        # ---- optimisation ----
        lr: float = 1e-4,
        encoder_lr: Optional[float] = None,
        weight_decay: float = 1e-4,
        ema_decay: float = 0.996,
        normalize_latents: bool = True,
        dropout: float = 0.0,
        # ---- data / bookkeeping ----
        pred_tps: Optional[List[int]] = None,
        n_total_tps: int = 3,
        tokenid_to_rowid_path: Optional[str] = None,
        output_dir: str = './T_perturb/res/jepa_gene_query/',
        var_list: Optional[List[str]] = None,
        seed: int = 42,
        # ---- only used by the small 'cell' test encoder ----
        tgt_vocab_size: int = 25000,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        d_ff: int = 1024,
        max_seq_length: int = 512,
        cell_pool: str = 'cls',
        # Extra CLI keys shared with other train modes; accepted and ignored.
        **unused_cli_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pred_tps = pred_tps if pred_tps is not None else [1, 2, 3]
        self.n_queries = n_queries
        self.frac_shared = frac_shared
        self.frac_tgt_only = frac_tgt_only
        self.query_mode = query_mode
        self.shared_max_queries = int(shared_max_queries)
        self.lambda_gene = lambda_gene
        self.lambda_cell = lambda_cell
        self.lambda_contrastive = float(lambda_contrastive)
        self.contrastive_temperature = float(contrastive_temperature)
        self.vicreg_var_coeff = float(vicreg_var_coeff)
        self.vicreg_cov_coeff = float(vicreg_cov_coeff)
        self.vicreg_gamma = vicreg_gamma
        gene_loss = str(gene_loss).lower()
        if gene_loss not in ('cosine', 'mse'):
            raise ValueError(f'gene_loss must be cosine or mse, got {gene_loss!r}')
        self.gene_loss_type = gene_loss
        self.lambda_mask_ce = float(lambda_mask_ce)
        self.mask_frac_of_shared = float(mask_frac_of_shared)
        self.use_query_src_pos = bool(use_query_src_pos)
        self.dump_raw_latents = bool(dump_raw_latents)
        self.src_mask_token_id = int(src_mask_token_id)
        self.lr = lr
        self.encoder_lr = float(lr if encoder_lr is None else encoder_lr)
        self.weight_decay = weight_decay
        self.jepa_encoder = jepa_encoder
        self.freeze_encoder_epochs = int(freeze_encoder_epochs)
        if self.freeze_encoder_epochs < 0:
            raise ValueError('freeze_encoder_epochs must be >= 0')
        self.encoder_lr_schedule = parse_encoder_lr_schedule(encoder_lr_schedule)
        if self.encoder_lr_schedule:
            start_frozen, _ = encoder_stage_at_epoch(self.encoder_lr_schedule, 0)
            encoder_trains = any(lr is not None for _, lr in self.encoder_lr_schedule)
        else:
            start_frozen = bool(freeze_jepa_encoder) or self.freeze_encoder_epochs > 0
            encoder_trains = (not freeze_jepa_encoder) or self.freeze_encoder_epochs > 0
        # DDP registers trainable parameters once, when the model is wrapped.
        # Flipping requires_grad later leaves every rank updating its own
        # encoder copy without gradient sync. So if the encoder trains at any
        # point, its used layers keep requires_grad=True from step 0 and a
        # "frozen" stage is expressed as encoder lr = 0 instead.
        self.encoder_trains = bool(encoder_trains)
        self._encoder_unfrozen = not start_frozen
        self._encoder_param_ids: set = set()
        self.output_dir = output_dir
        self.var_list = var_list or []
        os.makedirs(output_dir, exist_ok=True)

        # ---- the LOCAL <-> GLOBAL id dictionaries (see file docstring) ----
        if not tokenid_to_rowid_path:
            raise ValueError(
                'gene_query_jepa needs --tokenid_to_rowid_path (the pickle '
                'that maps global pretrain token ids to local HVG row ids)'
            )
        g2l, l2g = maybe_load_maps(tokenid_to_rowid_path)
        self.register_buffer('global_to_local', g2l, persistent=False)
        self.register_buffer('local_to_global', l2g, persistent=False)

        # Every real gene in the LOCAL vocabulary (ids 4..end that map to a
        # real global id). This is the pool absent-decoys are drawn from.
        self.all_gene_local_ids = [
            local_id
            for local_id in range(FIRST_REAL_GENE_LOCAL_ID, l2g.numel())
            if int(l2g[local_id]) != 0
        ]

        # ---- the model itself ----
        self.model = GeneQueryJEPA(
            encoder_type=jepa_encoder,
            encoder_path=encoder_path,
            n_encoder_layers=jepa_encoder_layers,
            freeze_encoder=start_frozen,
            n_time_steps=n_total_tps,
            predictor_layers=predictor_layers,
            dropout=dropout,
            ema_decay=ema_decay,
            normalize_latents=normalize_latents,
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            cell_pool=cell_pool,
            predictor_arch=predictor_arch,
            use_query_src_pos=self.use_query_src_pos,
        )
        self.mask_ce_head = nn.Linear(
            self.model.d_model, int(self.local_to_global.numel())
        )
        if self.encoder_trains:
            # Grads flow through the encoder from step 0; the schedule only
            # moves this group's lr (0 while "frozen").
            self.model.unfreeze_online_encoder()
        if self.lambda_mask_ce <= 0.0:
            # Mask-CE branch never enters the loss; leave its weights out of
            # the optimiser so DDP does not wait for gradients that never come.
            # query_src_pos_embedding stays trainable when use_query_src_pos.
            for param in self.mask_ce_head.parameters():
                param.requires_grad = False
            self.model.mask_query.requires_grad = False
            if not self.use_query_src_pos:
                for param in self.model.query_src_pos_embedding.parameters():
                    param.requires_grad = False
        else:
            # Mask-CE head lives on the LightningModule (not inside self.model).
            # Keep it explicitly trainable so configure_optimizers picks it up.
            for param in self.mask_ce_head.parameters():
                param.requires_grad = True
            self.model.mask_query.requires_grad = True
            if self.use_query_src_pos:
                for param in self.model.query_src_pos_embedding.parameters():
                    param.requires_grad = True
        print(
            f'GeneQueryJEPA: encoder={jepa_encoder}, d_model={self.model.d_model}, '
            f'cell_pool={cell_pool}, '
            f'n_queries={n_queries}+1 cell_query (shared {frac_shared:.0%} / '
            f'tgt-only {frac_tgt_only:.0%} / absent rest), '
            f'predictor_layers={predictor_layers}, '
            f'use_query_src_pos={self.use_query_src_pos}, '
            f'freeze_encoder_epochs={self.freeze_encoder_epochs}, '
            f'encoder_lr_schedule={encoder_lr_schedule or "off"}, '
            f'lr={self.lr}, encoder_lr={self.encoder_lr}, '
            f'lambda_gene={lambda_gene}, lambda_cell={lambda_cell}, '
            f'lambda_contr={self.lambda_contrastive}, '
            f'tau={self.contrastive_temperature}, '
            f'vicreg_var={self.vicreg_var_coeff}, '
            f'vicreg_cov={self.vicreg_cov_coeff}, '
            f'gene_loss={self.gene_loss_type}, '
            f'lambda_mask_ce={self.lambda_mask_ce}, '
            f'mask_frac_of_shared={self.mask_frac_of_shared}, '
            f'dump_raw_latents={self.dump_raw_latents}'
        )

        # Random generator for query sampling; re-seeded per DDP rank in setup().
        self.rng = random.Random(seed)

        # Buffers for the test-time embedding dump (see test_step).
        self._test_gene_sums: Dict[str, torch.Tensor] = {}
        self._test_gene_counts: Dict[str, torch.Tensor] = {}
        self._test_cell_rows: Dict[str, list] = {}

    def on_test_epoch_start(self) -> None:
        self._test_gene_sums = {}
        self._test_gene_counts = {}
        self._test_cell_rows = {}

    def setup(self, stage: Optional[str] = None) -> None:
        # Give each DDP rank its own sampling stream so GPUs don't all ask
        # the exact same questions.
        self.rng = random.Random(self.hparams.seed + self.global_rank)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Load checkpoints of either predictor shape without losing weights.

        ``arch='torch'`` (the default, used up to 2026-09-22) stores
        ``model.predictor.cross_attention.*``; ``arch='block'`` (the
        2026-09-25 run) stores ``model.predictor.decoder_block.*``. Whichever
        the checkpoint holds wins, so the saved weights load exactly. Keys
        that simply did not exist yet (mask-CE / mask-query) keep the current
        init.
        """
        saved = checkpoint.get('state_dict', {})
        if any(k.startswith('model.predictor.cross_attention.') for k in saved):
            ckpt_arch = 'torch'
        elif any(k.startswith('model.predictor.decoder_block.') for k in saved):
            ckpt_arch = 'block'
        else:
            ckpt_arch = None
        if ckpt_arch is not None and ckpt_arch != self.model.predictor_arch:
            print(
                f'Checkpoint predictor is arch={ckpt_arch!r}, model was built as '
                f'{self.model.predictor_arch!r}; rebuilding to load those weights.'
            )
            self.model.predictor_arch = ckpt_arch
            self.hparams.predictor_arch = ckpt_arch
            self.model.predictor = GeneQueryPredictor(
                d_model=self.model.d_model,
                n_time_steps=self.hparams.n_total_tps,
                num_layers=self.hparams.predictor_layers,
                dropout=self.hparams.dropout,
                arch=ckpt_arch,
            )

        current = self.state_dict()
        missing = [key for key in current if key not in saved]
        unexpected = [key for key in saved if key not in current]
        if unexpected:
            print(
                f'Checkpoint has {len(unexpected)} unused keys, '
                f'e.g. {unexpected[:3]}'
            )
        if missing:
            print(
                f'Checkpoint is missing {len(missing)} keys '
                f'(kept at init), e.g. {missing[:3]}'
            )
        for key in missing:
            saved[key] = current[key]

    def _quiz_kwargs(self) -> dict:
        return {
            'n_queries': self.n_queries,
            'frac_shared': self.frac_shared,
            'frac_tgt_only': self.frac_tgt_only,
            'mask_frac_of_shared': (
                self.mask_frac_of_shared if self.lambda_mask_ce > 0 else 0.0
            ),
            'rng': self.rng,
        }

    # ------------------------------------------------------------------
    # id-space conversion: LOCAL quiz ids -> whatever the encoder expects.
    # ------------------------------------------------------------------
    def _local_ids_to_encoder_space(self, ids_local: torch.Tensor) -> torch.Tensor:
        if self.jepa_encoder == 'scmaskgit':
            # Pretrained encoder speaks GLOBAL.
            return apply_id_lookup(ids_local, self.local_to_global)
        # The small 'cell' encoder speaks LOCAL already.
        return ids_local

    def _src_ids_to_encoder_space(self, src_ids_global: torch.Tensor) -> torch.Tensor:
        if self.jepa_encoder == 'scmaskgit':
            # Source sequences already arrive in GLOBAL.
            return src_ids_global
        # The small encoder needs LOCAL; genes outside the HVG vocab become
        # 0 (pad) and are simply ignored by the attention mask.
        return apply_id_lookup(src_ids_global, self.global_to_local)

    # ------------------------------------------------------------------
    # One timepoint = sample queries, run the model, compute losses.
    # ------------------------------------------------------------------
    def _run_one_timestep(
        self, batch: dict, time_step: int
    ) -> Dict[str, torch.Tensor]:
        src_ids_global = batch['src_input_ids']                     # (B, Ls)
        tgt_ids_local = batch[f'tgt_input_ids_t{time_step}']        # (B, Lt)

        # 1) Sample the quiz sheet (on CPU, plain Python).
        quiz = sample_query_batch(
            src_input_ids_global=src_ids_global,
            tgt_input_ids_local=tgt_ids_local,
            global_to_local=self.global_to_local,
            all_gene_local_ids=self.all_gene_local_ids,
            **self._quiz_kwargs(),
        )
        device = src_ids_global.device
        gene_ids_local = quiz['gene_ids_local'].to(device)          # (B, Q)
        is_present = quiz['is_present'].to(device)                  # (B, Q)
        tgt_position = quiz['tgt_position'].to(device)              # (B, Q)
        src_position = quiz['src_position'].to(device)              # (B, Q)
        is_mask_ce = quiz['is_mask_ce'].to(device)                  # (B, Q)
        gene_mask = is_present & ~is_mask_ce

        src_for_encoder = self._src_ids_to_encoder_space(src_ids_global)
        if bool(is_mask_ce.any()):
            src_for_encoder = apply_src_masks(
                src_for_encoder,
                src_position,
                is_mask_ce,
                mask_token_id=self.src_mask_token_id,
            )

        # 2) Convert every id tensor into the encoder's id space.
        out = self.model.forward_one_timestep(
            src_input_ids=src_for_encoder,
            tgt_input_ids=self._local_ids_to_encoder_space(tgt_ids_local),
            query_gene_ids=self._local_ids_to_encoder_space(gene_ids_local),
            query_is_present=is_present,
            query_tgt_position=tgt_position,
            time_step=time_step,
            query_is_mask_ce=is_mask_ce,
            query_src_position=src_position,
        )

        # 3) Losses. -------------------------------------------------------
        z_hat_raw = out['z_hat_gene_raw']
        z_tgt_raw = out['z_tgt_gene_raw']
        per_query_distance = cosine_distance(
            out['z_hat_gene'], out['z_tgt_gene']
        )                                                           # (B, Q)
        if self.gene_loss_type == 'mse':
            per_query_mse_loss = (z_hat_raw - z_tgt_raw).pow(2).mean(dim=-1)
            gene_loss = masked_mean(per_query_mse_loss, gene_mask)
        else:
            gene_loss = masked_mean(per_query_distance, gene_mask)

        # Cell loss: learned cell-query prediction vs encoder CLS (or mean).
        cell_loss = cosine_distance(
            out['z_hat_cell'], out['z_tgt_cell']
        ).mean()

        if self.lambda_contrastive > 0.0:
            contrastive_loss = info_nce_present_genes(
                z_hat=out['z_hat_gene'],
                z_tgt=out['z_tgt_gene'],
                is_present=gene_mask,
                temperature=self.contrastive_temperature,
            )
        else:
            contrastive_loss = gene_loss.new_zeros(())

        if self.lambda_mask_ce > 0.0 and bool(is_mask_ce.any()):
            mask_h = F.normalize(z_hat_raw[is_mask_ce], dim=-1)
            logits = self.mask_ce_head(mask_h)
            labels = gene_ids_local[is_mask_ce]
            valid = (labels >= 0) & (labels < logits.size(-1))
            if bool(valid.any()):
                mask_ce_loss = F.cross_entropy(logits[valid], labels[valid])
            else:
                mask_ce_loss = gene_loss.new_zeros(())
        else:
            mask_ce_loss = gene_loss.new_zeros(())

        embedding_dim = out['z_hat_cell'].size(-1)
        if self.vicreg_var_coeff > 0.0 or self.vicreg_cov_coeff > 0.0:
            if self.vicreg_gamma is not None:
                gamma = self.vicreg_gamma
            elif self.hparams.normalize_latents:
                gamma = 1.0 / (embedding_dim ** 0.5)
            else:
                gamma = 1.0
            vicreg_var, vicreg_cov = vicreg_var_cov(
                out['z_hat_cell'], gamma=gamma
            )
        else:
            vicreg_var = gene_loss.new_zeros(())
            vicreg_cov = gene_loss.new_zeros(())

        total_loss = (
            self.lambda_gene * gene_loss
            + self.lambda_cell * cell_loss
            + self.lambda_contrastive * contrastive_loss
            + self.lambda_mask_ce * mask_ce_loss
            + self.vicreg_var_coeff * vicreg_var
            + self.vicreg_cov_coeff * vicreg_cov
        )

        # 4) Honesty metrics (no gradients needed). --------------------------
        with torch.no_grad():
            per_query_mse = (z_hat_raw - z_tgt_raw).pow(2).mean(dim=-1)
            gene_mse = masked_mean(per_query_mse, gene_mask)

            per_query_cosine = 1.0 - per_query_distance             # (B, Q)
            gene_cos_pred = masked_mean(per_query_cosine, gene_mask)

            static_cosine = F.cosine_similarity(
                out['z_static_gene'], out['z_tgt_gene'], dim=-1
            )
            gene_cos_static = masked_mean(static_cosine, gene_mask)

            in_both = gene_mask & (src_position >= 0)
            safe_position = src_position.clamp(min=0)
            index = safe_position.unsqueeze(-1).expand(
                -1, -1, embedding_dim
            )
            z_src_gene = torch.gather(out['h_src'], dim=1, index=index)
            copy_cosine = F.cosine_similarity(
                z_src_gene, out['z_tgt_gene'], dim=-1
            )
            gene_cos_copy_src = masked_mean(copy_cosine, in_both)
            gene_cos_pred_shared = masked_mean(per_query_cosine, in_both)

            cell_cos_pred = F.cosine_similarity(
                out['z_hat_cell'], out['z_tgt_cell'], dim=-1
            ).mean()
            cell_cos_copy_src = F.cosine_similarity(
                out['z_src_cell'], out['z_tgt_cell'], dim=-1
            ).mean()

        return {
            'total_loss': total_loss,
            'gene_loss': gene_loss,
            'cell_loss': cell_loss,
            'contrastive_loss': contrastive_loss,
            'mask_ce_loss': mask_ce_loss,
            'vicreg_var': vicreg_var,
            'vicreg_cov': vicreg_cov,
            'gene_mse': gene_mse,
            'gene_cos_pred': gene_cos_pred,
            'gene_gap_vs_static': gene_cos_pred - gene_cos_static,
            'gene_gap_vs_copy_src': gene_cos_pred_shared - gene_cos_copy_src,
            'cell_cos_pred': cell_cos_pred,
            'cell_gap_vs_copy_src': cell_cos_pred - cell_cos_copy_src,
            '_out': out,
            '_gene_ids_local': gene_ids_local,
            '_is_present': gene_mask,
            '_src_position': src_position,
        }

    def encode_one_timestep(
        self,
        batch: dict,
        time_step: int,
        quiz: Optional[dict] = None,
        src_ids_global: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """Forward one timepoint with a frozen quiz (control vs KO, same queries)."""
        src_ids_global = (
            batch['src_input_ids'] if src_ids_global is None else src_ids_global
        )
        tgt_ids_local = batch[f'tgt_input_ids_t{time_step}']
        if quiz is None:
            quiz = sample_query_batch(
                src_input_ids_global=src_ids_global,
                tgt_input_ids_local=tgt_ids_local,
                global_to_local=self.global_to_local,
                all_gene_local_ids=self.all_gene_local_ids,
                n_queries=self.n_queries,
                frac_shared=self.frac_shared,
                frac_tgt_only=self.frac_tgt_only,
                mask_frac_of_shared=0.0,
                rng=self.rng,
            )
        device = src_ids_global.device
        gene_ids_local = quiz['gene_ids_local'].to(device)
        is_present = quiz['is_present'].to(device)
        tgt_position = quiz['tgt_position'].to(device)
        src_position = quiz['src_position'].to(device)
        out = self.model.forward_one_timestep(
            src_input_ids=self._src_ids_to_encoder_space(src_ids_global),
            tgt_input_ids=self._local_ids_to_encoder_space(tgt_ids_local),
            query_gene_ids=self._local_ids_to_encoder_space(gene_ids_local),
            query_is_present=is_present,
            query_tgt_position=tgt_position,
            time_step=time_step,
            query_src_position=src_position,
        )
        return {
            '_out': out,
            '_is_present': is_present,
            '_quiz': quiz,
            '_gene_ids_local': gene_ids_local,
            '_src_position': src_position,
        }

    # ------------------------------------------------------------------
    # Lightning plumbing: average over timepoints, log, EMA update.
    # ------------------------------------------------------------------
    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        batch_size = batch['src_input_ids'].size(0)
        results_per_timestep = []
        for time_step in self.pred_tps:
            if f'tgt_input_ids_t{time_step}' not in batch:
                continue
            result = self._run_one_timestep(batch, time_step)
            results_per_timestep.append(result)
            self.log(
                f'{stage}/gene_loss_t{time_step}',
                result['gene_loss'],
                on_epoch=True, sync_dist=True, batch_size=batch_size,
            )
        if not results_per_timestep:
            raise RuntimeError(
                'Batch has no tgt_input_ids_t{t} entries for '
                f'pred_tps={self.pred_tps} — check the datamodule config.'
            )

        # Average each number over the timepoints and log it.
        keys_to_log = [
            'total_loss', 'gene_loss', 'cell_loss', 'contrastive_loss',
            'mask_ce_loss', 'vicreg_var', 'vicreg_cov', 'gene_mse',
            'gene_cos_pred', 'gene_gap_vs_static', 'gene_gap_vs_copy_src',
            'cell_cos_pred', 'cell_gap_vs_copy_src',
        ]
        averaged = {}
        for key in keys_to_log:
            values = [r[key] for r in results_per_timestep]
            averaged[key] = torch.stack(values).mean()
        step_keys = {
            'total_loss', 'gene_loss', 'mask_ce_loss', 'contrastive_loss',
            'cell_loss', 'gene_mse',
        }
        bar_keys = {
            'total_loss', 'gene_loss', 'mask_ce_loss', 'contrastive_loss',
            'gene_gap_vs_copy_src',
        }
        for key, value in averaged.items():
            self.log(
                f'{stage}/{key}',
                value,
                on_step=(stage == 'train' and key in step_keys),
                on_epoch=True,
                prog_bar=(key in bar_keys),
                sync_dist=True,
                batch_size=batch_size,
            )
        return averaged['total_loss']

    def training_step(self, batch, *args, **kwargs):
        return self._step(batch, 'train')

    def validation_step(self, batch, *args, **kwargs):
        return self._step(batch, 'val')

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # The EMA teacher follows the student, one small step per batch.
        self.model.update_target_encoder()

    def on_train_epoch_start(self) -> None:
        self._apply_encoder_schedule(int(self.current_epoch))

    def encoder_lr_at_epoch(self, epoch: int) -> float:
        """Online-encoder lr for a 0-based epoch; 0.0 means frozen."""
        if not self.encoder_trains:
            return 0.0
        if self.encoder_lr_schedule:
            frozen, lr = encoder_stage_at_epoch(self.encoder_lr_schedule, epoch)
            return 0.0 if frozen else float(lr)
        if self.freeze_encoder_epochs > 0 and epoch < self.freeze_encoder_epochs:
            return 0.0
        return self.encoder_lr

    def _apply_encoder_schedule(self, epoch: int) -> None:
        if not self.encoder_trains:
            return
        lr = self.encoder_lr_at_epoch(epoch)
        group = self._encoder_optimizer_group()
        if group is None:
            return
        if float(group['lr']) != lr:
            state = 'frozen (lr=0)' if lr == 0.0 else f'lr={lr:g}'
            print(f'Epoch {epoch}: online encoder {state}')
        group['lr'] = lr
        self._encoder_unfrozen = lr > 0.0

    def _encoder_optimizer_group(self, optimizer=None):
        if optimizer is None:
            trainer = getattr(self, '_trainer', None)
            if trainer is None or not trainer.optimizers:
                return None
            optimizer = trainer.optimizers[0]
        for group in optimizer.param_groups:
            if any(id(param) in self._encoder_param_ids for param in group['params']):
                return group
        return None

    def configure_optimizers(self):
        """Two AdamW groups: predictor/heads at ``lr``, encoder at its own lr.

        The encoder group exists from the first step (DDP needs a fixed
        trainable set); its lr follows ``encoder_lr_at_epoch`` and is 0.0
        while the schedule says frozen.

        IMPORTANT: collect heads via ``self.parameters()``, not
        ``self.model.parameters()``. ``mask_ce_head`` is registered on this
        LightningModule; using only ``self.model`` silently drops it from
        AdamW (Sep-24 run: CE stayed at chance for 9 epochs).
        """
        encoder_ids = {id(p) for p in self.model.online_encoder.parameters()}
        encoder_params = [
            p for p in self.model.online_encoder.parameters() if p.requires_grad
        ]
        head_params = [
            p
            for p in self.parameters()
            if p.requires_grad and id(p) not in encoder_ids
        ]
        if self.lambda_mask_ce > 0.0:
            ce_ids = {id(p) for p in self.mask_ce_head.parameters()}
            head_ids = {id(p) for p in head_params}
            missing = ce_ids - head_ids
            if missing:
                raise RuntimeError(
                    'mask_ce_head is trainable but missing from the optimiser; '
                    'use self.parameters() (not self.model.parameters()) when '
                    'building head_params'
                )
        groups = [
            {
                'params': head_params,
                'lr': self.lr,
                'weight_decay': self.weight_decay,
            }
        ]
        if encoder_params:
            self._encoder_param_ids = {id(p) for p in encoder_params}
            groups.append(
                {
                    'params': encoder_params,
                    'lr': self.encoder_lr_at_epoch(0),
                    'weight_decay': self.weight_decay,
                }
            )
        optimizer = optim.AdamW(groups, lr=self.lr, weight_decay=self.weight_decay)
        return {'optimizer': optimizer, 'monitor': 'val/total_loss'}

    # ------------------------------------------------------------------
    # Test = dump embeddings for downstream analysis (notebook 05 style).
    #
    # Per (gene, timepoint) we accumulate a RUNNING MEAN of the predicted
    # and the true (EMA) gene embeddings. Queries are random per cell, but
    # over a full test epoch each gene is asked hundreds of times, so the
    # means are well covered. Cell-level embeddings are stored per cell.
    # ------------------------------------------------------------------
    def _accumulate_gene_mean(
        self,
        key: str,
        gene_ids: torch.Tensor,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        vocab_size: int,
    ) -> None:
        if key not in self._test_gene_sums:
            self._test_gene_sums[key] = torch.zeros(
                vocab_size, self.model.d_model
            )
            self._test_gene_counts[key] = torch.zeros(vocab_size)
        flat_ids = gene_ids[mask].cpu()
        if flat_ids.numel() == 0:
            return
        self._test_gene_sums[key].index_add_(
            0, flat_ids, embeddings[mask].cpu().float()
        )
        self._test_gene_counts[key].index_add_(
            0, flat_ids, torch.ones(flat_ids.numel())
        )

    def test_step(self, batch, *args, **kwargs):
        batch_size = batch['src_input_ids'].size(0)
        vocab_size = int(self.local_to_global.numel())
        for time_step in self.pred_tps:
            if f'tgt_input_ids_t{time_step}' not in batch:
                continue
            # Inference quiz only (mask-CE off). Training losses are not computed.
            result = self.encode_one_timestep(batch, time_step)
            out = result['_out']
            gene_ids = result['_gene_ids_local']          # (B, Q) LOCAL ids
            is_present = result['_is_present']            # (B, Q)
            src_position = result['_src_position']        # (B, Q)

            # L2 and raw gene means in one pass (notebook 08 uses L2 keys).
            for suffix, hat_k, tgt_k in (
                ('', 'z_hat_gene', 'z_tgt_gene'),
                ('_raw', 'z_hat_gene_raw', 'z_tgt_gene_raw'),
            ):
                self._accumulate_gene_mean(
                    f'hat_t{time_step}{suffix}',
                    gene_ids,
                    out[hat_k],
                    is_present,
                    vocab_size,
                )
                self._accumulate_gene_mean(
                    f'tgt_t{time_step}{suffix}',
                    gene_ids,
                    out[tgt_k],
                    is_present,
                    vocab_size,
                )

            in_both = is_present & (src_position >= 0)
            if bool(in_both.any()):
                dim = out['h_src'].size(-1)
                safe_pos = src_position.clamp(min=0)
                index = safe_pos.unsqueeze(-1).expand(-1, -1, dim)
                for suffix, h_k in (('', 'h_src'), ('_raw', 'h_src_raw')):
                    z_src_gene = torch.gather(out[h_k], dim=1, index=index)
                    self._accumulate_gene_mean(
                        f'src_t{time_step}{suffix}',
                        gene_ids,
                        z_src_gene,
                        in_both,
                        vocab_size,
                    )

            # ---- store per-cell embeddings + metadata ---------------------
            rows = self._test_cell_rows
            rows.setdefault('z_src_cell', []).append(out['z_src_cell'].cpu())
            rows.setdefault('z_hat_cell', []).append(out['z_hat_cell'].cpu())
            rows.setdefault('z_tgt_cell', []).append(out['z_tgt_cell'].cpu())
            rows.setdefault('z_src_cell_raw', []).append(out['z_src_cell_raw'].cpu())
            rows.setdefault('z_hat_cell_raw', []).append(out['z_hat_cell_raw'].cpu())
            rows.setdefault('z_tgt_cell_raw', []).append(out['z_tgt_cell_raw'].cpu())
            rows.setdefault('time', []).append(
                torch.full((batch_size,), time_step)
            )
            for var in self.var_list:
                key = f'{var}_t{time_step}'
                if key in batch:
                    rows.setdefault(var, []).extend(list(batch[key]))
        return None

    def on_test_epoch_end(self):
        save_dir = os.path.join(self.output_dir, 'embeddings')
        os.makedirs(save_dir, exist_ok=True)

        payload = {
            'local_to_global': self.local_to_global.cpu(),
            'gene_mean': {},      # key like 'hat_t1' -> (vocab, D) mean embedding
            'gene_count': {},     # key like 'hat_t1' -> (vocab,) times queried
        }
        for key, sums in self._test_gene_sums.items():
            counts = self._test_gene_counts[key]
            safe_counts = counts.clamp(min=1.0).unsqueeze(-1)   # avoid 0-division
            payload['gene_mean'][key] = sums / safe_counts
            payload['gene_count'][key] = counts
        for key, rows in self._test_cell_rows.items():
            if len(rows) > 0 and isinstance(rows[0], torch.Tensor):
                payload[key] = torch.cat(rows, dim=0).numpy()
            else:
                payload[key] = rows

        path = os.path.join(save_dir, 'gene_query_jepa_embeddings.pt')
        torch.save(payload, path)
        print(f'Saved Gene-Query JEPA embeddings to {path}')
