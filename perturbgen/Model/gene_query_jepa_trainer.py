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
      ~50%  genes present in BOTH source and target   (shared — shift)
      ~30%  genes present ONLY in the target           (induced)
      ~20%  genes absent from the target               (decoys; answer is "absent")
3. The model predicts each queried gene's embedding at the target time.
   Timepoints are independent (no autoregressive context).
4. Losses:  gene loss   = 1 - cosine(predicted, true)   ... main signal
            cell loss   = 1 - cosine(pooled prediction, pooled target)
            contrastive = InfoNCE on present queries (anti-collapse)
            VICReg      = optional variance/covariance penalties
   total = lambda_gene * gene_loss + lambda_cell * cell_loss
           + lambda_contr * contrastive (+ optional VICReg)
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
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule

from perturbgen.Modules.gene_query_jepa import GeneQueryJEPA
from perturbgen.src.jepa_metrics import vicreg_var_cov
from perturbgen.src.jepa_token_maps import apply_id_lookup, maybe_load_maps

# In the LOCAL id space: 0=pad, 1=mask, 2=cls, 3=eos. Genes start here.
FIRST_REAL_GENE_LOCAL_ID = 4


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
) -> Dict[str, torch.Tensor]:
    """Build the per-cell quiz sheet. Returns CPU tensors, all shaped (B, Q).

    Three pools: shared (~50%), target-only (~30%), absent decoys (the rest).
    query_mode / shared_max_queries are accepted for old CLI flags and ignored.
    """
    del query_mode, shared_max_queries
    batch_size = tgt_input_ids_local.size(0)

    src_rows = src_input_ids_global.detach().cpu().tolist()
    tgt_rows = tgt_input_ids_local.detach().cpu().tolist()
    global_to_local_list = global_to_local.detach().cpu().tolist()

    all_genes_set = set(all_gene_local_ids)
    n_shared_wanted = round(frac_shared * n_queries)
    n_tgt_only_wanted = round(frac_tgt_only * n_queries)

    out_gene_ids: List[List[int]] = []
    out_is_present: List[List[bool]] = []
    out_tgt_position: List[List[int]] = []
    out_src_position: List[List[int]] = []

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
        for gene in chosen_shared + chosen_tgt_only:
            row_gene_ids.append(gene)
            row_is_present.append(True)
            row_tgt_position.append(tgt_gene_position[gene])
            row_src_position.append(src_gene_position.get(gene, -1))
        for gene in chosen_absent:
            row_gene_ids.append(gene)
            row_is_present.append(False)
            row_tgt_position.append(0)
            row_src_position.append(-1)
        while len(row_gene_ids) < n_queries:
            row_gene_ids.append(all_gene_local_ids[0])
            row_is_present.append(False)
            row_tgt_position.append(0)
            row_src_position.append(-1)

        out_gene_ids.append(row_gene_ids)
        out_is_present.append(row_is_present)
        out_tgt_position.append(row_tgt_position)
        out_src_position.append(row_src_position)

    return {
        'gene_ids_local': torch.tensor(out_gene_ids, dtype=torch.long),
        'is_present': torch.tensor(out_is_present, dtype=torch.bool),
        'tgt_position': torch.tensor(out_tgt_position, dtype=torch.long),
        'src_position': torch.tensor(out_src_position, dtype=torch.long),
    }


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


class GeneQueryJEPATrainer(LightningModule):
    """PyTorch-Lightning wrapper: batches in, losses/metrics/checkpoints out."""

    def __init__(
        self,
        # ---- encoder ----
        jepa_encoder: str = 'scmaskgit',        # 'scmaskgit' or 'cell'
        encoder_path: Optional[str] = None,     # pretrained ckpt (scmaskgit)
        jepa_encoder_layers: int = 6,           # early-exit depth (scmaskgit)
        freeze_jepa_encoder: bool = False,
        # ---- query sampling ----
        n_queries: int = 64,
        frac_shared: float = 0.5,
        frac_tgt_only: float = 0.3,
        query_mode: str = 'mixed',
        shared_max_queries: int = 0,
        predictor_layers: int = 2,
        # ---- loss weights (gene-primary, cell-auxiliary) ----
        lambda_gene: float = 1.0,
        lambda_cell: float = 0.1,
        lambda_contrastive: float = 0.0,
        contrastive_temperature: float = 0.1,
        vicreg_var_coeff: float = 0.0,
        vicreg_cov_coeff: float = 0.0,
        vicreg_gamma: Optional[float] = None,
        # ---- optimisation ----
        lr: float = 1e-4,
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
        max_seq_length: int = 2048,
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
        self.lr = lr
        self.weight_decay = weight_decay
        self.jepa_encoder = jepa_encoder
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
            freeze_encoder=freeze_jepa_encoder,
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
        )
        print(
            f'GeneQueryJEPA: encoder={jepa_encoder}, d_model={self.model.d_model}, '
            f'n_queries={n_queries} (shared {frac_shared:.0%} / '
            f'tgt-only {frac_tgt_only:.0%} / absent rest), '
            f'predictor_layers={predictor_layers}, '
            f'lambda_gene={lambda_gene}, lambda_cell={lambda_cell}, '
            f'lambda_contr={self.lambda_contrastive}, '
            f'tau={self.contrastive_temperature}, '
            f'vicreg_var={self.vicreg_var_coeff}, '
            f'vicreg_cov={self.vicreg_cov_coeff}'
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
            n_queries=self.n_queries,
            frac_shared=self.frac_shared,
            frac_tgt_only=self.frac_tgt_only,
            rng=self.rng,
        )
        device = src_ids_global.device
        gene_ids_local = quiz['gene_ids_local'].to(device)          # (B, Q)
        is_present = quiz['is_present'].to(device)                  # (B, Q)
        tgt_position = quiz['tgt_position'].to(device)              # (B, Q)
        src_position = quiz['src_position'].to(device)              # (B, Q)

        # 2) Convert every id tensor into the encoder's id space.
        out = self.model.forward_one_timestep(
            src_input_ids=self._src_ids_to_encoder_space(src_ids_global),
            tgt_input_ids=self._local_ids_to_encoder_space(tgt_ids_local),
            query_gene_ids=self._local_ids_to_encoder_space(gene_ids_local),
            query_is_present=is_present,
            query_tgt_position=tgt_position,
            time_step=time_step,
        )

        # 3) Losses. -------------------------------------------------------
        # Gene loss: every query counts — present vs EMA, absent vs null vec.
        per_query_distance = cosine_distance(
            out['z_hat_gene'], out['z_tgt_gene']
        )                                                           # (B, Q)
        gene_loss = per_query_distance.mean()

        # Cell loss: pooled prediction vs pooled target.
        cell_loss = cosine_distance(
            out['z_hat_cell'], out['z_tgt_cell']
        ).mean()

        if self.lambda_contrastive > 0.0:
            contrastive_loss = info_nce_present_genes(
                z_hat=out['z_hat_gene'],
                z_tgt=out['z_tgt_gene'],
                is_present=is_present,
                temperature=self.contrastive_temperature,
            )
        else:
            contrastive_loss = gene_loss.new_zeros(())

        embedding_dim = out['z_hat_gene'].size(-1)
        if self.vicreg_var_coeff > 0.0 or self.vicreg_cov_coeff > 0.0:
            if self.vicreg_gamma is not None:
                gamma = self.vicreg_gamma
            elif self.hparams.normalize_latents:
                gamma = 1.0 / (embedding_dim ** 0.5)
            else:
                gamma = 1.0
            flat_gene_predictions = out['z_hat_gene'].reshape(-1, embedding_dim)
            var_gene, cov_gene = vicreg_var_cov(
                flat_gene_predictions, gamma=gamma
            )
            var_cell, cov_cell = vicreg_var_cov(out['z_hat_cell'], gamma=gamma)
            vicreg_var = (var_gene + var_cell) / 2.0
            vicreg_cov = (cov_gene + cov_cell) / 2.0
        else:
            vicreg_var = gene_loss.new_zeros(())
            vicreg_cov = gene_loss.new_zeros(())

        total_loss = (
            self.lambda_gene * gene_loss
            + self.lambda_cell * cell_loss
            + self.lambda_contrastive * contrastive_loss
            + self.vicreg_var_coeff * vicreg_var
            + self.vicreg_cov_coeff * vicreg_cov
        )

        # 4) Honesty metrics (no gradients needed). --------------------------
        with torch.no_grad():
            per_query_cosine = 1.0 - per_query_distance             # (B, Q)
            gene_cos_pred = masked_mean(per_query_cosine, is_present)

            static_cosine = F.cosine_similarity(
                out['z_static_gene'], out['z_tgt_gene'], dim=-1
            )
            gene_cos_static = masked_mean(static_cosine, is_present)

            in_both = is_present & (src_position >= 0)
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
            'vicreg_var': vicreg_var,
            'vicreg_cov': vicreg_cov,
            'gene_cos_pred': gene_cos_pred,
            'gene_gap_vs_static': gene_cos_pred - gene_cos_static,
            'gene_gap_vs_copy_src': gene_cos_pred_shared - gene_cos_copy_src,
            'cell_cos_pred': cell_cos_pred,
            'cell_gap_vs_copy_src': cell_cos_pred - cell_cos_copy_src,
            '_out': out,
            '_gene_ids_local': gene_ids_local,
            '_is_present': is_present,
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
            'vicreg_var', 'vicreg_cov',
            'gene_cos_pred', 'gene_gap_vs_static', 'gene_gap_vs_copy_src',
            'cell_cos_pred', 'cell_gap_vs_copy_src',
        ]
        averaged = {}
        for key in keys_to_log:
            values = [r[key] for r in results_per_timestep]
            averaged[key] = torch.stack(values).mean()
        for key, value in averaged.items():
            show_in_progress_bar = key in (
                'total_loss', 'gene_gap_vs_copy_src'
            )
            self.log(
                f'{stage}/{key}',
                value,
                on_step=(stage == 'train' and key == 'total_loss'),
                on_epoch=True,
                prog_bar=show_in_progress_bar,
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

    def configure_optimizers(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable, lr=self.lr, weight_decay=self.weight_decay
        )
        return {'optimizer': optimizer, 'monitor': 'val/total_loss'}

    # ------------------------------------------------------------------
    # Test = dump embeddings for downstream analysis (notebook 05 style).
    #
    # Per (gene, timepoint) we accumulate a RUNNING MEAN of the predicted
    # and the true (EMA) gene embeddings. Queries are random per cell, but
    # over a full test epoch each gene is asked hundreds of times, so the
    # means are well covered. Cell-level embeddings are stored per cell.
    # ------------------------------------------------------------------
    def test_step(self, batch, *args, **kwargs):
        batch_size = batch['src_input_ids'].size(0)
        vocab_size = int(self.local_to_global.numel())
        for time_step in self.pred_tps:
            if f'tgt_input_ids_t{time_step}' not in batch:
                continue
            result = self._run_one_timestep(batch, time_step)
            out = result['_out']
            gene_ids = result['_gene_ids_local']          # (B, Q) LOCAL ids
            is_present = result['_is_present']            # (B, Q)
            src_position = result['_src_position']        # (B, Q)

            # ---- accumulate per-gene running sums (present queries only) --
            for name, embeddings in (
                ('hat', out['z_hat_gene']),
                ('tgt', out['z_tgt_gene']),
            ):
                key = f'{name}_t{time_step}'
                if key not in self._test_gene_sums:
                    self._test_gene_sums[key] = torch.zeros(
                        vocab_size, self.model.d_model
                    )
                    self._test_gene_counts[key] = torch.zeros(vocab_size)
                flat_ids = gene_ids[is_present].cpu()
                flat_embeddings = embeddings[is_present].cpu()
                self._test_gene_sums[key].index_add_(
                    0, flat_ids, flat_embeddings.float()
                )
                self._test_gene_counts[key].index_add_(
                    0, flat_ids, torch.ones(flat_ids.numel())
                )

            in_both = is_present & (src_position >= 0)
            if bool(in_both.any()):
                key = f'src_t{time_step}'
                if key not in self._test_gene_sums:
                    self._test_gene_sums[key] = torch.zeros(
                        vocab_size, self.model.d_model
                    )
                    self._test_gene_counts[key] = torch.zeros(vocab_size)
                dim = out['h_src'].size(-1)
                safe_pos = src_position.clamp(min=0)
                index = safe_pos.unsqueeze(-1).expand(-1, -1, dim)
                z_src_gene = torch.gather(out['h_src'], dim=1, index=index)
                flat_ids = gene_ids[in_both].cpu()
                flat_embeddings = z_src_gene[in_both].cpu()
                self._test_gene_sums[key].index_add_(
                    0, flat_ids, flat_embeddings.float()
                )
                self._test_gene_counts[key].index_add_(
                    0, flat_ids, torch.ones(flat_ids.numel())
                )

            # ---- store per-cell embeddings + metadata ---------------------
            rows = self._test_cell_rows
            rows.setdefault('z_src_cell', []).append(out['z_src_cell'].cpu())
            rows.setdefault('z_hat_cell', []).append(out['z_hat_cell'].cpu())
            rows.setdefault('z_tgt_cell', []).append(out['z_tgt_cell'].cpu())
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
