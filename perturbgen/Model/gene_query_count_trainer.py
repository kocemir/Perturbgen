"""Count decoder on a frozen Gene-Query JEPA.

Two input modes (``--count-input``):

- ``source_only``: mean of *present* predicted gene embeddings ``z_hat_gene``.
  Queries are the usual mixed 128-slot quiz (source cell only).
- ``copy_target``: same predictor / ``z_hat`` path, but queries are the first
  ``n_queries`` tokens of the true target sequence. Mean over unpadded slots
  only (pad id 0). CLS is kept if it sits in that prefix.

CountHead is the PerturbGen ZINB head. The cell_query slot is never used.

In-silico KO (``--eval-count-ko``) replaces the gene in the source with
``<mask>`` (id 1) by default; ``--ko-replace pad`` restores the old pad-slot
edit. Optional ``resync_quiz`` recomputes query ``src_position`` from the
edited source so knocked-out genes get the trained "absent from source"
code (``-1``) instead of keeping the control rank.

Train:
  python docs/examples/train_gene_query_jepa.py --train-count-decoder true \\
      --jepa-ckpt path/to/last.ckpt --count-input source_only
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from scvi.distributions import ZeroInflatedNegativeBinomial
from torchmetrics import MeanSquaredError

from perturbgen.Model.gene_query_jepa_trainer import (
    SRC_MASK_TOKEN_ID,
    GeneQueryJEPATrainer,
)
from perturbgen.Modules.gene_query_jepa import PAD_TOKEN_ID
from perturbgen.src.jepa_token_maps import apply_id_lookup
from perturbgen.src.metric import compute_emd

COUNT_INPUT_MODES = ('source_only', 'copy_target')


def present_gene_mean(
    z_hat_gene: torch.Tensor,
    is_present: torch.Tensor,
) -> torch.Tensor:
    """Average predicted gene embeddings over present query slots. (B, Q, D) -> (B, D)."""
    mask = is_present.unsqueeze(-1).to(dtype=z_hat_gene.dtype)
    n_present = mask.sum(dim=1).clamp(min=1.0)
    return (z_hat_gene * mask).sum(dim=1) / n_present


def _first_src_position(
    src_local: torch.Tensor,
    gene_ids: torch.Tensor,
    pad_id: int = PAD_TOKEN_ID,
) -> torch.Tensor:
    """First source index of each query gene. Missing / pad -> -1. (B, Q)."""
    match = src_local.unsqueeze(1) == gene_ids.unsqueeze(2)
    found = match.any(dim=-1)
    first = match.to(dtype=torch.long).argmax(dim=-1)
    missing = torch.full_like(first, -1)
    pos = torch.where(found, first, missing)
    return torch.where(gene_ids == pad_id, missing, pos)


def first_target_prefix_quiz(
    src_input_ids_global: torch.Tensor,
    tgt_input_ids_local: torch.Tensor,
    global_to_local: torch.Tensor,
    n_queries: int,
    pad_id: int = PAD_TOKEN_ID,
) -> Dict[str, torch.Tensor]:
    """Fixed-length quiz: target sequence prefix of size ``n_queries``.

    Unpadded prefix slots are ``is_present=True`` so the count decoder can
    average those ``z_hat`` rows and skip pad.
    """
    if n_queries < 1:
        raise ValueError(f'n_queries must be >= 1, got {n_queries}')
    device = tgt_input_ids_local.device
    batch_size, tgt_len = tgt_input_ids_local.shape
    if tgt_len >= n_queries:
        gene_ids = tgt_input_ids_local[:, :n_queries]
    else:
        gene_ids = F.pad(
            tgt_input_ids_local, (0, n_queries - tgt_len), value=int(pad_id)
        )
    arange = torch.arange(n_queries, device=device).unsqueeze(0).expand(batch_size, -1)
    is_present = gene_ids != int(pad_id)
    tgt_position = torch.where(arange < tgt_len, arange, torch.zeros_like(arange))
    src_local = apply_id_lookup(src_input_ids_global, global_to_local)
    src_position = _first_src_position(src_local, gene_ids, pad_id=int(pad_id))
    return {
        'gene_ids_local': gene_ids.long(),
        'is_present': is_present,
        'tgt_position': tgt_position.long(),
        'src_position': src_position.long(),
        'is_mask_ce': torch.zeros(
            batch_size, n_queries, dtype=torch.bool, device=device
        ),
    }


class GeneQueryCountDecoderTrainer(LightningModule):
    def __init__(
        self,
        jepa_ckpt: str,
        n_genes: int = 2000,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        dropout: float = 0.1,
        pred_tps: Optional[List[int]] = None,
        output_dir: str = './count_decoder_runs',
        seed: int = 0,
        count_input: str = 'source_only',
    ):
        super().__init__()
        if count_input not in COUNT_INPUT_MODES:
            raise ValueError(
                f'count_input must be one of {COUNT_INPUT_MODES}, got {count_input!r}'
            )
        self.save_hyperparameters(ignore=['jepa'])
        self.pred_tps = pred_tps if pred_tps is not None else [1, 2, 3]
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_genes = n_genes
        self.output_dir = output_dir
        self.count_input = count_input

        self.jepa = GeneQueryJEPATrainer.load_from_checkpoint(
            jepa_ckpt,
            map_location='cpu',
            strict=False,
        )
        self.jepa.freeze()
        self.jepa.eval()
        self.jepa.lambda_cell = 0.0
        self.jepa.lambda_mask_ce = 0.0
        self.jepa.vicreg_var_coeff = 0.0
        self.jepa.vicreg_cov_coeff = 0.0

        from perturbgen.Modules.transformer import CountHead

        d_model = int(self.jepa.model.d_model)
        self.count_head = CountHead(
            loss_mode='zinb',
            n_genes=n_genes,
            d_model=d_model,
            dropout=dropout,
            use_size_factor=True,
            use_observed_size_factor=True,
        )
        self.theta = torch.nn.Parameter(torch.randn(n_genes, 1) * 0.01)
        self.mse = MeanSquaredError()
        self.train_dict = {'pred_counts': [], 'true_counts': []}
        self.val_dict = {'pred_counts': [], 'true_counts': []}
        print(
            f'GeneQueryCountDecoder: n_genes={n_genes}, d_model={d_model}, '
            f'count_input={count_input}, jepa_ckpt={jepa_ckpt}'
        )

    def _prefix_quiz(self, batch: dict, time_step: int) -> dict:
        return first_target_prefix_quiz(
            src_input_ids_global=batch['src_input_ids'],
            tgt_input_ids_local=batch[f'tgt_input_ids_t{time_step}'],
            global_to_local=self.jepa.global_to_local,
            n_queries=int(self.jepa.n_queries),
        )

    def _cell_vector(self, batch: dict, time_step: int) -> torch.Tensor:
        if self.count_input == 'copy_target':
            result = self.jepa.encode_one_timestep(
                batch, time_step, quiz=self._prefix_quiz(batch, time_step)
            )
        else:
            result = self.jepa._run_one_timestep(batch, time_step)
        return present_gene_mean(
            result['_out']['z_hat_gene'],
            result['_is_present'],
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self.jepa.rng = random.Random(int(self.jepa.hparams.seed) + int(self.global_rank))

    def train(self, mode: bool = True):
        super().train(mode)
        self.jepa.eval()
        return self

    def _dispersion(self, batch: dict, batch_size: int) -> torch.Tensor:
        if batch.get('combined_batch') is None:
            idx = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        else:
            idx = batch['combined_batch'].long().view(-1)
        if idx.dim() == 1:
            idx = idx.unsqueeze(1)
        n_cls = self.theta.size(1)
        onehot = torch.zeros(idx.size(0), n_cls, device=self.device)
        onehot.scatter_(1, idx.clamp(max=n_cls - 1), 1)
        return torch.exp(F.linear(onehot, self.theta))

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        batch_size = batch['src_input_ids'].size(0)
        dispersion = self._dispersion(batch, batch_size)
        total_loss = 0.0
        total_mse = 0.0
        n_times = 0
        for time_step in self.pred_tps:
            if f'tgt_input_ids_t{time_step}' not in batch:
                continue
            if f'tgt_counts_t{time_step}' not in batch:
                raise RuntimeError(
                    f'missing tgt_counts_t{time_step}; count training needs h5ads'
                )
            with torch.no_grad():
                z_cell = self._cell_vector(batch, time_step)
            head_out = self.count_head(z_cell)
            size_factor = batch[f'tgt_size_factor_t{time_step}'].unsqueeze(1)
            mu = head_out['count_mean'] * size_factor.expand_as(head_out['count_mean'])
            true_counts = batch[f'tgt_counts_t{time_step}']
            dist = ZeroInflatedNegativeBinomial(
                mu=mu,
                theta=dispersion,
                zi_logits=head_out['count_dropout'],
            )
            loss = -dist.log_prob(true_counts).sum(dim=-1).mean()
            mse = self.mse(mu, true_counts)
            total_loss = total_loss + loss
            total_mse = total_mse + mse
            n_times += 1
            res_dict = self.train_dict if stage == 'train' else self.val_dict
            res_dict['pred_counts'].append(mu.detach().cpu())
            res_dict['true_counts'].append(true_counts.detach().cpu())
            self.log(
                f'{stage}/loss_t{time_step}',
                loss,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        n_times = max(n_times, 1)
        mean_loss = total_loss / n_times
        mean_mse = total_mse / n_times
        # Match PerturbGen's count-head logging: main loss is summed over target
        # timepoints, while MSE remains averaged over timepoints.
        self.log(
            f'{stage}/loss',
            total_loss,
            on_step=(stage == 'train'),
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f'{stage}/loss_mean_t',
            mean_loss,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f'{stage}/mse',
            mean_mse,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return total_loss

    def training_step(self, batch, *args, **kwargs):
        return self._step(batch, 'train')

    def validation_step(self, batch, *args, **kwargs):
        return self._step(batch, 'val')

    def _log_epoch_emd(self, stage: str) -> None:
        res_dict = self.train_dict if stage == 'train' else self.val_dict
        if not res_dict['pred_counts']:
            return
        pred_counts = torch.cat(res_dict['pred_counts'])
        true_counts = torch.cat(res_dict['true_counts'])
        if len(pred_counts) > 10000:
            random_ids = torch.randint(
                low=0,
                high=len(pred_counts),
                size=(10000,),
                generator=torch.Generator().manual_seed(42),
            ).tolist()
        else:
            random_ids = torch.arange(len(pred_counts)).tolist()
        pred_s = pred_counts[random_ids].float()
        true_s = true_counts[random_ids].float()
        mse = torch.mean((pred_s - true_s) ** 2).item()
        emd = float(compute_emd(pred_s, true_s))
        self.log(
            f'{stage}/emd',
            emd,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if int(self.global_rank) == 0:
            print(
                f'epoch={int(self.current_epoch):03d}  '
                f'{stage}/mse={mse:.4f}  {stage}/emd={emd:.4f}',
                flush=True,
            )
        res_dict['pred_counts'] = []
        res_dict['true_counts'] = []

    def on_train_epoch_end(self):
        self._log_epoch_emd('train')

    def on_validation_epoch_end(self):
        self._log_epoch_emd('val')

    def _decode_present_mean(
        self,
        batch: dict,
        time_step: int,
        z_hat_gene: torch.Tensor,
        is_present: torch.Tensor,
    ) -> torch.Tensor:
        z_cell = present_gene_mean(z_hat_gene, is_present)
        head_out = self.count_head(z_cell)
        size_factor = batch[f'tgt_size_factor_t{time_step}'].unsqueeze(1)
        size_factor = size_factor.to(
            z_cell.device, dtype=head_out['count_mean'].dtype
        )
        return head_out['count_mean'] * size_factor.expand_as(head_out['count_mean'])

    def _resync_quiz_src_position(
        self,
        quiz: dict,
        src_ids_global: torch.Tensor,
    ) -> dict:
        """Keep the same gene questions; refresh ``src_position`` from ``src``.

        After a source KO edit the knocked-out gene is no longer present as
        itself, so ``_first_src_position`` returns ``-1`` (absent-from-source
        PE). Gene ids / present mask / tgt ranks stay identical to control.
        """
        out = {
            key: (value.clone() if torch.is_tensor(value) else value)
            for key, value in quiz.items()
        }
        # Lookup table may live on CPU while KO src is on GPU.
        src_local = apply_id_lookup(
            src_ids_global.detach().cpu(),
            self.jepa.global_to_local.detach().cpu(),
        )
        gene_ids = out['gene_ids_local'].detach().cpu()
        out['src_position'] = _first_src_position(
            src_local,
            gene_ids,
            pad_id=PAD_TOKEN_ID,
        ).to(device=src_ids_global.device)
        return out

    def predict_control_and_ko(
        self,
        batch: dict,
        gene_token_id: int,
        replace_id: int = SRC_MASK_TOKEN_ID,
        resync_quiz: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Control vs KO: replace ``gene_token_id`` in the source with ``replace_id``.

        Default ``replace_id`` is ``<mask>`` (1), same edit as MaskGIT
        ``perturbation_mode=mask``. Pass pad id 0 for the old "delete slot"
        ablation. Quiz gene questions are taken from the control forward.

        If ``resync_quiz`` is True, KO reuses those questions but recomputes
        ``src_position`` on the edited source (knocked-out gene -> ``-1``).
        Default False keeps the old frozen-control-quiz behaviour.
        """
        src = batch['src_input_ids']
        keep = (src == int(gene_token_id)).any(dim=1)
        if not bool(keep.any()):
            return None
        ko_src = src.clone()
        ko_src[ko_src == int(gene_token_id)] = int(replace_id)
        control: Dict[int, torch.Tensor] = {}
        perturbed: Dict[int, torch.Tensor] = {}
        true: Dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for time_step in self.pred_tps:
                if f'tgt_counts_t{time_step}' not in batch:
                    continue
                quiz = (
                    self._prefix_quiz(batch, time_step)
                    if self.count_input == 'copy_target'
                    else None
                )
                z_ctrl = self.jepa.encode_one_timestep(
                    batch, time_step, quiz=quiz, src_ids_global=src
                )
                quiz_ko = z_ctrl['_quiz']
                if resync_quiz:
                    quiz_ko = self._resync_quiz_src_position(quiz_ko, ko_src)
                z_ko = self.jepa.encode_one_timestep(
                    batch,
                    time_step,
                    quiz=quiz_ko,
                    src_ids_global=ko_src,
                )
                control[time_step] = self._decode_present_mean(
                    batch,
                    time_step,
                    z_ctrl['_out']['z_hat_gene'],
                    z_ctrl['_is_present'],
                )[keep]
                perturbed[time_step] = self._decode_present_mean(
                    batch,
                    time_step,
                    z_ko['_out']['z_hat_gene'],
                    z_ko['_is_present'],
                )[keep]
                true[time_step] = batch[f'tgt_counts_t{time_step}'].to(
                    control[time_step].device, dtype=control[time_step].dtype
                )[keep]
        if not control:
            return None
        return {
            'keep': keep,
            'control': control,
            'perturbed': perturbed,
            'true': true,
        }

    def configure_optimizers(self):
        params = list(self.count_head.parameters()) + [self.theta]
        return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
