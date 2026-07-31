"""Lightning trainers for cell-trajectory JEPA (Phases A and D)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule

from perturbgen.Modules.jepa import CellTrajectoryJEPA, JEPACountDecoder
from perturbgen.src.jepa_metrics import (
    latent_collapse_stats,
    pairwise_latent_mse,
    trajectory_baselines,
)
from perturbgen.Modules.jepa import modify_ckpt_state_dict


class JEPATrainer(LightningModule):
    """Phase A: train CellTrajectoryJEPA with latent MSE + collapse monitors."""

    def __init__(
        self,
        tgt_vocab_size: int = 25000,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        d_ff: int = 1024,
        max_seq_length: int = 2048,
        dropout: float = 0.0,
        weight_decay: float = 1e-4,
        lr: float = 1e-4,
        pred_tps: Optional[List[int]] = None,
        n_total_tps: int = 3,
        ema_decay: float = 0.996,
        normalize_latents: bool = True,
        loss_type: Literal['mse', 'smooth_l1'] = 'mse',
        pos_encoding_mode: Literal[
            'time_pos_sin', 'comb_sin', 'sin_learnt', 'time_pos_learnt'
        ] = 'time_pos_sin',
        ckpt_masking_path: Optional[str] = None,
        mapping_dict_path: Optional[str] = None,
        tokenid_to_rowid_path: Optional[str] = None,
        output_dir: str = './T_perturb/res/jepa/',
        var_list: Optional[List[str]] = None,
        seed: int = 42,
        # Accepted for CLI compatibility with train.py shared kwargs; unused here.
        encoder: Optional[str] = None,
        encoder_path: Optional[str] = None,
        condition_dict: Optional[Dict] = None,
        context_tps: Optional[List[int]] = None,
        mask_scheduler: Optional[str] = None,
        temperature: Optional[float] = None,
        iterations: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        pred_tps = pred_tps if pred_tps is not None else [1, 2, 3]
        self.pred_tps = pred_tps
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.output_dir = output_dir
        self.var_list = var_list or []
        os.makedirs(output_dir, exist_ok=True)

        self.model = CellTrajectoryJEPA(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            n_total_tps=n_total_tps,
            pred_tps=pred_tps,
            dropout=dropout,
            ema_decay=ema_decay,
            pos_encoding_mode=pos_encoding_mode,
            normalize_latents=normalize_latents,
        )
        if ckpt_masking_path is not None and os.path.isfile(ckpt_masking_path):
            loaded = self.model.load_token_embedding_from_masking_ckpt(
                ckpt_masking_path
            )
            print(f'JEPA: loaded from masking ckpt: {loaded}')

        self._train_emb_buffer: List[torch.Tensor] = []
        self.test_embeddings: Dict[str, List[Any]] = {
            'z_src': [],
            'z_tgt': [],
            'z_hat': [],
            'time': [],
        }
        for var in self.var_list:
            self.test_embeddings[var] = []

    def _latent_loss(
        self, z_hat: torch.Tensor, z_tgt: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss(z_hat, z_tgt)
        return F.mse_loss(z_hat, z_tgt)

    def forward(self, batch, pred_tps: Optional[List[int]] = None):
        tgt_input_id_dict = {
            f'tgt_input_ids_t{t}': batch[f'tgt_input_ids_t{t}']
            for t in self.pred_tps
            if f'tgt_input_ids_t{t}' in batch
        }
        return self.model(
            src_input_ids=batch['src_input_ids'],
            tgt_input_id_dict=tgt_input_id_dict,
            pred_tps=pred_tps,
        )

    def configure_optimizers(self):
        params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        try:
            optimizer = optim.AdamW(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
                fused=torch.cuda.is_available(),
            )
        except (RuntimeError, ValueError, TypeError):
            optimizer = optim.AdamW(
                params, lr=self.lr, weight_decay=self.weight_decay
            )
        return {
            'optimizer': optimizer,
            'monitor': 'val/jepa_loss',
        }

    def _step(self, batch, stage: str) -> torch.Tensor:
        outputs = self.forward(batch)
        losses = []
        z_hats = []
        z_tgts = []
        for t, out in outputs.items():
            loss_t = self._latent_loss(out['z_hat'], out['z_tgt'])
            losses.append(loss_t)
            z_hats.append(out['z_hat'])
            z_tgts.append(out['z_tgt'])
            self.log(
                f'{stage}/jepa_loss_t{t}',
                loss_t,
                on_step=stage == 'train',
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch['src_input_ids'].size(0),
            )
        loss = torch.stack(losses).mean()
        z_hat_cat = torch.cat(z_hats, dim=0)
        z_tgt_cat = torch.cat(z_tgts, dim=0)
        stats = latent_collapse_stats(z_tgt_cat)
        metrics = pairwise_latent_mse(z_hat_cat, z_tgt_cat)
        self.log(
            f'{stage}/jepa_loss',
            loss,
            on_step=stage == 'train',
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch['src_input_ids'].size(0),
        )
        self.log(
            f'{stage}/latent_cosine',
            metrics['cosine'],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch['src_input_ids'].size(0),
        )
        for k, v in stats.items():
            self.log(
                f'{stage}/collapse_{k}',
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=(k == 'std_mean'),
                sync_dist=True,
                batch_size=batch['src_input_ids'].size(0),
            )
        if stage == 'train':
            self._train_emb_buffer.append(z_tgt_cat.detach().cpu())
        if stage == 'val' and outputs:
            first = outputs[next(iter(outputs))]
            bases = trajectory_baselines(
                first['z_src'], first['z_tgt'], first['z_hat']
            )
            self.log(
                'val/baseline_identity_mse',
                bases['identity']['mse'],
                on_epoch=True,
                sync_dist=True,
                batch_size=batch['src_input_ids'].size(0),
            )
            self.log(
                'val/baseline_jepa_mse',
                bases['jepa']['mse'],
                on_epoch=True,
                sync_dist=True,
                batch_size=batch['src_input_ids'].size(0),
            )
        return loss

    def training_step(self, batch, *args, **kwargs):
        loss = self._step(batch, 'train')
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.model.update_target_encoder()

    def validation_step(self, batch, *args, **kwargs):
        return self._step(batch, 'val')

    def test_step(self, batch, *args, **kwargs):
        outputs = self.forward(batch)
        for t, out in outputs.items():
            self.test_embeddings['z_src'].append(out['z_src'].detach().cpu())
            self.test_embeddings['z_tgt'].append(out['z_tgt'].detach().cpu())
            self.test_embeddings['z_hat'].append(out['z_hat'].detach().cpu())
            self.test_embeddings['time'].append(
                torch.full((out['z_src'].size(0),), t)
            )
            for var in self.var_list:
                key = f'{var}_t{t}'
                if key in batch:
                    self.test_embeddings[var].extend(list(batch[key]))
        return self._step(batch, 'test')

    def on_test_epoch_end(self):
        save_dir = os.path.join(self.output_dir, 'embeddings')
        os.makedirs(save_dir, exist_ok=True)
        payload = {}
        for k, v in self.test_embeddings.items():
            if len(v) == 0:
                continue
            if isinstance(v[0], torch.Tensor):
                payload[k] = torch.cat(v, dim=0).numpy()
            else:
                payload[k] = v
        path = os.path.join(save_dir, 'jepa_cell_embeddings.pt')
        torch.save(payload, path)
        print(f'Saved JEPA embeddings to {path}')


class JEPADecoderTrainer(LightningModule):
    """Phase D: train a count head on JEPA predicted latents."""

    def __init__(
        self,
        tgt_vocab_size: int = 25000,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        d_ff: int = 1024,
        max_seq_length: int = 2048,
        dropout: float = 0.0,
        weight_decay: float = 1e-3,
        lr: float = 1e-3,
        pred_tps: Optional[List[int]] = None,
        n_total_tps: int = 3,
        n_genes: int = 2000,
        ema_decay: float = 0.996,
        normalize_latents: bool = True,
        pos_encoding_mode: Literal[
            'time_pos_sin', 'comb_sin', 'sin_learnt', 'time_pos_learnt'
        ] = 'time_pos_sin',
        ckpt_jepa_path: Optional[str] = None,
        ckpt_masking_path: Optional[str] = None,
        freeze_jepa: bool = True,
        use_predicted_latent: bool = True,
        output_dir: str = './T_perturb/res/jepa/',
        mapping_dict_path: Optional[str] = None,
        tokenid_to_rowid_path: Optional[str] = None,
        seed: int = 42,
        encoder: Optional[str] = None,
        encoder_path: Optional[str] = None,
        condition_dict: Optional[Dict] = None,
        context_tps: Optional[List[int]] = None,
        mask_scheduler: Optional[str] = None,
        temperature: Optional[float] = None,
        iterations: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        pred_tps = pred_tps if pred_tps is not None else [1, 2, 3]
        self.pred_tps = pred_tps
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_predicted_latent = use_predicted_latent
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        jepa = CellTrajectoryJEPA(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            n_total_tps=n_total_tps,
            pred_tps=pred_tps,
            dropout=dropout,
            ema_decay=ema_decay,
            pos_encoding_mode=pos_encoding_mode,
            normalize_latents=normalize_latents,
        )
        if ckpt_masking_path is not None and os.path.isfile(ckpt_masking_path):
            jepa.load_token_embedding_from_masking_ckpt(ckpt_masking_path)
        if ckpt_jepa_path is not None and os.path.isfile(ckpt_jepa_path):
            checkpoint = torch.load(ckpt_jepa_path, map_location='cpu')
            state = modify_ckpt_state_dict(checkpoint, 'model.')
            missing, unexpected = jepa.load_state_dict(state, strict=False)
            if missing:
                print(f'JEPADecoder: missing keys: {missing[:8]}...')
            if unexpected:
                print(f'JEPADecoder: unexpected keys: {unexpected[:8]}...')

        self.decoder = JEPACountDecoder(
            jepa=jepa,
            n_genes=n_genes,
            d_model=d_model,
            dropout=dropout,
            freeze_jepa=freeze_jepa,
        )
        self.mse = nn.MSELoss()

    def forward(self, batch):
        tgt_input_id_dict = {
            f'tgt_input_ids_t{t}': batch[f'tgt_input_ids_t{t}']
            for t in self.pred_tps
            if f'tgt_input_ids_t{t}' in batch
        }
        return self.decoder(
            src_input_ids=batch['src_input_ids'],
            tgt_input_id_dict=tgt_input_id_dict,
            pred_tps=self.pred_tps,
            use_predicted_latent=self.use_predicted_latent,
        )

    def configure_optimizers(self):
        params = [p for p in self.decoder.parameters() if p.requires_grad]
        return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _step(self, batch, stage: str):
        preds = self.forward(batch)
        losses = []
        for t, pred in preds.items():
            count_key = f'tgt_counts_t{t}'
            if count_key not in batch or batch[count_key] is None:
                continue
            target = batch[count_key]
            if hasattr(target, 'toarray'):
                target = target.toarray()
            target = torch.as_tensor(
                target, device=pred.device, dtype=pred.dtype
            )
            # log1p normalize for stable MSE
            target = torch.log1p(target)
            pred_log = torch.log1p(pred)
            # align gene dim if needed
            g = min(pred_log.size(-1), target.size(-1))
            loss_t = self.mse(pred_log[..., :g], target[..., :g])
            losses.append(loss_t)
            self.log(
                f'{stage}/count_mse_t{t}',
                loss_t,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch['src_input_ids'].size(0),
            )
        if not losses:
            # no counts in batch — zero loss that still wires the graph
            loss = sum(p.sum() for p in preds.values()) * 0.0
        else:
            loss = torch.stack(losses).mean()
        self.log(
            f'{stage}/mse',
            loss,
            on_step=stage == 'train',
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch['src_input_ids'].size(0),
        )
        return loss

    def training_step(self, batch, *args, **kwargs):
        return self._step(batch, 'train')

    def validation_step(self, batch, *args, **kwargs):
        return self._step(batch, 'val')
