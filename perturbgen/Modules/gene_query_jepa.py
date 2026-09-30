"""Gene-Query JEPA model: predict target-time GENE embeddings (not a cell vector).

KEEP THIS FILE. This is the only JEPA architecture we train.
Train: docs/examples/train_gene_query_jepa.py (--data toy|full)
Hyperparameter search: docs/examples/run_gene_query_toy_sweep.sh
Honesty metric: val/gene_gap_vs_copy_src must be > 0 (beats copy-source).
Index: docs/examples/GENE_QUERY_JEPA.md

WHY THIS MODEL EXISTS (plain language)
--------------------------------------
A cell-pooled JEPA would squeeze each cell into ONE vector and predict the
future cell vector. That answers "where does the cell go?" but can never
answer "what happens to gene IL1B at 6 hours?" — the gene axis was averaged
away before prediction ever happened.

This model keeps the gene axis. The predictor is asked, gene by gene:

    "Here is a cell at the source time, and here is the name of a gene.
     Tell me what that gene's contextual embedding will look like at
     the target time."

Queries are three kinds per cell:
      ~40%  genes present in BOTH source and target   (shared — shift)
      ~50%  genes present ONLY in the target           (induced)
      ~10%  genes absent from the target               (placeholder; no loss)

THE PICTURE
-----------

      ONLINE SIDE (gets gradients)                EMA SIDE (frozen copy, no grad)

  src tokens (normal) --> encoder                 tgt tokens (90m / 6h / 10h)
             |                                              |
             v                                              v
   gene embeddings H_src                            EMA encoder
   (batch, src_len, 768)                                    |
             |                                              v
             v                                  gene embeddings H_tgt
        PREDICTOR  <-- Q gene queries + 1 cell query        |
   (cross-attention to H_src)                   (batch, tgt_len, 768) stop-grad
             |                                              |
             v                                              v
   predicted gene embeddings  ---- gene loss ---->  true gene embeddings
   z_hat_gene (batch, Q, 768)     (1 - cosine)     z_tgt_gene (batch, Q, 768)
             |                                    (or learned "absent" vector)
   extra learned cell query  ---- cell loss ---->  encoder CLS of H_tgt
   z_hat_cell (batch, 768)        (1 - cosine)     z_tgt_cell (batch, 768)

Design rules we committed to:
  1. TIME lives ONLY in the predictor. The encoders never see the timepoint,
     so their embeddings describe "what the cell/gene is", not "when it is".
  2. No batch population context. Predictor memory is H_src as the encoder
     produced it (pasting the same cell vector onto every gene is a no-op
     for attention and only wasted compute).
  3. The EMA (exponential moving average) target encoder provides the
     prediction targets and receives no gradients — standard JEPA recipe
     to avoid the trivial "everything maps to the same point" solution.
  4. Timepoints are predicted independently (no autoregressive context).

Terminology used everywhere in this file:
  batch    = number of cells in the mini-batch                (short: B)
  src_len  = padded source gene-sequence length               (short: Ls)
  tgt_len  = padded target gene-sequence length               (short: Lt)
  Q        = number of gene queries we ask per cell
  D        = embedding width (768 for the pretrained encoder)
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from perturbgen.Modules.transformer import Block

# Token id 0 means "padding" in BOTH id spaces (global and local).
PAD_TOKEN_ID = 0
# PerturbGen decoder Block: YAML d_ff is attention dim_head, FFN is d_model.
PREDICTOR_DIM_HEAD = 32


class GeneQueryPredictor(nn.Module):
    """Answer gene (and one cell) questions about the future.

    Each query = identity vector + learned time embedding.
    Identity is a gene LUT row for the first Q slots, or the learned
    ``cell_query`` for the extra slot. Queries cross-attend to the source
    cell's gene embeddings and see each other through self-attention.

    arch='torch' (default): ``nn.TransformerDecoder``, FFN = 4*d_model
    (3072 at d_model=768). This is what every run up to 2026-09-22 used.
    arch='block' is the PerturbGen MaskGIT ``Block`` (attention dim_head=32,
    FFN=d_model); only the 2026-09-25 run was trained with it, and it stays
    here so that checkpoint still loads. The frozen scMaskGIT encoder is
    dim_head=96 either way (pretrain ckpt).
    """

    def __init__(
        self,
        d_model: int,
        n_time_steps: int = 3,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
        dim_head: int = PREDICTOR_DIM_HEAD,
        arch: Literal['block', 'torch'] = 'torch',
    ):
        super().__init__()
        if arch not in ('block', 'torch'):
            raise ValueError(f'predictor arch must be block or torch, got {arch!r}')
        self.arch = arch
        self.time_embedding = nn.Embedding(n_time_steps + 1, d_model)
        if arch == 'block':
            self.decoder_block = nn.ModuleList(
                [
                    Block(
                        dim=d_model,
                        num_heads=num_heads,
                        d_ff=dim_head,
                        hidden_size=d_model,
                        dropout=dropout,
                        return_attn=False,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                activation='gelu',
                dropout=dropout,
                batch_first=True,
            )
            self.cross_attention = nn.TransformerDecoder(
                decoder_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(d_model),
            )

    def forward(
        self,
        query_gene_embeddings: torch.Tensor,  # (B, Q or Q+1, D) identity vectors
        time_step: int,                       # scalar: 1, 2, or 3
        source_memory: torch.Tensor,          # (B, Ls, D) source gene embeddings
        source_is_padding: torch.Tensor,      # (B, Ls)    True where src is pad
    ) -> torch.Tensor:
        time_index = torch.tensor(
            [time_step], device=query_gene_embeddings.device
        )
        time_vector = self.time_embedding(time_index)         # (1, D)
        queries = query_gene_embeddings + time_vector         # (B, Q, D)
        if self.arch == 'torch':
            return self.cross_attention(
                tgt=queries,
                memory=source_memory,
                memory_key_padding_mask=source_is_padding,
            )
        predicted = queries
        for dec_layer in self.decoder_block:
            predicted, _, _ = dec_layer(
                x=predicted,
                src_mask=source_is_padding,
                tgt_mask=None,
                enc_output=source_memory,
            )
        return predicted                                      # (B, Q, D)


class GeneQueryJEPA(nn.Module):
    """The full model: online encoder + EMA target encoder + predictor.

    encoder_type:
      'scmaskgit' — the pretrained MaskGIT source encoder (768-wide),
                    early-exited after n_encoder_layers blocks. Default.
      'cell'      — the small randomly-initialised CellEncoder. Mainly for
                    fast tests without a checkpoint.
    """

    def __init__(
        self,
        encoder_type: Literal['scmaskgit', 'cell'] = 'scmaskgit',
        encoder_path: Optional[str] = None,
        n_encoder_layers: int = 6,
        freeze_encoder: bool = False,
        n_time_steps: int = 3,
        predictor_layers: int = 2,
        predictor_heads: int = 8,
        dropout: float = 0.0,
        ema_decay: float = 0.996,
        normalize_latents: bool = True,
        # Only used by the small 'cell' encoder:
        vocab_size: int = 25000,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        d_ff: int = 1024,
        max_seq_length: int = 512,
        cell_pool: str = 'cls',
        cls_token_id: int = 2,
        predictor_arch: Literal['block', 'torch'] = 'torch',
        use_query_src_pos: bool = True,
    ):
        super().__init__()
        self.ema_decay = ema_decay
        self.normalize_latents = normalize_latents
        self.encoder_type = encoder_type
        self.cell_pool = cell_pool
        self.use_query_src_pos = bool(use_query_src_pos)

        # ------------------------------------------------------------------
        # 1) The two encoders: online (trains) and EMA target (frozen copy).
        # ------------------------------------------------------------------
        if encoder_type == 'scmaskgit':
            from perturbgen.Modules.jepa_scmaskgit import SCMaskGITCellEncoder

            self.online_encoder = SCMaskGITCellEncoder(
                encoder_path=encoder_path,
                freeze=freeze_encoder,
                n_encoder_layers=n_encoder_layers,
                cell_pool=cell_pool,
                cls_token_id=cls_token_id,
            )
            self.target_encoder = self.online_encoder.clone_as_ema_target()
            self.d_model = self.online_encoder.d_model
        elif encoder_type == 'cell':
            from perturbgen.Modules.jepa import CellEncoder

            def build_encoder() -> CellEncoder:
                return CellEncoder(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    d_ff=d_ff,
                    max_seq_length=max_seq_length,
                    n_time_steps=n_time_steps + 1,
                    dropout=dropout,
                    cell_pool=cell_pool,
                    cls_token_id=cls_token_id,
                )

            self.online_encoder = build_encoder()
            self.target_encoder = build_encoder()
            self.target_encoder.load_state_dict(self.online_encoder.state_dict())
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad = False
            if freeze_encoder:
                for parameter in self.online_encoder.parameters():
                    parameter.requires_grad = False
            self.d_model = d_model
        else:
            raise ValueError(f'unknown encoder_type: {encoder_type!r}')

        # ------------------------------------------------------------------
        # 2) The time-conditioned gene-query predictor (design rule 1:
        #    this is the ONLY place the model learns about time).
        # ------------------------------------------------------------------
        self.predictor_arch = predictor_arch
        self.predictor = GeneQueryPredictor(
            d_model=self.d_model,
            n_time_steps=n_time_steps,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
            dropout=dropout,
            arch=predictor_arch,
        )

        # ------------------------------------------------------------------
        # 3) The learned "this gene is absent" answer. When we query a gene
        #    that is NOT expressed at the target time, the correct answer is
        #    this vector.
        # ------------------------------------------------------------------
        self.absent_gene_embedding = nn.Parameter(
            torch.randn(self.d_model) * 0.02
        )
        # Extra predictor query (not a gene LUT row, not encoder token id 2).
        self.cell_query = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        # Mask-CE query: no gene identity; shares the src-rank table below.
        self.mask_query = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        # Learnable source-rank PE for EVERY gene query (not target rank).
        # Indices 0..max_seq_length-1 = rank in the source sequence.
        # Last index = "gene absent from source" (src_position < 0).
        # Kept small (std=0.02) so it does not drown gene identity / time.
        pos_slots = max(int(max_seq_length), 2048) + 1
        self.query_src_pos_embedding = nn.Embedding(pos_slots, self.d_model)
        nn.init.normal_(self.query_src_pos_embedding.weight, std=0.02)
        # Alias for older mask-CE checkpoints / trainer freeze paths.
        self.mask_src_pos_embedding = self.query_src_pos_embedding

    # ----------------------------------------------------------------------
    # Small helpers
    # ----------------------------------------------------------------------
    def _gene_identity_table(self, encoder: nn.Module) -> nn.Embedding:
        """The gene-id -> vector lookup table living inside an encoder."""
        if self.encoder_type == 'scmaskgit':
            return encoder.model.token_embedding
        return encoder.token_embedding

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalise the last axis so cosine losses are well behaved."""
        if self.normalize_latents:
            return F.normalize(x, dim=-1)
        return x

    def _src_rank_pe(self, query_src_position: torch.Tensor) -> torch.Tensor:
        """Learnable PE from source expression rank. (B, Q) -> (B, Q, D).

        ``query_src_position < 0`` means the gene is absent from the source
        cell; those slots use the last embedding row (dedicated absent code).
        """
        n_pos = self.query_src_pos_embedding.num_embeddings
        absent_idx = n_pos - 1
        safe = query_src_position.clone()
        safe = torch.where(safe < 0, torch.full_like(safe, absent_idx), safe)
        safe = safe.clamp(0, absent_idx)
        return self.query_src_pos_embedding(safe)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """Move the EMA encoder a tiny step towards the online encoder.

        new_target = decay * old_target + (1 - decay) * online
        Called once after every optimiser step.
        """
        for online_param, target_param in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_param.data.mul_(self.ema_decay)
            target_param.data.add_(online_param.data, alpha=1.0 - self.ema_decay)

    def unfreeze_online_encoder(self) -> None:
        """Unfreeze the student encoder used layers (EMA target stays frozen)."""
        if hasattr(self.online_encoder, 'unfreeze_used_layers'):
            self.online_encoder.unfreeze_used_layers()
            return
        for parameter in self.online_encoder.parameters():
            parameter.requires_grad = True

    def freeze_online_encoder(self) -> None:
        if hasattr(self.online_encoder, 'freeze_used_layers'):
            self.online_encoder.freeze_used_layers()
            return
        for parameter in self.online_encoder.parameters():
            parameter.requires_grad = False

    # ----------------------------------------------------------------------
    # The forward pass for ONE target timepoint.
    # The trainer calls this once per timepoint (t = 1, 2, 3).
    # ----------------------------------------------------------------------
    def forward_one_timestep(
        self,
        src_input_ids: torch.Tensor,      # (B, Ls) source gene tokens
        tgt_input_ids: torch.Tensor,      # (B, Lt) target gene tokens
        query_gene_ids: torch.Tensor,     # (B, Q)  which genes we ask about
        query_is_present: torch.Tensor,   # (B, Q)  True if gene really is in tgt
        query_tgt_position: torch.Tensor, # (B, Q)  where in tgt_input_ids it sits
        time_step: int,                   # which target timepoint (1, 2 or 3)
        query_is_mask_ce: Optional[torch.Tensor] = None,  # (B, Q) mask-CE slots
        query_src_position: Optional[torch.Tensor] = None,  # (B, Q) src index
    ) -> Dict[str, torch.Tensor]:
        """Run the whole diagram once. Returns every arrow's endpoint.

        All id tensors must already be in the id space the encoder expects
        (the trainer takes care of converting; see the trainer docstring).
        For absent queries, query_tgt_position is 0 and simply unused.
        """
        # ---- Step 1: online encoder reads the source cell. -----------------
        online_out = self.online_encoder(src_input_ids, time_step=0)
        h_src = online_out['token_embedding']           # (B, Ls, D)
        z_src_cell = online_out['cell_embedding']       # (B, D)

        # ---- Step 2: predictor memory is H_src (no batch context). ---------
        source_is_padding = src_input_ids == PAD_TOKEN_ID   # (B, Ls)

        # ---- Step 3: EMA encoder reads the target cell (no gradients). -----
        with torch.no_grad():
            target_out = self.target_encoder(tgt_input_ids, time_step=0)
            h_tgt = target_out['token_embedding']       # (B, Lt, D)
            z_tgt_cell = target_out['cell_embedding']   # (B, D)

        # ---- Step 4: true answer = that gene's row in H_tgt (or absent). ---
        batch_size, n_queries = query_gene_ids.shape
        position_as_index = query_tgt_position.unsqueeze(-1)     # (B, Q, 1)
        position_as_index = position_as_index.expand(-1, -1, self.d_model)
        true_gene_embedding = torch.gather(h_tgt, dim=1, index=position_as_index)
        absent_answer = self.absent_gene_embedding.expand(
            batch_size, n_queries, self.d_model
        )
        is_present = query_is_present.unsqueeze(-1)     # (B, Q, 1)
        z_tgt_gene = torch.where(is_present, true_gene_embedding, absent_answer)

        # ---- Step 5: Q gene identities + optional src-rank PE + 1 cell query.
        identity_table = self._gene_identity_table(self.online_encoder)
        identity = identity_table(query_gene_ids)              # (B, Q, D)
        if query_is_mask_ce is not None and bool(query_is_mask_ce.any()):
            # Drop gene identity for mask-CE slots; keep only mask_query (+ PE).
            mask_q = self.mask_query.expand(batch_size, n_queries, -1)
            identity = torch.where(
                query_is_mask_ce.unsqueeze(-1), mask_q, identity
            )
        if (
            self.use_query_src_pos
            and query_src_position is not None
        ):
            identity = identity + self._src_rank_pe(query_src_position)
        cell_query = self.cell_query.expand(batch_size, 1, -1)  # (B, 1, D)
        queries = torch.cat([identity, cell_query], dim=1)     # (B, Q+1, D)
        predicted = self.predictor(
            query_gene_embeddings=queries,
            time_step=time_step,
            source_memory=h_src,
            source_is_padding=source_is_padding,
        )                                               # (B, Q+1, D)
        z_hat_gene_raw = predicted[:, :-1]              # (B, Q, D)
        z_hat_cell_raw = predicted[:, -1]               # (B, D)
        z_tgt_gene_raw = z_tgt_gene
        z_tgt_cell_raw = z_tgt_cell
        z_src_cell_raw = z_src_cell
        h_src_raw = h_src

        # ---- Step 7: normalise copies that enter cosine / CE. -------------
        z_hat_gene = self._maybe_normalize(z_hat_gene_raw)
        z_tgt_gene = self._maybe_normalize(z_tgt_gene_raw)
        z_hat_cell = self._maybe_normalize(z_hat_cell_raw)
        z_tgt_cell = self._maybe_normalize(z_tgt_cell_raw)
        z_src_cell = self._maybe_normalize(z_src_cell_raw)

        # ---- Step 8 (metrics only): the "no learning" baselines. -----------
        with torch.no_grad():
            static_table = self._gene_identity_table(self.target_encoder)
            z_static_gene = self._maybe_normalize(
                static_table(query_gene_ids)
            )                                           # (B, Q, D)

        return {
            'z_hat_gene': z_hat_gene,
            'z_tgt_gene': z_tgt_gene,
            'z_hat_gene_raw': z_hat_gene_raw,
            'z_tgt_gene_raw': z_tgt_gene_raw,
            'z_hat_cell': z_hat_cell,
            'z_tgt_cell': z_tgt_cell,
            'z_src_cell': z_src_cell,
            'z_hat_cell_raw': z_hat_cell_raw,
            'z_tgt_cell_raw': z_tgt_cell_raw,
            'z_src_cell_raw': z_src_cell_raw,
            'h_src': self._maybe_normalize(h_src_raw),
            'h_src_raw': h_src_raw,
            'z_static_gene': z_static_gene,
        }
