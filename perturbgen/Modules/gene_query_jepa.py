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

Queries are only genes present at the target time: shared (also in the
source / normal cell) and target-only (induced). No absent decoys.
Each query gets its own time vector. Shared: DCT shift of H_src[g].
Target-only: DCT gain on the gene-name vector e(g).

THE PICTURE (matches the architecture PDF + our discussion)
-----------------------------------------------------------

      ONLINE SIDE (gets gradients)                EMA SIDE (frozen copy, no grad)

  src tokens (normal) --> encoder                 tgt tokens (90m / 6h / 10h)
             |                                              |
             v                                              v
   gene embeddings H_src                            EMA encoder
   (batch, src_len, 768)                                    |
             |                                              v
   [H_src ; population context] --> linear      gene embeddings H_tgt
             |         (768*2 -> 768)           (batch, tgt_len, 768) stop-grad
             v                                              |
        PREDICTOR  <-- per-query time:                      |
   (transformer decoder)  shared:  H_src[g] + DCT_shift_g(t)|
                          induced: e(g) ⊙ σ(DCT_gain_g(t))  |
             |                                              |
             v                                              v
   predicted gene embeddings  ---- gene loss ---->  true gene embeddings
   z_hat_gene (batch, Q, 768)     (1 - cosine)     z_tgt_gene (batch, Q, 768)
             |
             v
   mean over present queries  ---- cell loss ---->  mean over H_tgt
   z_hat_cell (batch, 768)        (1 - cosine)     z_tgt_cell (batch, 768)

Design rules we committed to:
  1. TIME lives ONLY in the predictor. The encoders never see the timepoint,
     so their embeddings describe "what the cell/gene is", not "when it is".
  2. CONTEXT: each cell's source embedding is enriched with the average
     embedding of the OTHER cells in the batch (leave-one-out), because a
     single cell alone cannot identify where the population is heading.
  3. The EMA (exponential moving average) target encoder provides the
     prediction targets and receives no gradients — standard JEPA recipe
     to avoid the trivial "everything maps to the same point" solution.

Terminology used everywhere in this file:
  batch    = number of cells in the mini-batch                (short: B)
  src_len  = padded source gene-sequence length               (short: Ls)
  tgt_len  = padded target gene-sequence length               (short: Lt)
  Q        = number of gene queries we ask per cell
  D        = embedding width (768 for the pretrained encoder)
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Token id 0 means "padding" in BOTH id spaces (global and local).
PAD_TOKEN_ID = 0

# LPS hours for predictor time_step 1/2/3. Unused index 0 -> 0 h.
TIME_STEP_HOURS = {0: 0.0, 1: 1.5, 2: 6.0, 3: 10.0}
DCT_T_MAX = 10.0
DCT_N_BASIS = 2


class PopulationContext(nn.Module):
    """Mix each cell's embedding with the average of the OTHER cells.

    Idea: one cell alone cannot tell us where the tissue is heading, but the
    surrounding population can. All source cells in a batch come from the
    same source timepoint, so the batch average is a cheap stand-in for
    "the state of the population right now".

    "Leave-one-out" means: for cell i we average everyone EXCEPT cell i,
    so the cell cannot simply read information about itself from the context.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Takes [own gene embedding ; population context] and squeezes it
        # back down to the normal embedding width.
        self.mix = nn.Linear(d_model * 2, d_model)

    def leave_one_out_mean(self, cell_embeddings: torch.Tensor) -> torch.Tensor:
        """Average of the other cells' embeddings, one row per cell.

        cell_embeddings: (B, D)   returns: (B, D)
        """
        batch_size = cell_embeddings.size(0)
        if batch_size == 1:
            # Only one cell: there is no "other cell", fall back to itself.
            return cell_embeddings
        total = cell_embeddings.sum(dim=0, keepdim=True)     # (1, D)
        others_sum = total - cell_embeddings                 # (B, D)
        return others_sum / (batch_size - 1)                 # (B, D)

    def forward(
        self,
        token_embeddings: torch.Tensor,   # (B, Ls, D) per-gene embeddings
        cell_embeddings: torch.Tensor,    # (B, D)     pooled per-cell embeddings
    ) -> torch.Tensor:
        context = self.leave_one_out_mean(cell_embeddings)   # (B, D)

        # Give every gene position of cell i the same context vector.
        context_per_token = context.unsqueeze(1)             # (B, 1, D)
        context_per_token = context_per_token.expand_as(token_embeddings)

        combined = torch.cat([token_embeddings, context_per_token], dim=-1)
        return self.mix(combined)                            # (B, Ls, D)


class DCTQueryBuilder(nn.Module):
    """Per-query time vectors on a K=2 DCT clock (hours 1.5 / 6 / 10).

    Shared (gene in source and target) — quantify the shift:
        q_g(t) = H_src[g] + Σ_k (W_k H_src[g]) φ_k(t)

    Target-only (induced) — open the name prototype, do not shift:
        q_g(t) = e(g) ⊙ σ( Σ_k (U_k e(g)) φ_k(t) )

    φ_k(t) = cos(π k t / T), T=10, K=2. W_k and U_k are separate.
    """

    def __init__(
        self,
        d_model: int,
        n_basis: int = DCT_N_BASIS,
        t_max: float = DCT_T_MAX,
    ):
        super().__init__()
        self.n_basis = int(n_basis)
        self.t_max = float(t_max)
        self.shared_maps = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(self.n_basis)]
        )
        self.induced_maps = nn.ModuleList(
            [nn.Linear(d_model, d_model, bias=False) for _ in range(self.n_basis)]
        )

    def basis(self, time_step: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t_hours = float(TIME_STEP_HOURS.get(int(time_step), 0.0))
        k = torch.arange(self.n_basis, device=device, dtype=dtype)
        return torch.cos(math.pi * k * t_hours / self.t_max)

    def forward(
        self,
        h_src_at_query: torch.Tensor,   # (B, Q, D) source row; dummy if not shared
        identity: torch.Tensor,         # (B, Q, D) e(g)
        is_shared: torch.Tensor,        # (B, Q) bool
        time_step: int,
    ) -> torch.Tensor:
        phi = self.basis(time_step, identity.device, identity.dtype)  # (K,)

        shift = torch.zeros_like(h_src_at_query)
        for k, layer in enumerate(self.shared_maps):
            shift = shift + phi[k] * layer(h_src_at_query)
        q_shared = h_src_at_query + shift

        pre_gain = torch.zeros_like(identity)
        for k, layer in enumerate(self.induced_maps):
            pre_gain = pre_gain + phi[k] * layer(identity)
        q_induced = identity * torch.sigmoid(pre_gain)

        return torch.where(is_shared.unsqueeze(-1), q_shared, q_induced)


class GeneQueryPredictor(nn.Module):
    """Answer gene questions about the future.

    Builds a type-specific query (DCT shift vs DCT gain), then cross-attends
    to the source cell. Queries also see each other through self-attention.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.query_time = DCTQueryBuilder(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            activation='gelu',
            dropout=dropout,
            batch_first=True,   # tensors are (batch, sequence, feature)
        )
        self.cross_attention = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(
        self,
        h_src_at_query: torch.Tensor,         # (B, Q, D)
        identity: torch.Tensor,               # (B, Q, D)
        is_shared: torch.Tensor,              # (B, Q)
        time_step: int,
        source_memory: torch.Tensor,          # (B, Ls, D)
        source_is_padding: torch.Tensor,      # (B, Ls)
    ) -> torch.Tensor:
        queries = self.query_time(
            h_src_at_query, identity, is_shared, time_step
        )
        predicted = self.cross_attention(
            tgt=queries,
            memory=source_memory,
            memory_key_padding_mask=source_is_padding,
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
        max_seq_length: int = 2048,
    ):
        super().__init__()
        self.ema_decay = ema_decay
        self.normalize_latents = normalize_latents
        self.encoder_type = encoder_type

        # ------------------------------------------------------------------
        # 1) The two encoders: online (trains) and EMA target (frozen copy).
        # ------------------------------------------------------------------
        if encoder_type == 'scmaskgit':
            from perturbgen.Modules.jepa_scmaskgit import SCMaskGITCellEncoder

            self.online_encoder = SCMaskGITCellEncoder(
                encoder_path=encoder_path,
                freeze=freeze_encoder,
                n_encoder_layers=n_encoder_layers,
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
        # 2) Population context mixer (design rule 2 in the file docstring).
        # ------------------------------------------------------------------
        self.population_context = PopulationContext(self.d_model)

        # ------------------------------------------------------------------
        # 3) The time-conditioned gene-query predictor (design rule 1:
        #    this is the ONLY place the model learns about time).
        # ------------------------------------------------------------------
        self.predictor = GeneQueryPredictor(
            d_model=self.d_model,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # 4) Fallback vector for padded (invalid) query slots only.
        # Mixed sampling has no absent decoys; this is just a shape filler.
        # ------------------------------------------------------------------
        self.absent_gene_embedding = nn.Parameter(
            torch.randn(self.d_model) * 0.02
        )

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
        query_src_position: torch.Tensor, # (B, Q)  where in src, or -1 if induced
        time_step: int,                   # which target timepoint (1, 2 or 3)
    ) -> Dict[str, torch.Tensor]:
        """Run the whole diagram once. Returns every arrow's endpoint.

        All id tensors must already be in the id space the encoder expects
        (the trainer takes care of converting; see the trainer docstring).
        For padded invalid queries, query_tgt_position is 0 and unused.
        """
        # ---- Step 1: online encoder reads the source cell. -----------------
        online_out = self.online_encoder(src_input_ids, time_step=0)
        h_src = online_out['token_embedding']           # (B, Ls, D)
        z_src_cell = online_out['cell_embedding']       # (B, D)

        # ---- Step 2: enrich the source with population context. ------------
        source_memory = self.population_context(
            token_embeddings=h_src,
            cell_embeddings=z_src_cell,
        )                                               # (B, Ls, D)
        source_is_padding = src_input_ids == PAD_TOKEN_ID   # (B, Ls)

        # ---- Step 3: EMA encoder reads the target cell (no gradients). -----
        with torch.no_grad():
            target_out = self.target_encoder(tgt_input_ids, time_step=0)
            h_tgt = target_out['token_embedding']       # (B, Lt, D)
            z_tgt_cell = target_out['cell_embedding']   # (B, D)

        # ---- Step 4: true answer = that gene's row in H_tgt (or pad fill). -
        batch_size, n_queries = query_gene_ids.shape
        position_as_index = query_tgt_position.unsqueeze(-1)     # (B, Q, 1)
        position_as_index = position_as_index.expand(-1, -1, self.d_model)
        true_gene_embedding = torch.gather(h_tgt, dim=1, index=position_as_index)
        absent_answer = self.absent_gene_embedding.expand(
            batch_size, n_queries, self.d_model
        )
        is_present = query_is_present.unsqueeze(-1)     # (B, Q, 1)
        z_tgt_gene = torch.where(is_present, true_gene_embedding, absent_answer)

        # ---- Step 5: per-query DCT time, then predict. ---------------------
        is_shared = query_src_position >= 0
        safe_src = query_src_position.clamp(min=0)
        src_index = safe_src.unsqueeze(-1).expand(-1, -1, self.d_model)
        h_src_at_query = torch.gather(h_src, dim=1, index=src_index)
        identity_table = self._gene_identity_table(self.online_encoder)
        identity = identity_table(query_gene_ids)              # (B, Q, D)
        z_hat_gene = self.predictor(
            h_src_at_query=h_src_at_query,
            identity=identity,
            is_shared=is_shared,
            time_step=time_step,
            source_memory=source_memory,
            source_is_padding=source_is_padding,
        )                                               # (B, Q, D)

        # ---- Step 6: cell-level view = average over the present queries. ---
        present_mask = query_is_present.unsqueeze(-1).float()    # (B, Q, 1)
        n_present = present_mask.sum(dim=1).clamp(min=1.0)       # (B, 1)
        z_hat_cell = (z_hat_gene * present_mask).sum(dim=1) / n_present  # (B, D)

        # ---- Step 7: normalise everything that enters a cosine loss. -------
        z_hat_gene = self._maybe_normalize(z_hat_gene)
        z_tgt_gene = self._maybe_normalize(z_tgt_gene)
        z_hat_cell = self._maybe_normalize(z_hat_cell)
        z_tgt_cell = self._maybe_normalize(z_tgt_cell)
        z_src_cell = self._maybe_normalize(z_src_cell)

        # ---- Step 8 (metrics only): the "no learning" baselines. -----------
        with torch.no_grad():
            static_table = self._gene_identity_table(self.target_encoder)
            z_static_gene = self._maybe_normalize(
                static_table(query_gene_ids)
            )                                           # (B, Q, D)

        return {
            'z_hat_gene': z_hat_gene,       # (B, Q, D) predicted gene embeddings
            'z_tgt_gene': z_tgt_gene,       # (B, Q, D) true gene embeddings (EMA)
            'z_hat_cell': z_hat_cell,       # (B, D)    predicted cell embedding
            'z_tgt_cell': z_tgt_cell,       # (B, D)    true cell embedding (EMA)
            'z_src_cell': z_src_cell,       # (B, D)    source cell embedding
            'h_src': self._maybe_normalize(h_src),  # (B, Ls, D) for baselines
            'z_static_gene': z_static_gene, # (B, Q, D) context-free baseline
        }
