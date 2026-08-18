"""Gene-Query JEPA smoke tests. KEEP THIS FILE.

CPU only, encoder_type='cell' (no pretrained ckpt, no GPU).
Does not replace toy training or the hyperparameter sweep.

    pytest perturbgen/tests/test_gene_query_jepa.py -v

Honesty metric in real runs: val/gene_gap_vs_copy_src > 0.
Index: docs/examples/GENE_QUERY_JEPA.md
"""

import pickle
import random

import pytest
import torch

from perturbgen.Model.gene_query_jepa_trainer import (
    FIRST_REAL_GENE_LOCAL_ID,
    GeneQueryJEPATrainer,
    info_nce_present_genes,
    sample_query_batch,
)
from perturbgen.src.jepa_token_maps import build_lookup_tables
from perturbgen.Modules.gene_query_jepa import DCTQueryBuilder


# --------------------------------------------------------------------------
# A tiny made-up vocabulary shared by all tests:
#   GLOBAL gene ids 100..149  <-->  LOCAL gene ids 4..53
# --------------------------------------------------------------------------
N_GENES = 50


def make_id_maps():
    tokenid_to_rowid = {
        100 + i: FIRST_REAL_GENE_LOCAL_ID + i for i in range(N_GENES)
    }
    global_to_local, local_to_global = build_lookup_tables(tokenid_to_rowid)
    return tokenid_to_rowid, global_to_local, local_to_global


def test_info_nce_perfect_match_is_low():
    """If each prediction equals its own target, InfoNCE should be near 0."""
    # 4 present genes with orthogonal-ish distinct directions.
    eye = torch.eye(4)
    z_hat = eye.unsqueeze(0)          # (1, 4, 4)
    z_tgt = eye.unsqueeze(0)
    is_present = torch.ones(1, 4, dtype=torch.bool)
    loss = info_nce_present_genes(z_hat, z_tgt, is_present, temperature=0.1)
    assert torch.isfinite(loss)
    assert loss.item() < 0.05


def test_info_nce_skipped_when_too_few_present():
    z_hat = torch.randn(1, 2, 8)
    z_tgt = torch.randn(1, 2, 8)
    is_present = torch.tensor([[True, False]])
    loss = info_nce_present_genes(z_hat, z_tgt, is_present)
    assert float(loss) == 0.0


def test_dct_query_builder_shared_shift_vs_induced_gain():
    """Shared adds a DCT shift to H_src; induced scales e(g). Same t, different type."""
    torch.manual_seed(0)
    builder = DCTQueryBuilder(d_model=8, n_basis=2)
    h_src = torch.randn(1, 3, 8)
    identity = torch.randn(1, 3, 8)
    is_shared = torch.tensor([[True, False, True]])
    q1 = builder(h_src, identity, is_shared, time_step=1)
    q2 = builder(h_src, identity, is_shared, time_step=2)
    assert q1.shape == (1, 3, 8)
    # Time must change the query.
    assert (q1 - q2).abs().sum().item() > 0
    # Induced slot is a scaled identity, not H_src + shift.
    induced = q1[0, 1]
    ident = identity[0, 1]
    # Gain is sigmoid → same sign as e(g), strictly smaller magnitude.
    assert torch.all(induced * ident >= -1e-5)
    assert torch.all(induced.abs() <= ident.abs() + 1e-5)


def test_sampler_pads_when_present_pool_is_small():
    """Q larger than present genes -> invalid pads, not absent decoys."""
    _, global_to_local, _ = make_id_maps()
    src = torch.tensor([[100, 101, 102, 105, 0, 0]])
    tgt = torch.tensor([[4, 5, 10, 11, 0, 0]])
    all_gene_local_ids = list(
        range(FIRST_REAL_GENE_LOCAL_ID, FIRST_REAL_GENE_LOCAL_ID + N_GENES)
    )
    quiz = sample_query_batch(
        src_input_ids_global=src,
        tgt_input_ids_local=tgt,
        global_to_local=global_to_local,
        all_gene_local_ids=all_gene_local_ids,
        n_queries=8,
        frac_shared=0.5,
        frac_tgt_only=0.5,
        query_mode='mixed',
        shared_max_queries=0,
        rng=random.Random(0),
    )
    is_present = quiz['is_present'][0]
    is_valid = quiz['is_valid'][0]
    assert int(is_present.sum()) == 4
    assert int(is_valid.sum()) == 4
    assert not bool((~is_valid & is_present).any())
    tgt_genes = {4, 5, 10, 11}
    for gene, valid in zip(quiz['gene_ids_local'][0].tolist(), is_valid.tolist()):
        if valid:
            assert gene in tgt_genes


def test_sampler_mix_and_positions():
    """One cell with a known gene layout: check pools, counts, positions."""
    _, global_to_local, _ = make_id_maps()

    # Source cell expresses genes (GLOBAL): 100, 101, 102, 105  -> LOCAL 4,5,6,9
    src = torch.tensor([[100, 101, 102, 105, 0, 0]])
    # Target cell expresses genes (LOCAL): 4, 5, 10, 11
    #   shared with source: {4, 5}      target-only: {10, 11}
    tgt = torch.tensor([[4, 5, 10, 11, 0, 0]])

    all_gene_local_ids = list(
        range(FIRST_REAL_GENE_LOCAL_ID, FIRST_REAL_GENE_LOCAL_ID + N_GENES)
    )
    quiz = sample_query_batch(
        src_input_ids_global=src,
        tgt_input_ids_local=tgt,
        global_to_local=global_to_local,
        all_gene_local_ids=all_gene_local_ids,
        n_queries=4,
        frac_shared=0.5,      # wants 2 shared, exactly 2 exist
        frac_tgt_only=0.5,    # wants 2 target-only, exactly 2 exist
        query_mode='mixed',
        shared_max_queries=0,
        rng=random.Random(0),
    )

    gene_ids = quiz['gene_ids_local'][0].tolist()
    is_present = quiz['is_present'][0].tolist()
    tgt_position = quiz['tgt_position'][0].tolist()
    src_position = quiz['src_position'][0].tolist()

    # 4 present questions (2 shared + 2 target-only). No absent decoys.
    assert sum(is_present) == 4
    assert len(gene_ids) == 4
    assert all(quiz['is_valid'][0].tolist())

    tgt_genes = {4, 5, 10, 11}
    for gene, present, pos_t, pos_s in zip(
        gene_ids, is_present, tgt_position, src_position
    ):
        assert present
        assert gene in tgt_genes
        # The recorded target position must really hold that gene.
        assert tgt[0, pos_t].item() == gene
        if gene in (4, 5):          # shared genes: source position valid
            assert pos_s >= 0
            local_of_src_token = global_to_local[src[0, pos_s]].item()
            assert local_of_src_token == gene
        else:                        # target-only genes: not in source
            assert pos_s == -1


@pytest.fixture()
def tiny_trainer(tmp_path):
    """A GeneQueryJEPATrainer with the small CPU 'cell' encoder."""
    tokenid_to_rowid, _, _ = make_id_maps()
    map_path = tmp_path / 'tokenid_to_rowid_test.pkl'
    with open(map_path, 'wb') as f:
        pickle.dump(tokenid_to_rowid, f)

    trainer = GeneQueryJEPATrainer(
        jepa_encoder='cell',
        n_queries=8,
        pred_tps=[1],
        n_total_tps=3,
        tokenid_to_rowid_path=str(map_path),
        output_dir=str(tmp_path / 'out'),
        # Small 'cell' encoder settings (LOCAL vocab has ids up to 53):
        tgt_vocab_size=60,
        d_model=32,
        num_heads=4,
        num_layers=1,
        d_ff=64,
        max_seq_length=64,
        seed=0,
    )
    return trainer


def make_fake_batch(batch_size=4, src_len=12, tgt_len=10, seed=1):
    """Random cells: source tokens GLOBAL (100..149), target tokens LOCAL (4..53)."""
    rng = random.Random(seed)
    src_rows, tgt_rows = [], []
    for _ in range(batch_size):
        src_genes = rng.sample(range(100, 100 + N_GENES), src_len - 2)
        tgt_genes = rng.sample(
            range(FIRST_REAL_GENE_LOCAL_ID, FIRST_REAL_GENE_LOCAL_ID + N_GENES),
            tgt_len - 2,
        )
        src_rows.append(src_genes + [0, 0])   # two padding tokens at the end
        tgt_rows.append(tgt_genes + [0, 0])
    return {
        'src_input_ids': torch.tensor(src_rows),
        'tgt_input_ids_t1': torch.tensor(tgt_rows),
    }


def test_forward_shapes_and_losses(tiny_trainer):
    batch = make_fake_batch()
    result = tiny_trainer._run_one_timestep(batch, time_step=1)

    out = result['_out']
    B, Q, D = 4, 8, 32
    assert out['z_hat_gene'].shape == (B, Q, D)
    assert out['z_tgt_gene'].shape == (B, Q, D)
    assert out['z_hat_cell'].shape == (B, D)
    assert out['z_tgt_cell'].shape == (B, D)

    for key in ('total_loss', 'gene_loss', 'cell_loss', 'contrastive_loss'):
        assert torch.isfinite(result[key]), f'{key} is not finite'
    # Normalised embeddings -> cosine in [-1, 1] -> distance in [0, 2].
    assert 0.0 <= result['gene_loss'].item() <= 2.0

    # EMA targets must carry no gradient; predictions must carry gradient.
    assert not out['z_tgt_cell'].requires_grad
    assert out['z_hat_gene'].requires_grad


def test_backward_reaches_predictor_and_context(tiny_trainer):
    batch = make_fake_batch()
    result = tiny_trainer._run_one_timestep(batch, time_step=1)
    result['total_loss'].backward()

    def has_gradient(module):
        return any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters()
        )

    assert has_gradient(tiny_trainer.model.predictor)
    assert has_gradient(tiny_trainer.model.predictor.query_time)
    assert has_gradient(tiny_trainer.model.population_context)
    assert has_gradient(tiny_trainer.model.online_encoder)
    # The EMA target encoder must never receive gradients.
    assert all(
        p.grad is None for p in tiny_trainer.model.target_encoder.parameters()
    )


def test_ema_update_moves_target_towards_online(tiny_trainer):
    model = tiny_trainer.model
    online_param = next(model.online_encoder.parameters())
    target_param = next(model.target_encoder.parameters())

    with torch.no_grad():
        online_param.add_(1.0)          # pretend training changed the online side
    before = target_param.clone()
    model.update_target_encoder()
    moved = (target_param - before).abs().mean().item()

    # decay=0.996 -> the target should move by about 0.004 towards online.
    assert moved == pytest.approx(0.004, rel=0.05)
