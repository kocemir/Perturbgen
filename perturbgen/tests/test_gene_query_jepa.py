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
    encoder_stage_at_epoch,
    info_nce_present_genes,
    parse_encoder_lr_schedule,
    sample_query_batch,
)
from perturbgen.src.jepa_token_maps import build_lookup_tables
from perturbgen.Modules.gene_query_jepa import GeneQueryPredictor


# --------------------------------------------------------------------------
# A tiny made-up vocabulary shared by all tests:
#   GLOBAL gene ids 100..149  <-->  LOCAL gene ids 4..53
# --------------------------------------------------------------------------
N_GENES = 50


def make_id_maps():
    tokenid_to_rowid = {
        2: 2,  # <cls> in both GLOBAL and LOCAL
        **{100 + i: FIRST_REAL_GENE_LOCAL_ID + i for i in range(N_GENES)},
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


def test_predictor_time_changes_output():
    """Different time_step must produce different predictions."""
    torch.manual_seed(0)
    d = 8
    predictor = GeneQueryPredictor(d_model=d, n_time_steps=3, num_layers=1, num_heads=2)
    gene_embs = torch.randn(1, 3, d)
    source = torch.randn(1, 5, d)
    pad = torch.zeros(1, 5, dtype=torch.bool)
    q1 = predictor(gene_embs, time_step=1, source_memory=source, source_is_padding=pad)
    q2 = predictor(gene_embs, time_step=2, source_memory=source, source_is_padding=pad)
    assert q1.shape == (1, 3, d)
    assert (q1 - q2).abs().sum().item() > 0


def test_default_predictor_is_transformer_decoder_ffn_4x(tmp_path):
    """Default predictor must match every run up to 2026-09-22."""
    tokenid_to_rowid, _, _ = make_id_maps()
    map_path = tmp_path / 'tokenid_to_rowid_test.pkl'
    with open(map_path, 'wb') as handle:
        pickle.dump(tokenid_to_rowid, handle)
    trainer = GeneQueryJEPATrainer(
        jepa_encoder='cell',
        n_queries=4,
        predictor_layers=1,
        pred_tps=[1],
        n_total_tps=3,
        tokenid_to_rowid_path=str(map_path),
        output_dir=str(tmp_path / 'out'),
        tgt_vocab_size=60,
        d_model=32,
        num_heads=4,
        num_layers=1,
        d_ff=64,
        max_seq_length=64,
        seed=0,
        cell_pool='cls',
    )
    predictor = trainer.model.predictor
    assert predictor.arch == 'torch'
    assert not hasattr(predictor, 'decoder_block')
    layer = predictor.cross_attention.layers[0]
    assert layer.linear1.out_features == 4 * trainer.model.d_model


@pytest.mark.parametrize('ckpt_arch', ['torch', 'block'])
def test_checkpoint_predictor_shape_wins_on_load(tmp_path, ckpt_arch):
    """Either predictor shape in a ckpt must load without re-initialising."""
    tokenid_to_rowid, _, _ = make_id_maps()
    map_path = tmp_path / 'tokenid_to_rowid_test.pkl'
    with open(map_path, 'wb') as handle:
        pickle.dump(tokenid_to_rowid, handle)
    trainer = GeneQueryJEPATrainer(
        jepa_encoder='cell',
        n_queries=4,
        predictor_layers=1,
        pred_tps=[1],
        n_total_tps=3,
        tokenid_to_rowid_path=str(map_path),
        output_dir=str(tmp_path / 'out'),
        tgt_vocab_size=60,
        d_model=32,
        num_heads=4,
        num_layers=1,
        d_ff=64,
        max_seq_length=64,
        seed=0,
        cell_pool='cls',
    )
    saved_predictor = GeneQueryPredictor(
        d_model=trainer.model.d_model,
        n_time_steps=3,
        num_layers=1,
        arch=ckpt_arch,
    )
    with torch.no_grad():
        saved_predictor.time_embedding.weight.fill_(0.25)
    state = {
        name: tensor
        for name, tensor in trainer.state_dict().items()
        if not name.startswith('model.predictor.')
    }
    state.update(
        {
            f'model.predictor.{name}': tensor
            for name, tensor in saved_predictor.state_dict().items()
        }
    )
    checkpoint = {'state_dict': state}
    trainer.on_load_checkpoint(checkpoint)
    trainer.load_state_dict(checkpoint['state_dict'])
    assert trainer.model.predictor.arch == ckpt_arch
    assert torch.allclose(
        trainer.model.predictor.time_embedding.weight,
        torch.full_like(trainer.model.predictor.time_embedding.weight, 0.25),
    )


def test_sampler_mix_and_positions():
    """One cell: ~50% shared / ~30% tgt-only / rest absent decoys."""
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
        n_queries=8,
        frac_shared=0.5,      # wants 4 shared; only 2 exist
        frac_tgt_only=0.3,    # wants 2 target-only; 2 exist
        query_mode='mixed',
        shared_max_queries=0,
        rng=random.Random(0),
    )

    gene_ids = quiz['gene_ids_local'][0].tolist()
    is_present = quiz['is_present'][0].tolist()
    tgt_position = quiz['tgt_position'][0].tolist()
    src_position = quiz['src_position'][0].tolist()

    # Present pool has 4 genes, so 4 present + 4 absent decoys.
    assert len(gene_ids) == 8
    assert sum(is_present) == 4
    assert sum(1 for p in is_present if not p) == 4
    assert 'is_valid' not in quiz
    assert not quiz['is_mask_ce'][0].any()

    tgt_genes = {4, 5, 10, 11}
    for gene, present, pos_t, pos_s in zip(
        gene_ids, is_present, tgt_position, src_position
    ):
        if present:
            assert gene in tgt_genes
            assert tgt[0, pos_t].item() == gene
            if gene in (4, 5):
                assert pos_s >= 0
                local_of_src_token = global_to_local[src[0, pos_s]].item()
                assert local_of_src_token == gene
            else:
                assert pos_s == -1
        else:
            assert gene not in tgt_genes
            assert pos_t == 0
            assert pos_s == -1


def test_mask_frac_marks_shared_only():
    _, global_to_local, _ = make_id_maps()
    src = torch.tensor([[100, 101, 102, 105, 0, 0]])
    tgt = torch.tensor([[4, 5, 6, 9, 10, 11]])
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
        mask_frac_of_shared=1.0,
        rng=random.Random(0),
    )
    is_mask = quiz['is_mask_ce'][0]
    is_present = quiz['is_present'][0]
    src_pos = quiz['src_position'][0]
    assert is_mask.any()
    assert bool((is_mask & (src_pos < 0)).sum() == 0)
    assert bool((is_mask & ~is_present).sum() == 0)


def test_mse_and_mask_ce_loss(tmp_path):
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
        tgt_vocab_size=60,
        d_model=32,
        num_heads=4,
        num_layers=1,
        d_ff=64,
        max_seq_length=64,
        seed=0,
        cell_pool='cls',
        gene_loss='mse',
        lambda_mask_ce=1.0,
        mask_frac_of_shared=1.0,
        dump_raw_latents=True,
    )
    batch = make_fake_batch()
    result = trainer._run_one_timestep(batch, time_step=1)
    assert torch.isfinite(result['gene_loss'])
    assert torch.isfinite(result['mask_ce_loss'])
    assert 'z_hat_gene_raw' in result['_out']
    result['total_loss'].backward()
    assert trainer.mask_ce_head.weight.grad is not None
    assert trainer.model.mask_query.grad is not None
    # CE head must be in AdamW (lives on LightningModule, not self.model).
    optimizer = trainer.configure_optimizers()['optimizer']
    trainable = {id(p) for g in optimizer.param_groups for p in g['params']}
    assert id(trainer.mask_ce_head.weight) in trainable
    assert id(trainer.mask_ce_head.bias) in trainable
    assert id(trainer.model.mask_query) in trainable


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
        cell_pool='cls',
    )
    return trainer


def make_fake_batch(batch_size=4, src_len=12, tgt_len=10, seed=1):
    """Random cells: source tokens GLOBAL (100..149), target tokens LOCAL (4..53)."""
    rng = random.Random(seed)
    src_rows, tgt_rows = [], []
    for _ in range(batch_size):
        src_genes = rng.sample(range(100, 100 + N_GENES), src_len - 3)
        tgt_genes = rng.sample(
            range(FIRST_REAL_GENE_LOCAL_ID, FIRST_REAL_GENE_LOCAL_ID + N_GENES),
            tgt_len - 3,
        )
        src_rows.append([2] + src_genes + [0, 0])   # CLS + genes + pad
        tgt_rows.append([2] + tgt_genes + [0, 0])
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
    assert out['z_hat_cell_raw'].shape == (B, D)
    assert out['z_src_cell_raw'].shape == (B, D)

    for key in (
        'total_loss', 'gene_loss', 'cell_loss', 'contrastive_loss',
        'mask_ce_loss', 'gene_mse',
    ):
        assert torch.isfinite(result[key]), f'{key} is not finite'
    # Normalised embeddings -> cosine in [-1, 1] -> distance in [0, 2].
    assert 0.0 <= result['gene_loss'].item() <= 2.0

    # EMA targets must carry no gradient; predictions must carry gradient.
    assert not out['z_tgt_cell'].requires_grad
    assert out['z_hat_gene'].requires_grad


def test_test_step_dumps_l2_and_raw(tiny_trainer):
    tiny_trainer.on_test_epoch_start()
    tiny_trainer.test_step(make_fake_batch())
    assert 'hat_t1' in tiny_trainer._test_gene_sums
    assert 'hat_t1_raw' in tiny_trainer._test_gene_sums
    assert 'tgt_t1_raw' in tiny_trainer._test_gene_sums
    assert tiny_trainer._test_cell_rows['z_hat_cell_raw']
    l2 = torch.nn.functional.normalize(
        tiny_trainer._test_cell_rows['z_hat_cell'][0].float(), dim=-1
    )
    assert torch.allclose(tiny_trainer._test_cell_rows['z_hat_cell'][0].float(), l2, atol=1e-5)


def test_encode_one_timestep_skips_mask_ce(tiny_trainer):
    tiny_trainer.lambda_mask_ce = 1.0
    tiny_trainer.mask_frac_of_shared = 1.0
    result = tiny_trainer.encode_one_timestep(make_fake_batch(), time_step=1)
    assert not result['_quiz']['is_mask_ce'].any()
    assert '_gene_ids_local' in result
    assert '_src_position' in result


def test_backward_reaches_predictor(tiny_trainer):
    batch = make_fake_batch()
    result = tiny_trainer._run_one_timestep(batch, time_step=1)
    result['total_loss'].backward()

    def has_gradient(module):
        return any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters()
        )

    assert has_gradient(tiny_trainer.model.predictor)
    assert has_gradient(tiny_trainer.model.predictor.time_embedding)
    assert tiny_trainer.model.cell_query.grad is not None
    assert tiny_trainer.model.cell_query.grad.abs().sum() > 0
    assert not hasattr(tiny_trainer.model, 'population_context')
    assert has_gradient(tiny_trainer.model.online_encoder)
    # The EMA target encoder must never receive gradients.
    assert all(
        p.grad is None for p in tiny_trainer.model.target_encoder.parameters()
    )


def test_freeze_encoder_epochs_starts_frozen_then_unfreezes(tmp_path):
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
        tgt_vocab_size=60,
        d_model=32,
        num_heads=4,
        num_layers=1,
        d_ff=64,
        max_seq_length=64,
        seed=0,
        freeze_jepa_encoder=True,
        freeze_encoder_epochs=5,
        cell_pool='cls',
        vicreg_var_coeff=0.0,
        vicreg_cov_coeff=0.0,
    )
    # DDP-safe freeze: grads flow from step 0, the encoder group lr is 0.
    assert trainer.encoder_trains
    assert all(p.requires_grad for p in trainer.model.online_encoder.parameters())
    assert any(p.requires_grad for p in trainer.model.predictor.parameters())
    optimizer = trainer.configure_optimizers()['optimizer']
    group = trainer._encoder_optimizer_group(optimizer)
    assert group is not None
    assert group['lr'] == 0.0
    assert trainer.encoder_lr_at_epoch(4) == 0.0
    assert trainer.encoder_lr_at_epoch(5) == trainer.encoder_lr
    encoder_ids = {id(p) for p in trainer.model.online_encoder.parameters()}
    head_group = optimizer.param_groups[0]
    assert not any(id(p) in encoder_ids for p in head_group['params'])
    assert head_group['lr'] == trainer.lr


def test_encoder_lr_schedule_drives_group_lr(tmp_path):
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
        tgt_vocab_size=60,
        d_model=32,
        num_heads=4,
        num_layers=1,
        d_ff=64,
        max_seq_length=64,
        seed=0,
        cell_pool='cls',
        freeze_jepa_encoder=False,
        encoder_lr_schedule='freeze:2,1e-5:3',
        lambda_mask_ce=0.0,
    )
    assert trainer.encoder_trains
    assert all(p.requires_grad for p in trainer.model.online_encoder.parameters())
    assert trainer.encoder_lr_at_epoch(0) == 0.0
    assert trainer.encoder_lr_at_epoch(1) == 0.0
    assert abs(trainer.encoder_lr_at_epoch(2) - 1e-5) < 1e-15
    assert abs(trainer.encoder_lr_at_epoch(10) - 1e-5) < 1e-15
    optimizer = trainer.configure_optimizers()['optimizer']
    assert len(optimizer.param_groups) == 2
    assert trainer._encoder_optimizer_group(optimizer)['lr'] == 0.0
    # mask-CE off -> head / mask_query stay out of the optimiser.
    # query_src_pos stays trainable (default on) even when mask-CE is off.
    assert not trainer.mask_ce_head.weight.requires_grad
    assert not trainer.model.mask_query.requires_grad
    assert trainer.model.query_src_pos_embedding.weight.requires_grad
    assert trainer.model.mask_src_pos_embedding is trainer.model.query_src_pos_embedding
    trainable = {id(p) for g in optimizer.param_groups for p in g['params']}
    assert id(trainer.mask_ce_head.weight) not in trainable
    # A fully frozen run keeps the cheap path: no encoder group at all.
    frozen = GeneQueryJEPATrainer(
        jepa_encoder='cell',
        n_queries=8,
        pred_tps=[1],
        n_total_tps=3,
        tokenid_to_rowid_path=str(map_path),
        output_dir=str(tmp_path / 'out2'),
        tgt_vocab_size=60,
        d_model=32,
        num_heads=4,
        num_layers=1,
        d_ff=64,
        max_seq_length=64,
        seed=0,
        cell_pool='cls',
        freeze_jepa_encoder=True,
        encoder_lr_schedule='freeze:50',
    )
    assert not frozen.encoder_trains
    assert not any(p.requires_grad for p in frozen.model.online_encoder.parameters())
    assert len(frozen.configure_optimizers()['optimizer'].param_groups) == 1


def test_all_trainable_params_receive_grad(tmp_path):
    """What DDP(find_unused_parameters=False) checks: every trainable param
    must get a gradient from one training step, with mask-CE off and the
    encoder in its lr=0 'frozen' stage."""
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
        tgt_vocab_size=60,
        d_model=32,
        num_heads=4,
        num_layers=1,
        d_ff=64,
        max_seq_length=64,
        seed=0,
        cell_pool='cls',
        freeze_jepa_encoder=False,
        encoder_lr_schedule='freeze:2,1e-5:3',
        gene_loss='mse',
        lambda_contrastive=0.1,
        lambda_mask_ce=0.0,
    )
    batch = make_fake_batch()
    result = trainer._run_one_timestep(batch, time_step=1)
    result['total_loss'].backward()
    missing = [
        name
        for name, p in trainer.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert missing == [], f'trainable params without grad: {missing}'


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


def test_cls_pool_takes_cls_position():
    from perturbgen.Modules.jepa import cls_pool_tokens

    embs = torch.arange(24, dtype=torch.float).view(2, 4, 3)
    ids = torch.tensor([[2, 10, 11, 0], [7, 2, 8, 0]])
    pooled = cls_pool_tokens(embs, ids, cls_token_id=2)
    assert torch.equal(pooled[0], embs[0, 0])
    assert torch.equal(pooled[1], embs[1, 1])


def test_cls_pool_errors_without_cls():
    from perturbgen.Modules.jepa import cls_pool_tokens

    embs = torch.zeros(1, 3, 4)
    ids = torch.tensor([[4, 5, 0]])
    with pytest.raises(ValueError, match='cell_pool=cls'):
        cls_pool_tokens(embs, ids)


def test_cell_pool_cls_forward(tmp_path):
    tokenid_to_rowid, _, _ = make_id_maps()
    tokenid_to_rowid[2] = 2  # <cls> is 2 in both GLOBAL and LOCAL
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
        tgt_vocab_size=60,
        d_model=32,
        num_heads=4,
        num_layers=1,
        d_ff=64,
        max_seq_length=64,
        seed=0,
        cell_pool='cls',
    )
    batch = make_fake_batch()
    result = trainer._run_one_timestep(batch, time_step=1)
    assert torch.isfinite(result['total_loss'])
    assert trainer.model.online_encoder.cell_pool == 'cls'


def test_z_hat_cell_is_not_present_gene_mean(tiny_trainer):
    import torch.nn.functional as F

    batch = make_fake_batch()
    result = tiny_trainer._run_one_timestep(batch, time_step=1)
    out = result['_out']
    is_present = result['_is_present']
    present_mask = is_present.unsqueeze(-1).float()
    n_present = present_mask.sum(dim=1).clamp(min=1.0)
    gene_mean = (out['z_hat_gene'] * present_mask).sum(dim=1) / n_present
    gene_mean = F.normalize(gene_mean, dim=-1)
    assert not torch.allclose(out['z_hat_cell'], gene_mean, atol=1e-3)
    assert tiny_trainer.model.cell_query.shape[-1] == out['z_hat_cell'].shape[-1]


def test_parse_encoder_lr_schedule_stages():
    stages = parse_encoder_lr_schedule(
        'freeze:5,5e-7:5,1e-6:5,1e-5:5,freeze:10'
    )
    assert stages[0] == (5, None)
    assert stages[1][0] == 5 and abs(stages[1][1] - 5e-7) < 1e-15
    frozen, lr = encoder_stage_at_epoch(stages, 0)
    assert frozen and lr is None
    frozen, lr = encoder_stage_at_epoch(stages, 5)
    assert not frozen and abs(lr - 5e-7) < 1e-15
    frozen, lr = encoder_stage_at_epoch(stages, 19)
    assert not frozen and abs(lr - 1e-5) < 1e-15
    frozen, lr = encoder_stage_at_epoch(stages, 20)
    assert frozen and lr is None
    frozen, lr = encoder_stage_at_epoch(stages, 29)
    assert frozen


def test_parse_toy_20ep_encoder_schedule():
    stages = parse_encoder_lr_schedule('freeze:5,5e-7:5,1e-6:5,freeze:5')
    frozen, lr = encoder_stage_at_epoch(stages, 0)
    assert frozen and lr is None
    frozen, lr = encoder_stage_at_epoch(stages, 9)
    assert not frozen and abs(lr - 5e-7) < 1e-15
    frozen, lr = encoder_stage_at_epoch(stages, 10)
    assert not frozen and abs(lr - 1e-6) < 1e-15
    frozen, lr = encoder_stage_at_epoch(stages, 14)
    assert not frozen
    frozen, lr = encoder_stage_at_epoch(stages, 15)
    assert frozen and lr is None
    frozen, lr = encoder_stage_at_epoch(stages, 19)
    assert frozen


def test_gene_loss_ignores_absent(tiny_trainer):
    batch = make_fake_batch()
    result = tiny_trainer._run_one_timestep(batch, time_step=1)
    is_present = result['_is_present']
    out = result['_out']
    dist = 1.0 - torch.nn.functional.cosine_similarity(
        out['z_hat_gene'], out['z_tgt_gene'], dim=-1
    )
    expected = (dist * is_present.float()).sum() / is_present.float().sum().clamp(min=1)
    assert torch.allclose(result['gene_loss'], expected, atol=1e-5)
    if (~is_present).any():
        scrambled = out['z_hat_gene'].clone()
        scrambled[~is_present] = torch.randn_like(scrambled[~is_present])
        dist_scrambled = 1.0 - torch.nn.functional.cosine_similarity(
            scrambled, out['z_tgt_gene'], dim=-1
        )
        scrambled_present = (
            dist_scrambled * is_present.float()
        ).sum() / is_present.float().sum().clamp(min=1)
        assert torch.allclose(expected.detach(), scrambled_present.detach(), atol=1e-5)


def test_present_gene_mean_skips_absent():
    from perturbgen.Model.gene_query_count_trainer import present_gene_mean

    z = torch.zeros(1, 3, 2)
    z[0, 0] = torch.tensor([2.0, 0.0])
    z[0, 1] = torch.tensor([4.0, 0.0])
    z[0, 2] = torch.tensor([99.0, 99.0])
    is_present = torch.tensor([[True, True, False]])
    mean = present_gene_mean(z, is_present)
    assert torch.allclose(mean, torch.tensor([[3.0, 0.0]]))


def test_first_target_prefix_quiz_takes_prefix_and_skips_pad():
    from perturbgen.Model.gene_query_count_trainer import first_target_prefix_quiz

    _, global_to_local, _ = make_id_maps()
    # Source GLOBAL 100,101 -> LOCAL 4,5 at positions 0,1.
    src = torch.tensor([[100, 101, 0, 0]])
    # Target LOCAL prefix: CLS=2, genes 4,5, then pad. Query length 4.
    tgt = torch.tensor([[2, 4, 5, 0, 0, 0]])
    quiz = first_target_prefix_quiz(
        src_input_ids_global=src,
        tgt_input_ids_local=tgt,
        global_to_local=global_to_local,
        n_queries=4,
    )
    assert quiz['gene_ids_local'].shape == (1, 4)
    assert quiz['gene_ids_local'][0].tolist() == [2, 4, 5, 0]
    assert quiz['is_present'][0].tolist() == [True, True, True, False]
    assert quiz['tgt_position'][0].tolist() == [0, 1, 2, 3]
    assert quiz['src_position'][0].tolist() == [-1, 0, 1, -1]
    assert not quiz['is_mask_ce'].any()


def test_first_target_prefix_quiz_pads_short_target():
    from perturbgen.Model.gene_query_count_trainer import first_target_prefix_quiz

    _, global_to_local, _ = make_id_maps()
    src = torch.tensor([[100, 0]])
    tgt = torch.tensor([[4, 5]])
    quiz = first_target_prefix_quiz(
        src_input_ids_global=src,
        tgt_input_ids_local=tgt,
        global_to_local=global_to_local,
        n_queries=4,
    )
    assert quiz['gene_ids_local'][0].tolist() == [4, 5, 0, 0]
    assert quiz['is_present'][0].tolist() == [True, True, False, False]
    assert quiz['tgt_position'][0].tolist() == [0, 1, 0, 0]


def test_count_input_modes():
    from perturbgen.Model.gene_query_count_trainer import COUNT_INPUT_MODES

    assert COUNT_INPUT_MODES == ('source_only', 'copy_target')
