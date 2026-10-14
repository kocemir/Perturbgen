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
    # No filler needed here (46 candidate decoys), so every slot is valid.
    assert quiz['is_valid'][0].all()
    assert not quiz['is_mask_ce'][0].any()

    tgt_genes = {4, 5, 10, 11}
    src_genes = {4, 5, 6, 9}
    for gene, present, pos_t, pos_s in zip(
        gene_ids, is_present, tgt_position, src_position
    ):
        # src_position describes the SOURCE only, for present and absent alike.
        if gene in src_genes:
            assert pos_s >= 0
            local_of_src_token = global_to_local[src[0, pos_s]].item()
            assert local_of_src_token == gene
        else:
            assert pos_s == -1
        if present:
            assert gene in tgt_genes
            assert tgt[0, pos_t].item() == gene
        else:
            assert gene not in tgt_genes
            assert pos_t == 0


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


# ==========================================================================
# source_blind quiz + absent supervision (Sep-30 redesign)
# ==========================================================================
ALL_GENES = list(range(FIRST_REAL_GENE_LOCAL_ID, FIRST_REAL_GENE_LOCAL_ID + N_GENES))


def _blind_quiz(src, tgt, n_queries=8, n_src=6, seed=0, **extra):
    _, global_to_local, _ = make_id_maps()
    return sample_query_batch(
        src_input_ids_global=src,
        tgt_input_ids_local=tgt,
        global_to_local=global_to_local,
        all_gene_local_ids=ALL_GENES,
        n_queries=n_queries,
        frac_shared=0.4,          # ignored by source_blind
        frac_tgt_only=0.5,        # ignored by source_blind
        rng=random.Random(seed),
        query_sampling='source_blind',
        n_src_queries=n_src,
        **extra,
    )


def test_source_blind_selection_ignores_target():
    """Same source + same seed -> identical questions whatever the target says."""
    src = torch.tensor([[2, 100, 101, 102, 105, 0]])          # LOCAL 4,5,6,9
    tgt_a = torch.tensor([[2, 4, 5, 10, 11, 0]])
    tgt_b = torch.tensor([[2, 20, 21, 22, 23, 0]])             # disjoint target
    quiz_a = _blind_quiz(src, tgt_a, mask_frac_of_shared=0.5)
    quiz_b = _blind_quiz(src, tgt_b, mask_frac_of_shared=0.5)
    for key in ('gene_ids_local', 'src_position', 'is_mask_ce', 'is_valid'):
        assert torch.equal(quiz_a[key], quiz_b[key]), key
    # Labels are the only thing that may differ.
    assert quiz_a['is_present'].sum() > 0
    assert quiz_b['is_present'].sum() == 0


def test_source_blind_positions_and_labels():
    _, global_to_local, _ = make_id_maps()
    src = torch.tensor([[2, 100, 101, 102, 105, 0]])          # LOCAL 4,5,6,9 at 1..4
    tgt = torch.tensor([[2, 4, 5, 10, 11, 0]])
    quiz = _blind_quiz(src, tgt, n_queries=8, n_src=6)
    gene_ids = quiz['gene_ids_local'][0].tolist()
    src_pos = quiz['src_position'][0].tolist()
    present = quiz['is_present'][0].tolist()
    tgt_pos = quiz['tgt_position'][0].tolist()

    assert quiz['is_valid'][0].all()
    assert len(set(gene_ids)) == 8                             # no repeats
    src_genes = {4, 5, 6, 9}
    # Source has only 4 genes -> all 4 asked, remaining 4 from non-source genes.
    assert sum(g in src_genes for g in gene_ids) == 4
    for gene, pos_s, is_p, pos_t in zip(gene_ids, src_pos, present, tgt_pos):
        if gene in src_genes:
            assert global_to_local[src[0, pos_s]].item() == gene
        else:
            assert pos_s == -1
        assert is_p == (gene in {4, 5, 10, 11})
        if is_p:
            assert tgt[0, pos_t].item() == gene
        else:
            assert pos_t == 0
    # Shared genes are present with rank >= 0; genes 6 and 9 are present in the
    # source but ABSENT in the target -> the "gene turns off" case.
    off = [g for g, p in zip(gene_ids, present) if g in (6, 9)]
    assert off and all(not present[gene_ids.index(g)] for g in off)


def test_source_blind_filler_marked_invalid():
    """Tiny universe: pools run dry -> filler slots exist and are invalid."""
    _, global_to_local, _ = make_id_maps()
    src = torch.tensor([[2, 100, 101, 102, 105, 0]])          # LOCAL 4,5,6,9
    tgt = torch.tensor([[2, 4, 5, 10, 0, 0]])
    quiz = sample_query_batch(
        src_input_ids_global=src,
        tgt_input_ids_local=tgt,
        global_to_local=global_to_local,
        all_gene_local_ids=[4, 5, 6, 9, 10],                    # only 1 non-source gene
        n_queries=8,
        frac_shared=0.4,
        frac_tgt_only=0.5,
        rng=random.Random(0),
        query_sampling='source_blind',
        n_src_queries=6,
    )
    valid = quiz['is_valid'][0]
    assert int(valid.sum()) == 5                               # 4 source + 1 non-source
    assert not quiz['is_present'][0][~valid].any()
    assert (quiz['src_position'][0][~valid] == -1).all()
    assert not quiz['is_mask_ce'][0][~valid].any()


def test_source_blind_weights_bias_nonsource_draws():
    src = torch.tensor([[2, 100, 101, 102, 105, 0]])          # LOCAL 4,5,6,9
    tgt = torch.tensor([[2, 4, 5, 0, 0, 0]])
    weights = torch.full((FIRST_REAL_GENE_LOCAL_ID + N_GENES,), 1e-6)
    weights[40] = 1.0                                          # one hot favourite
    weights[41] = 1.0
    hits = 0
    for seed in range(10):
        quiz = _blind_quiz(src, tgt, n_queries=6, n_src=4, seed=seed, nonsrc_weights=weights)
        drawn = set(quiz['gene_ids_local'][0].tolist()) - {4, 5, 6, 9}
        assert len(drawn) == 2
        hits += int(drawn == {40, 41})
    assert hits == 10
    # Weighting is reproducible through the Python seed.
    a = _blind_quiz(src, tgt, n_queries=6, n_src=4, seed=3, nonsrc_weights=weights)
    b = _blind_quiz(src, tgt, n_queries=6, n_src=4, seed=3, nonsrc_weights=weights)
    assert torch.equal(a['gene_ids_local'], b['gene_ids_local'])


def test_source_blind_mask_ce_marks_source_queries_only():
    src = torch.tensor([[2, 100, 101, 102, 105, 0]])
    tgt = torch.tensor([[2, 4, 5, 10, 11, 0]])
    quiz = _blind_quiz(src, tgt, n_queries=8, n_src=4, mask_frac_of_shared=1.0)
    is_mask = quiz['is_mask_ce'][0]
    assert int(is_mask.sum()) == 4
    assert (quiz['src_position'][0][is_mask] >= 0).all()
    assert not (quiz['src_position'][0][~is_mask] >= 0).any()


# --------------------------------------------------------------------------
# n_top_common_queries: deterministic "likely present" slot (Oct-12 redesign)
# --------------------------------------------------------------------------
def test_source_blind_top_common_zero_matches_legacy_path():
    """n_top_common_queries=0 (the default) reproduces the pre-redesign draw
    bit-for-bit given the same seed and weights."""
    src = torch.tensor([[2, 100, 101, 102, 105, 0]])
    tgt = torch.tensor([[2, 4, 5, 0, 0, 0]])
    weights = torch.full((FIRST_REAL_GENE_LOCAL_ID + N_GENES,), 1e-6)
    weights[40] = 1.0
    weights[41] = 1.0
    a = _blind_quiz(src, tgt, n_queries=6, n_src=4, seed=3, nonsrc_weights=weights)
    b = _blind_quiz(
        src, tgt, n_queries=6, n_src=4, seed=3, nonsrc_weights=weights,
        n_top_common_queries=0,
    )
    assert torch.equal(a['gene_ids_local'], b['gene_ids_local'])


def test_source_blind_top_common_is_deterministic_and_distinct():
    """n_top_common_queries picks the K highest-``nonsrc_weights`` non-source
    genes every time (no randomness), never colliding with chosen_src."""
    src = torch.tensor([[2, 100, 101, 102, 105, 0]])          # LOCAL 4,5,6,9
    tgt = torch.tensor([[2, 4, 5, 10, 11, 12, 0]])
    # Strictly increasing in LOCAL id -> the top-K is unambiguous.
    weights = torch.arange(FIRST_REAL_GENE_LOCAL_ID + N_GENES, dtype=torch.float)
    src_genes = {4, 5, 6, 9}
    universe = range(FIRST_REAL_GENE_LOCAL_ID, FIRST_REAL_GENE_LOCAL_ID + N_GENES)
    expected_top = set(sorted(g for g in universe if g not in src_genes)[-3:])

    for seed in range(5):
        quiz = _blind_quiz(
            src, tgt, n_queries=8, n_src=4, seed=seed,
            nonsrc_weights=weights, n_top_common_queries=3,
        )
        gene_ids = quiz['gene_ids_local'][0].tolist()
        assert len(set(gene_ids)) == 8                         # no duplicates
        non_source = [g for g in gene_ids if g not in src_genes]
        assert len(non_source) == 4                            # 8 - 4 source
        assert expected_top.issubset(set(non_source))          # same every seed


def test_source_blind_top_common_leaves_rest_to_lowfreq_weights():
    """The non-top-common remainder is drawn with ``lowfreq_weights``,
    independent of (and never overlapping with) the top-common genes."""
    src = torch.tensor([[2, 100, 101, 102, 105, 0]])          # LOCAL 4,5,6,9
    tgt = torch.tensor([[2, 4, 5, 0, 0, 0]])
    present_weights = torch.full((FIRST_REAL_GENE_LOCAL_ID + N_GENES,), 1e-6)
    present_weights[10] = 1.0                                  # sole top-common pick
    absent_weights = torch.full((FIRST_REAL_GENE_LOCAL_ID + N_GENES,), 1e-6)
    absent_weights[40] = 1.0
    absent_weights[41] = 1.0
    hits = 0
    for seed in range(10):
        quiz = _blind_quiz(
            src, tgt, n_queries=7, n_src=4, seed=seed,
            nonsrc_weights=present_weights,
            n_top_common_queries=1,
            lowfreq_weights=absent_weights,
        )
        gene_ids = quiz['gene_ids_local'][0].tolist()
        assert len(set(gene_ids)) == 7
        assert 10 in gene_ids                                  # deterministic top-1
        drawn_lowfreq = set(gene_ids) - {4, 5, 6, 9, 10}
        assert len(drawn_lowfreq) == 2
        hits += int(drawn_lowfreq == {40, 41})
    assert hits == 10


def test_source_blind_n_top_common_queries_validates_range():
    _, global_to_local, _ = make_id_maps()
    src = torch.tensor([[2, 100, 101, 102, 105, 0]])
    tgt = torch.tensor([[2, 4, 5, 0, 0, 0]])
    with pytest.raises(ValueError, match='n_top_common_queries'):
        sample_query_batch(
            src_input_ids_global=src,
            tgt_input_ids_local=tgt,
            global_to_local=global_to_local,
            all_gene_local_ids=ALL_GENES,
            n_queries=8,
            frac_shared=0.4,
            frac_tgt_only=0.5,
            rng=random.Random(0),
            query_sampling='source_blind',
            n_src_queries=4,
            n_top_common_queries=5,    # > n_queries - n_src_queries (4)
        )


def test_trainer_n_top_common_queries_requires_n_src_queries():
    # query_sampling='target' never auto-fills n_src_queries, so it stays
    # None and the check must fire before any encoder/path args are needed.
    with pytest.raises(ValueError, match='n_top_common_queries'):
        GeneQueryJEPATrainer(
            jepa_encoder='cell',
            n_queries=8,
            query_sampling='target',
            n_top_common_queries=2,
        )


def test_sampler_rejects_unknown_mode():
    src = torch.tensor([[2, 100, 0]])
    tgt = torch.tensor([[2, 4, 0]])
    with pytest.raises(ValueError):
        sample_query_batch(
            src_input_ids_global=src,
            tgt_input_ids_local=tgt,
            global_to_local=make_id_maps()[1],
            all_gene_local_ids=ALL_GENES,
            n_queries=4,
            frac_shared=0.4,
            frac_tgt_only=0.5,
            rng=random.Random(0),
            query_sampling='nope',
        )


@pytest.fixture()
def blind_trainer(tmp_path):
    """source_blind quiz + absent loss, MSE gene loss, tiny CPU encoder."""
    tokenid_to_rowid, _, _ = make_id_maps()
    map_path = tmp_path / 'tokenid_to_rowid_test.pkl'
    with open(map_path, 'wb') as f:
        pickle.dump(tokenid_to_rowid, f)
    return GeneQueryJEPATrainer(
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
        query_sampling='source_blind',
        n_src_queries=6,
        lambda_absent=0.5,
    )


def test_absent_null_is_frozen_buffer_at_layernorm_scale(blind_trainer):
    model = blind_trainer.model
    names = {name for name, _ in blind_trainer.named_parameters()}
    assert 'model.absent_gene_embedding' not in names
    assert 'model.absent_gene_embedding' in blind_trainer.state_dict()
    null = model.absent_gene_embedding
    assert not null.requires_grad
    assert torch.isclose(null.norm(), torch.tensor(32.0 ** 0.5), atol=1e-4)
    result = blind_trainer._run_one_timestep(make_fake_batch(), time_step=1)
    result['total_loss'].backward()
    assert null.grad is None
    # Legacy trainer default keeps the old objective: absent weight 0.
    assert blind_trainer.lambda_absent == 0.5


def test_absent_loss_targets_absent_slots_only(blind_trainer):
    batch = make_fake_batch()
    result = blind_trainer._run_one_timestep(batch, time_step=1)
    present = result['_is_present']
    absent = result['_absent_mask']
    valid = result['_is_valid']
    assert valid.all()
    assert not (present & absent).any()
    assert (present | absent).all()                             # no mask-CE here
    assert absent.any() and present.any()

    out = result['_out']
    per_query_mse = (out['z_hat_gene_raw'] - out['z_tgt_gene_raw']).pow(2).mean(-1)
    expected_gene = (per_query_mse * present).sum() / present.sum()
    expected_absent = (per_query_mse * absent).sum() / absent.sum()
    assert torch.allclose(result['gene_loss'], expected_gene, atol=1e-5)
    assert torch.allclose(result['absent_loss'], expected_absent, atol=1e-5)
    # Absent targets are the null itself.
    null = blind_trainer.model.absent_gene_embedding
    assert torch.allclose(out['z_tgt_gene_raw'][absent], null.expand_as(out['z_tgt_gene_raw'][absent]))
    # And the null enters the objective with lambda_absent.
    others = (
        blind_trainer.lambda_gene * result['gene_loss']
        + blind_trainer.lambda_absent * result['absent_loss']
        + blind_trainer.lambda_cell * result['cell_loss']
        + blind_trainer.lambda_contrastive * result['contrastive_loss']
        + blind_trainer.lambda_mask_ce * result['mask_ce_loss']
    )
    assert torch.allclose(result['total_loss'], others, atol=1e-5)
    for key in ('absent_cos_null', 'present_cos_null', 'quiz_absent_frac',
                'quiz_p_present_neg', 'quiz_p_present_pos'):
        assert torch.isfinite(result[key]), key
    assert 0.0 <= result['quiz_absent_frac'] <= 1.0


def test_load_checkpoint_drops_surplus_src_pos_rows(tiny_trainer):
    """A ckpt saved with one extra PE row (1 Oct 14:30 run) still loads."""
    state = {k: v.clone() for k, v in tiny_trainer.state_dict().items()}
    weight = state['model.query_src_pos_embedding.weight']
    extra = torch.cat([weight, torch.full((1, weight.size(1)), 7.0)], dim=0)
    state['model.query_src_pos_embedding.weight'] = extra
    state['model.mask_src_pos_embedding.weight'] = extra.clone()
    tiny_trainer.on_load_checkpoint({'state_dict': state})
    tiny_trainer.load_state_dict(state)
    assert tiny_trainer.model.query_src_pos_embedding.weight.size(0) == weight.size(0)
    assert torch.equal(tiny_trainer.model.query_src_pos_embedding.weight, weight)


def test_legacy_default_has_no_absent_loss(tiny_trainer):
    assert tiny_trainer.query_sampling == 'target'
    assert tiny_trainer.lambda_absent == 0.0
    result = tiny_trainer._run_one_timestep(make_fake_batch(), time_step=1)
    assert torch.isfinite(result['absent_loss'])
    expected = (
        tiny_trainer.lambda_gene * result['gene_loss']
        + tiny_trainer.lambda_cell * result['cell_loss']
    )
    assert torch.allclose(result['total_loss'], expected, atol=1e-5)


def test_source_blind_encode_one_timestep_masks(blind_trainer):
    result = blind_trainer.encode_one_timestep(make_fake_batch(), time_step=1)
    assert '_is_valid' in result and '_absent_mask' in result
    assert not (result['_is_present'] & result['_absent_mask']).any()
    assert not result['_quiz']['is_mask_ce'].any()


def test_nonsrc_freq_table_roundtrip_and_lookup(tmp_path):
    tokenid_to_rowid, _, _ = make_id_maps()
    map_path = tmp_path / 'tokenid_to_rowid_test.pkl'
    with open(map_path, 'wb') as f:
        pickle.dump(tokenid_to_rowid, f)
    vocab = FIRST_REAL_GENE_LOCAL_ID + N_GENES
    table = torch.zeros(2, 4, vocab)
    table[0, 1, 40] = 1.0                                       # type A loves gene 40 at t1
    table[1, 1, 41] = 1.0                                       # type B loves gene 41 at t1
    common = dict(
        jepa_encoder='cell', n_queries=8, pred_tps=[1], n_total_tps=3,
        tokenid_to_rowid_path=str(map_path), output_dir=str(tmp_path / 'out'),
        tgt_vocab_size=60, d_model=32, num_heads=4, num_layers=1, d_ff=64,
        max_seq_length=64, seed=0, cell_pool='cls', query_sampling='source_blind',
        lambda_absent=0.5, nonsrc_weight_power=1.0, nonsrc_weight_floor=0.0,
    )
    src = GeneQueryJEPATrainer(
        nonsrc_celltypes=['A', 'B'], nonsrc_gene_freq=table, **common
    )
    state = src.state_dict()
    assert 'nonsrc_gene_freq' in state
    assert 'nonsrc_gene_freq' not in src.hparams

    # A fresh trainer (as load_from_checkpoint builds it) gets the table back.
    fresh = GeneQueryJEPATrainer(nonsrc_celltypes=['A', 'B'], **common)
    assert fresh.nonsrc_gene_freq is None
    fresh.on_load_checkpoint({'state_dict': state})
    fresh.load_state_dict(state)
    assert torch.equal(fresh.nonsrc_gene_freq, table)

    batch = {'cell_type_harmonized_t1': ['A', 'B', 'unknown']}
    weights = fresh._nonsrc_weights_for_batch(batch, time_step=1, batch_size=3)
    assert weights.shape == (3, vocab)
    assert weights[0, 40] == 1.0 and weights[0, 41] == 0.0
    assert weights[1, 41] == 1.0 and weights[1, 40] == 0.0
    # Unknown cell type -> mean over the table rows.
    assert torch.isclose(weights[2, 40], torch.tensor(0.5))
    # No cell types in the batch -> mean row for everyone.
    weights = fresh._nonsrc_weights_for_batch({}, time_step=1, batch_size=2)
    assert torch.isclose(weights[1, 41], torch.tensor(0.5))
    # Legacy sampler never weights.
    legacy = GeneQueryJEPATrainer(**{**common, 'query_sampling': 'target'})
    assert legacy._nonsrc_weights_for_batch(batch, 1, 3) is None
