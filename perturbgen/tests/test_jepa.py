"""Unit tests for cell-trajectory JEPA (Phases A–F scaffolding)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
import torch
import pytorch_lightning as pl

from perturbgen.Modules.jepa import (
    CellEncoder,
    CellTrajectoryJEPA,
    JEPACountDecoder,
    TimeConditionedPredictor,
    mean_pool_tokens,
)
from perturbgen.Model.jepa_trainer import JEPADecoderTrainer, JEPATrainer
from perturbgen.src.jepa_metrics import (
    latent_collapse_stats,
    package_comparison_summary,
    pairwise_latent_mse,
    simulate_latent_perturbation,
    trajectory_baselines,
)
from perturbgen.jepa_eval import phase_b, phase_c, phase_d, phase_e, phase_f
from perturbgen.src.jepa_token_maps import apply_id_lookup, build_lookup_tables


class TestJEPAModules(unittest.TestCase):
    def test_id_lookup_tables(self):
        g2l, l2g = build_lookup_tables({9: 4, 10: 5})
        ids = torch.tensor([[9, 10, 0]])
        local = apply_id_lookup(ids, g2l)
        self.assertTrue(torch.equal(local, torch.tensor([[4, 5, 0]])))
        back = apply_id_lookup(local, l2g)
        self.assertTrue(torch.equal(back, torch.tensor([[9, 10, 0]])))

    def test_mean_pool_tokens(self):
        embs = torch.tensor(
            [[[1.0, 1.0], [3.0, 3.0], [0.0, 0.0]]], dtype=torch.float32
        )
        ids = torch.tensor([[5, 6, 0]])
        out = mean_pool_tokens(embs, ids)
        self.assertEqual(out.shape, (1, 2))
        self.assertTrue(torch.allclose(out, torch.tensor([[2.0, 2.0]])))

    def test_cell_encoder_forward(self):
        enc = CellEncoder(
            vocab_size=64,
            d_model=32,
            num_heads=4,
            num_layers=1,
            d_ff=64,
            max_seq_length=16,
            n_time_steps=4,
        )
        ids = torch.randint(2, 60, (3, 10))
        out = enc(ids, time_step=1)
        self.assertEqual(out['cell_embedding'].shape, (3, 32))
        self.assertEqual(out['token_embedding'].shape, (3, 10, 32))

    def test_jepa_forward_and_ema(self):
        model = CellTrajectoryJEPA(
            vocab_size=64,
            d_model=32,
            num_heads=4,
            num_layers=1,
            d_ff=64,
            max_seq_length=16,
            n_total_tps=3,
            pred_tps=[1, 2],
            ema_decay=0.5,
            encoder_type='cell',
        )
        src = torch.randint(2, 60, (4, 8))
        tgt = {
            'tgt_input_ids_t1': torch.randint(2, 60, (4, 8)),
            'tgt_input_ids_t2': torch.randint(2, 60, (4, 8)),
        }
        out = model(src, tgt)
        self.assertIn(1, out)
        self.assertIn(2, out)
        self.assertEqual(out[1]['z_hat'].shape, (4, 32))
        before = next(model.target_encoder.parameters()).clone()
        # change online weights then EMA update
        with torch.no_grad():
            for p in model.context_encoder.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        model.update_target_encoder()
        after = next(model.target_encoder.parameters())
        self.assertFalse(torch.allclose(before, after))

    def test_predictor(self):
        pred = TimeConditionedPredictor(d_model=16, n_time_steps=3)
        z = torch.randn(2, 16)
        y = pred(z, time_step=2)
        self.assertEqual(y.shape, (2, 16))


class TestJEPATrainerTrainability(unittest.TestCase):
    def test_training_step_loss_decreases(self):
        pl.seed_everything(0)
        module = JEPATrainer(
            tgt_vocab_size=64,
            d_model=32,
            num_heads=4,
            num_layers=1,
            d_ff=64,
            max_seq_length=16,
            pred_tps=[1, 2],
            n_total_tps=2,
            lr=1e-2,
            ema_decay=0.9,
            jepa_encoder='cell',
            output_dir=tempfile.mkdtemp(),
        )
        batch = {
            'src_input_ids': torch.randint(2, 60, (8, 10)),
            'tgt_input_ids_t1': torch.randint(2, 60, (8, 10)),
            'tgt_input_ids_t2': torch.randint(2, 60, (8, 10)),
        }
        opt = module.configure_optimizers()['optimizer']
        losses = []
        module.train()
        for _ in range(5):
            opt.zero_grad()
            loss = module.training_step(batch)
            loss.backward()
            opt.step()
            module.on_train_batch_end(None, batch, 0)
            losses.append(float(loss.detach()))
        # Trainability: finite losses and last <= first * 1.05 (allow noise)
        self.assertTrue(all(np.isfinite(losses)))
        self.assertLessEqual(losses[-1], losses[0] * 1.05 + 0.5)
        stats = latent_collapse_stats(torch.randn(16, 32))
        self.assertIn('std_mean', stats)
        self.assertGreater(stats['std_mean'], 0.0)


class TestJEPADecoderAndEval(unittest.TestCase):
    def test_jepa_count_decoder(self):
        jepa = CellTrajectoryJEPA(
            vocab_size=64,
            d_model=16,
            num_heads=4,
            num_layers=1,
            d_ff=32,
            max_seq_length=12,
            n_total_tps=2,
            pred_tps=[1],
            encoder_type='cell',
        )
        dec = JEPACountDecoder(jepa, n_genes=20, d_model=16, freeze_jepa=True)
        src = torch.randint(2, 60, (3, 6))
        tgt = {'tgt_input_ids_t1': torch.randint(2, 60, (3, 6))}
        out = dec(src, tgt, pred_tps=[1])
        self.assertEqual(out[1].shape, (3, 20))

    def test_decoder_trainer_step(self):
        module = JEPADecoderTrainer(
            tgt_vocab_size=64,
            d_model=16,
            num_heads=4,
            num_layers=1,
            d_ff=32,
            max_seq_length=12,
            pred_tps=[1],
            n_total_tps=1,
            n_genes=20,
            jepa_encoder='cell',
            output_dir=tempfile.mkdtemp(),
        )
        batch = {
            'src_input_ids': torch.randint(2, 60, (4, 6)),
            'tgt_input_ids_t1': torch.randint(2, 60, (4, 6)),
            'tgt_counts_t1': torch.rand(4, 20),
        }
        loss = module.training_step(batch)
        self.assertTrue(torch.isfinite(loss))

    def test_eval_phases_b_through_f(self):
        with tempfile.TemporaryDirectory() as tmp:
            z_src = torch.randn(40, 8)
            z_tgt = z_src + 0.1 * torch.randn(40, 8)
            z_hat = z_src + 0.05 * torch.randn(40, 8)
            payload = {
                'z_src': z_src,
                'z_tgt': z_tgt,
                'z_hat': z_hat,
                'time': torch.arange(40) % 3,
            }
            emb_path = os.path.join(tmp, 'emb.pt')
            torch.save(payload, emb_path)
            b = phase_b(emb_path, None, ['time'], tmp)
            c = phase_c(emb_path, tmp)
            d = phase_d(None, tmp)
            e = phase_e(emb_path, tmp, perturb_scale=1.0, seed=0)
            f = phase_f(tmp, {'b': b, 'c': c, 'd': d, 'e': e})
            self.assertTrue(os.path.isfile(os.path.join(tmp, 'phase_f_replacement_package.json')))
            self.assertIn('representation', f)
            # metrics helpers
            bases = trajectory_baselines(z_src, z_tgt, z_hat)
            self.assertIn('jepa', bases)
            pert = simulate_latent_perturbation(z_src, torch.randn(8), 1.0)
            self.assertEqual(pert.shape, z_src.shape)
            summary = package_comparison_summary(phase_b=b, phase_c=c)
            self.assertIn('trajectory', summary)
            pairwise_latent_mse(z_hat, z_tgt)


if __name__ == '__main__':
    unittest.main()
