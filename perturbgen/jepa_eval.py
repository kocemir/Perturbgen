"""CLI for JEPA research Phases B–F: representation, trajectory, apps, package."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

import numpy as np
import torch

from perturbgen.src.jepa_metrics import (
    compare_representation_quality,
    neighbor_purity,
    package_comparison_summary,
    silhouette_safe,
    simulate_latent_perturbation,
    trajectory_baselines,
)
def get_args(args=None):
    parser = argparse.ArgumentParser(description='JEPA evaluation toolkit')
    parser.add_argument(
        '--phase',
        type=str,
        required=True,
        choices=['b', 'c', 'd', 'e', 'f', 'all'],
        help='Research phase to run',
    )
    parser.add_argument(
        '--jepa_embeddings',
        type=str,
        default=None,
        help='Path to jepa_cell_embeddings.pt from JEPATrainer.test',
    )
    parser.add_argument(
        '--masking_embeddings',
        type=str,
        default=None,
        help='Optional MaskGIT/PerturbGen embedding .pt or .npy for Phase B',
    )
    parser.add_argument(
        '--label_key',
        type=str,
        default='time',
        help='Primary label key inside embedding payload',
    )
    parser.add_argument(
        '--extra_label_keys',
        nargs='*',
        default=[],
        help='Additional label keys for probes',
    )
    parser.add_argument(
        '--generation_metrics_json',
        type=str,
        default=None,
        help='Optional JSON with Phase D generation metrics (mse/mmd/emd)',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./T_perturb/res/jepa/eval',
    )
    parser.add_argument(
        '--perturb_scale',
        type=float,
        default=1.0,
        help='Phase E latent perturbation scale',
    )
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(args)


def _load_payload(path: str) -> Dict:
    obj = torch.load(path, map_location='cpu')
    if not isinstance(obj, dict):
        raise ValueError(f'Expected dict payload at {path}')
    return obj


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def phase_b(jepa_path: str, masking_path: Optional[str], label_keys, out_dir: str):
    payload = _load_payload(jepa_path)
    emb_jepa = _as_numpy(payload.get('z_tgt', payload.get('z_src')))
    labels = {}
    for key in label_keys:
        if key in payload:
            labels[key] = _as_numpy(payload[key]).reshape(-1)
    if not labels:
        # fallback: use time if present
        if 'time' in payload:
            labels['time'] = _as_numpy(payload['time']).reshape(-1)

    report = {}
    if masking_path is None:
        # self-metrics only
        for name, y in labels.items():
            report[name] = {
                'jepa_silhouette': silhouette_safe(emb_jepa, y),
                'jepa_nn_purity': neighbor_purity(emb_jepa, y),
            }
    else:
        mask_payload = _load_payload(masking_path)
        emb_mask = _as_numpy(
            mask_payload.get('cls_embeddings', mask_payload.get('z_tgt'))
        )
        n = min(len(emb_jepa), len(emb_mask))
        report = compare_representation_quality(
            emb_jepa[:n],
            emb_mask[:n],
            {k: v[:n] for k, v in labels.items()},
        )
    path = os.path.join(out_dir, 'phase_b_representation.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Phase B report → {path}')
    return report


def phase_c(jepa_path: str, out_dir: str):
    payload = _load_payload(jepa_path)
    required = ['z_src', 'z_tgt', 'z_hat']
    for k in required:
        if k not in payload:
            raise KeyError(f'{k} missing in {jepa_path}')
    z_src = torch.as_tensor(_as_numpy(payload['z_src']))
    z_tgt = torch.as_tensor(_as_numpy(payload['z_tgt']))
    z_hat = torch.as_tensor(_as_numpy(payload['z_hat']))
    n = min(len(z_src), len(z_tgt), len(z_hat))
    report = trajectory_baselines(z_src[:n], z_tgt[:n], z_hat[:n])
    # serialize
    report = {k: {mk: float(mv) for mk, mv in v.items()} for k, v in report.items()}
    path = os.path.join(out_dir, 'phase_c_trajectory.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Phase C report → {path}')
    beats_identity = report['jepa']['mse'] < report['identity']['mse']
    print(f"JEPA beats identity baseline: {beats_identity}")
    return report


def phase_d(generation_metrics_json: Optional[str], out_dir: str):
    if generation_metrics_json and os.path.isfile(generation_metrics_json):
        with open(generation_metrics_json) as f:
            report = json.load(f)
    else:
        report = {
            'note': (
                'Train with --train_mode jepa_decoder and log val/mse; '
                'pass --generation_metrics_json once available.'
            ),
            'status': 'pending_metrics',
        }
    path = os.path.join(out_dir, 'phase_d_generation.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Phase D report → {path}')
    return report


def phase_e(jepa_path: str, out_dir: str, perturb_scale: float, seed: int):
    payload = _load_payload(jepa_path)
    z_src = torch.as_tensor(_as_numpy(payload['z_src']), dtype=torch.float32)
    z_tgt = torch.as_tensor(_as_numpy(payload.get('z_tgt', payload['z_src'])))
    rng = np.random.default_rng(seed)
    # gene-program proxy: PCA-less random directions clustered by cosine
    direction = torch.as_tensor(
        rng.normal(size=(z_src.shape[1],)), dtype=torch.float32
    )
    z_pert = simulate_latent_perturbation(z_src, direction, scale=perturb_scale)
    delta = (z_pert - z_src).norm(dim=-1).mean().item()
    # program discovery proxy: cluster z_tgt with kmeans if sklearn available
    programs = {}
    try:
        from sklearn.cluster import KMeans

        n_clusters = int(min(8, max(2, z_tgt.shape[0] // 50)))
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = km.fit_predict(_as_numpy(z_tgt))
        programs = {
            'n_programs': n_clusters,
            'cluster_sizes': {
                str(i): int((labels == i).sum()) for i in range(n_clusters)
            },
        }
    except Exception as exc:  # noqa: BLE001
        programs = {'error': str(exc)}

    report = {
        'latent_perturbation_mean_delta': float(delta),
        'perturb_scale': perturb_scale,
        'programs': programs,
    }
    path = os.path.join(out_dir, 'phase_e_applications.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    # save perturbed latents for downstream atlas plotting
    torch.save(
        {'z_src': z_src, 'z_pert': z_pert, 'direction': direction},
        os.path.join(out_dir, 'phase_e_latent_perturbation.pt'),
    )
    print(f'Phase E report → {path}')
    return report


def phase_f(out_dir: str, reports: Dict):
    summary = package_comparison_summary(
        phase_b=reports.get('b'),
        phase_c=reports.get('c'),
        phase_d=reports.get('d'),
        phase_e=reports.get('e'),
    )
    path = os.path.join(out_dir, 'phase_f_replacement_package.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Phase F package → {path}')
    return summary


def main(args=None):
    args = get_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    label_keys = [args.label_key] + list(args.extra_label_keys)
    reports = {}

    run_all = args.phase == 'all'
    if run_all or args.phase == 'b':
        if not args.jepa_embeddings:
            raise ValueError('--jepa_embeddings required for phase b/all')
        reports['b'] = phase_b(
            args.jepa_embeddings,
            args.masking_embeddings,
            label_keys,
            args.output_dir,
        )
    if run_all or args.phase == 'c':
        if not args.jepa_embeddings:
            raise ValueError('--jepa_embeddings required for phase c/all')
        reports['c'] = phase_c(args.jepa_embeddings, args.output_dir)
    if run_all or args.phase == 'd':
        reports['d'] = phase_d(args.generation_metrics_json, args.output_dir)
    if run_all or args.phase == 'e':
        if not args.jepa_embeddings:
            raise ValueError('--jepa_embeddings required for phase e/all')
        reports['e'] = phase_e(
            args.jepa_embeddings,
            args.output_dir,
            args.perturb_scale,
            args.seed,
        )
    if run_all or args.phase == 'f':
        # load any existing phase jsons if running f alone
        if not reports:
            for key, fname in [
                ('b', 'phase_b_representation.json'),
                ('c', 'phase_c_trajectory.json'),
                ('d', 'phase_d_generation.json'),
                ('e', 'phase_e_applications.json'),
            ]:
                path = os.path.join(args.output_dir, fname)
                if os.path.isfile(path):
                    with open(path) as f:
                        reports[key] = json.load(f)
        reports['f'] = phase_f(args.output_dir, reports)
    return reports


if __name__ == '__main__':
    main()
