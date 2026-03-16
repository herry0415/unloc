#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HeRCULES Single-Modality Evaluation Script

Evaluates trained FusionModel on HeRCULES test data with a SINGLE modality
(LiDAR or Radar). Outputs quantitative metrics:
- Mean/Median Translation Error (ATE) in meters
- Mean/Median Rotation Error (ARE) in degrees
- Trajectory visualization
- Error distribution plots

Usage:
    python scripts/hercules_test_single.py \
        --modality lidar \
        --checkpoint checkpoints/lidar_only/hercules_best_lidar.pt \
        --output_dir results/hercules_eval

    python scripts/hercules_test_single.py \
        --modality radar \
        --checkpoint checkpoints/radar_only/hercules_best_radar.pt \
        --output_dir results/hercules_eval
"""

import os
import sys
import argparse
import time
import numpy as np

# ============================================================
# Setup paths BEFORE any project imports
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Parse arguments
test_parser = argparse.ArgumentParser(description='HeRCULES Single-Modality Evaluation')
test_parser.add_argument('--modality', type=str, default='lidar',
                         choices=['lidar', 'radar'],
                         help='Which modality to evaluate')
test_parser.add_argument('--sequence', type=str, default='Library',
                         choices=['Library', 'Sports'],
                         help='Sequence to evaluate')
test_parser.add_argument('--data_root', type=str, default='/data/drj/HeRCULES/',
                         help='Path to HeRCULES dataset')
test_parser.add_argument('--config', type=str,
                         default=os.path.join(PROJECT_ROOT, 'config', 'hercules_fusion.yaml'),
                         help='Path to config file')
test_parser.add_argument('--checkpoint', type=str, required=True,
                         help='Path to model checkpoint')
test_parser.add_argument('--batch_size', type=int, default=1,
                         help='Batch size for evaluation')
test_parser.add_argument('--output_dir', type=str, default='results/hercules_eval',
                         help='Directory to save evaluation results')
test_parser.add_argument('--gpu', type=int, default=0,
                         help='GPU ID to use')
test_parser.add_argument('--split', type=str, default='val',
                         choices=['val', 'train'],
                         help='Data split to evaluate on')
test_args = test_parser.parse_args()

# Override sys.argv for Combinedmodel.py
sys.argv = [sys.argv[0], '-y', test_args.config]

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# Pose utility functions
# ============================================================

def qexp(q):
    """Convert log quaternion (3,) to quaternion (4,)."""
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n / np.pi) * q))
    return q


def val_translation(pred, gt):
    """Translation error: Euclidean distance in meters."""
    return np.linalg.norm(pred - gt)


def val_rotation(pred_q, gt_q):
    """Rotation error: angular distance in degrees."""
    d = abs(np.dot(pred_q, gt_q))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta


# ============================================================
# Evaluation functions
# ============================================================

def load_pose_stats(data_root, sequence_name):
    """Load pose normalization statistics (mean_t, std_t) from training."""
    pose_stats_file = os.path.join(
        data_root, sequence_name,
        f'{sequence_name}_fusion_pose_stats.txt'
    )
    if not os.path.exists(pose_stats_file):
        print(f"  WARNING: Pose stats file not found: {pose_stats_file}")
        print(f"  Using zeros for mean and ones for std (no de-normalization)")
        return np.zeros(3), np.ones(3)

    stats = np.loadtxt(pose_stats_file)
    mean_t = stats[0]
    std_t = stats[1]
    print(f"  Pose stats loaded: mean_t={mean_t}, std_t={std_t}")
    return mean_t, std_t


def build_test_loader():
    """Build test data loader."""
    from data.hercules_fusion import HerculesFusion
    from dataloader.hercules_dataset import hercules_cylinder_dataset, collate_fn_BEV

    test_pc_dataset = HerculesFusion(
        data_root=test_args.data_root,
        sequence_name=test_args.sequence,
        split=test_args.split
    )
    test_cyl_dataset = hercules_cylinder_dataset(
        test_pc_dataset,
        grid_size=[480, 360, 32],
        fixed_volume_space=False
    )
    test_loader = DataLoader(
        test_cyl_dataset,
        batch_size=test_args.batch_size,
        collate_fn=collate_fn_BEV,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return test_loader, len(test_cyl_dataset)


def build_model(device):
    """Build and load model."""
    from FusionModel import Fusionmodel

    model = Fusionmodel()
    model.to(device)

    # Load checkpoint (handle DDP 'module.' prefix)
    state_dict = torch.load(test_args.checkpoint, map_location=device)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
        print("  Detected DDP checkpoint, removed 'module.' prefix")
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {total_params:,} parameters")
    print(f"  Checkpoint: {test_args.checkpoint}")
    return model


def evaluate_single_modality(model, test_loader, lenset, device, mean_t, std_t,
                             modality='lidar'):
    """
    Evaluate a single modality on all test samples.

    Args:
        modality: 'lidar' or 'radar'

    Returns:
        error_t, error_q, pred_translations, gt_translations, pred_rotations, gt_rotations, times
    """
    gt_translation = np.zeros((lenset, 3))
    pred_translation = np.zeros((lenset, 3))
    gt_rotation = np.zeros((lenset, 4))
    pred_rotation = np.zeros((lenset, 4))
    error_t = np.zeros(lenset)
    error_q = np.zeros(lenset)
    time_results = np.zeros(lenset)

    tqdm_loader = tqdm(test_loader, total=len(test_loader),
                       desc=f'Eval [{modality.upper()}]', ncols=100, ascii=True)

    sample_idx = 0
    with torch.no_grad():
        for step, data in enumerate(tqdm_loader):
            batch_size = data[10].shape[0]
            start_idx = sample_idx
            end_idx = min(sample_idx + batch_size, lenset)
            actual_bs = end_idx - start_idx

            # Ground truth pose: [trans(3), log_quat(3)] normalized
            pose_raw = data[10][:actual_bs]
            pose_gt = pose_raw if isinstance(pose_raw, np.ndarray) else pose_raw.numpy()

            # De-normalize GT translation
            gt_trans_raw = pose_gt[:, :3] * std_t + mean_t
            gt_translation[start_idx:end_idx] = gt_trans_raw

            # GT rotation: log_quat -> quaternion
            for i in range(actual_bs):
                gt_rotation[start_idx + i] = qexp(pose_gt[i, 3:6])

            # Forward pass
            start_time = time.time()

            if modality == 'lidar':
                vox = [torch.from_numpy(i).to(device) for i in data[1]]
                fea = [torch.from_numpy(i).type(torch.FloatTensor).to(device)
                       for i in data[2]]
                trans_pred, rot_pred = model([fea, vox, actual_bs])

            elif modality == 'radar':
                vox = [torch.from_numpy(i).to(device) for i in data[4]]
                fea = [torch.from_numpy(i).type(torch.FloatTensor).to(device)
                       for i in data[5]]
                trans_pred, rot_pred = model([fea, vox, actual_bs])

            end_time = time.time()
            cost_time = (end_time - start_time) / actual_bs

            # Predictions to numpy
            trans_np = trans_pred.cpu().numpy()
            rot_np = rot_pred.cpu().numpy()

            # De-normalize predicted translation
            pred_trans_raw = trans_np * std_t + mean_t
            pred_translation[start_idx:end_idx] = pred_trans_raw[:actual_bs]

            # Predicted rotation: log_quat -> quaternion
            for i in range(actual_bs):
                pred_rotation[start_idx + i] = qexp(rot_np[i])

            # Compute errors
            for i in range(actual_bs):
                idx = start_idx + i
                error_t[idx] = val_translation(
                    pred_translation[idx], gt_translation[idx])
                error_q[idx] = val_rotation(
                    pred_rotation[idx], gt_rotation[idx])

            time_results[start_idx:end_idx] = cost_time
            sample_idx = end_idx

            # Update tqdm
            if sample_idx > 0:
                tqdm_loader.set_postfix({
                    'ATE': f'{np.mean(error_t[:sample_idx]):.3f}m',
                    'ARE': f'{np.mean(error_q[:sample_idx]):.3f}deg'
                })

    return (error_t[:sample_idx], error_q[:sample_idx],
            pred_translation[:sample_idx], gt_translation[:sample_idx],
            pred_rotation[:sample_idx], gt_rotation[:sample_idx],
            time_results[:sample_idx])


def save_trajectory_plot(pred_t, gt_t, modality, save_dir):
    """Save trajectory comparison plot."""
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(gt_t[:, 1], gt_t[:, 0], s=1, c='black', label='Ground Truth')
    plt.scatter(pred_t[:, 1], pred_t[:, 0], s=1, c='red', label='Predicted')
    plt.plot(gt_t[0, 1], gt_t[0, 0], 'y*', markersize=15, label='Start')
    plt.xlabel('Y [m]')
    plt.ylabel('X [m]')
    plt.title(f'Trajectory Comparison - {modality.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    filepath = os.path.join(save_dir, f'trajectory_{modality}.png')
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filepath}")


def save_error_distribution(error_t, error_q, modality, save_dir):
    """Save error distribution plots."""
    # Translation error distribution
    fig = plt.figure(figsize=(10, 4))
    t_num = np.arange(len(error_t))
    plt.scatter(t_num, error_t, s=1, c='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Translation Error (m)')
    plt.title(f'Translation Error Distribution - {modality.upper()}')
    plt.grid(True, alpha=0.3)
    filepath = os.path.join(save_dir, f'error_translation_{modality}.png')
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Rotation error distribution
    fig = plt.figure(figsize=(10, 4))
    q_num = np.arange(len(error_q))
    plt.scatter(q_num, error_q, s=1, c='blue')
    plt.xlabel('Sample Index')
    plt.ylabel('Rotation Error (deg)')
    plt.title(f'Rotation Error Distribution - {modality.upper()}')
    plt.grid(True, alpha=0.3)
    filepath = os.path.join(save_dir, f'error_rotation_{modality}.png')
    fig.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: error_translation_{modality}.png, error_rotation_{modality}.png")


def save_numeric_results(error_t, error_q, pred_t, gt_t, pred_q, modality,
                         save_dir):
    """Save numeric results to text files."""
    np.savetxt(os.path.join(save_dir, f'error_t_{modality}.txt'),
               error_t, fmt='%8.7f')
    np.savetxt(os.path.join(save_dir, f'error_q_{modality}.txt'),
               error_q, fmt='%8.7f')
    np.savetxt(os.path.join(save_dir, f'pred_t_{modality}.txt'),
               pred_t, fmt='%8.7f')
    np.savetxt(os.path.join(save_dir, f'gt_t_{modality}.txt'),
               gt_t, fmt='%8.7f')
    np.savetxt(os.path.join(save_dir, f'pred_q_{modality}.txt'),
               pred_q, fmt='%8.7f')
    print(f"  Saved: error/pred/gt text files for {modality}")


# ============================================================
# Main
# ============================================================

def main():
    modality = test_args.modality

    print("=" * 80)
    print(f"  HERCULES SINGLE-MODALITY EVALUATION [{modality.upper()}]")
    print("=" * 80)
    print(f"  Modality:    {modality.upper()}")
    print(f"  Sequence:    {test_args.sequence}")
    print(f"  Split:       {test_args.split}")
    print(f"  Checkpoint:  {test_args.checkpoint}")
    print(f"  Batch size:  {test_args.batch_size}")
    print(f"  Output dir:  {test_args.output_dir}")

    # Device
    device = torch.device(f'cuda:{test_args.gpu}' if torch.cuda.is_available()
                          else 'cpu')
    print(f"  Device:      {device}")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(device)}")
    print("=" * 80)

    # Create output directory (per modality)
    output_dir = os.path.join(test_args.output_dir, modality)
    os.makedirs(output_dir, exist_ok=True)

    # Load pose stats for de-normalization
    print("\n--- Loading Pose Statistics ---")
    mean_t, std_t = load_pose_stats(test_args.data_root, test_args.sequence)

    # Build data loader
    print("\n--- Building Test Data Loader ---")
    test_loader, lenset = build_test_loader()
    print(f"  Test samples: {lenset}")

    # Build and load model
    print("\n--- Loading Model ---")
    model = build_model(device)

    # Open log file
    log_path = os.path.join(output_dir, 'evaluation_log.txt')
    LOG = open(log_path, 'w')

    def log_string(s):
        LOG.write(s + '\n')
        LOG.flush()
        print(s)

    log_string(f"\nEvaluation: {test_args.sequence} | Split: {test_args.split} | Modality: {modality.upper()}")
    log_string(f"Checkpoint: {test_args.checkpoint}")
    log_string(f"Test samples: {lenset}\n")

    # ============================================================
    # Evaluate single modality
    # ============================================================
    log_string("\n" + "=" * 80)
    log_string(f"  Evaluating: {modality.upper()}")
    log_string("=" * 80)

    error_t, error_q, pred_t, gt_t, pred_q, gt_q, times = \
        evaluate_single_modality(
            model, test_loader, lenset, device, mean_t, std_t,
            modality=modality
        )

    # Print metrics
    log_string(f"\n  --- {modality.upper()} Results ---")
    log_string(f"  Mean  Position Error (m):     {np.mean(error_t):.4f}")
    log_string(f"  Median Position Error (m):    {np.median(error_t):.4f}")
    log_string(f"  Mean  Orientation Error (deg): {np.mean(error_q):.4f}")
    log_string(f"  Median Orientation Error (deg):{np.median(error_q):.4f}")
    log_string(f"  Mean  Inference Time (ms):    {np.mean(times)*1000:.1f}")

    # Save plots and numeric results
    save_trajectory_plot(pred_t, gt_t, modality, output_dir)
    save_error_distribution(error_t, error_q, modality, output_dir)
    save_numeric_results(error_t, error_q, pred_t, gt_t, pred_q, modality,
                         output_dir)

    # ============================================================
    # Summary
    # ============================================================
    log_string("\n" + "=" * 80)
    log_string("  EVALUATION SUMMARY")
    log_string("=" * 80)
    log_string(f"\n  {'Modality':<10} {'Mean ATE(m)':<15} {'Med ATE(m)':<15} "
               f"{'Mean ARE(deg)':<15} {'Med ARE(deg)':<15} {'Time(ms)':<10}")
    log_string("  " + "-" * 78)
    log_string(f"  {modality:<10} {np.mean(error_t):<15.4f} {np.median(error_t):<15.4f} "
               f"{np.mean(error_q):<15.4f} {np.median(error_q):<15.4f} "
               f"{np.mean(times)*1000:<10.1f}")
    log_string("  " + "-" * 78)
    log_string(f"\n  Results saved to: {output_dir}/")
    log_string("=" * 80)

    LOG.close()
    print(f"\n  Log saved to: {log_path}")
    matplotlib.pyplot.close('all')


if __name__ == '__main__':
    main()
