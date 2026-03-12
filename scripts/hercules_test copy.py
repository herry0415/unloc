#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HeRCULES Test/Inference Script

Validates the model on HeRCULES test data with all 3 modalities.

Tests:
1. Data loading (HerculesFusion + hercules_cylinder_dataset)
2. Model creation
3. Checkpoint loading
4. LiDAR inference
5. Radar inference
6. Camera inference
7. Full multi-modal inference
8. Gradient flow

Usage:
    cd <project_root>
    python scripts/hercules_test.py --sequence Library --checkpoint checkpoints/hercules_best.pt
"""

import os
import sys
import argparse
import time

# ============================================================
# Setup paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Parse our arguments first
test_parser = argparse.ArgumentParser(description='HeRCULES inference test')
test_parser.add_argument('--sequence', type=str, default='Library',
                         choices=['Library', 'Sports'],
                         help='Sequence to test on')
test_parser.add_argument('--data_root', type=str, default='/data/drj/HeRCULES/',
                         help='Path to HeRCULES dataset')
test_parser.add_argument('--config', type=str,
                         default=os.path.join(PROJECT_ROOT, 'config', 'hercules_fusion.yaml'),
                         help='Path to config file')
test_parser.add_argument('--checkpoint', type=str, default=None,
                         help='Path to model checkpoint')
test_parser.add_argument('--batch_size', type=int, default=4,
                         help='Batch size for inference')
test_args = test_parser.parse_args()

# Now override sys.argv for Combinedmodel.py
sys.argv = [sys.argv[0], '-y', test_args.config]

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ============================================================
# Helper Functions
# ============================================================

def check_tensor(name, tensor):
    """Check tensor for NaN/Inf."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    status = "✓ OK"
    if has_nan:
        status = "✗ NaN DETECTED"
    elif has_inf:
        status = "✗ Inf DETECTED"

    print(f"  {name:30s} shape={str(tuple(tensor.shape)):20s} "
          f"range=[{tensor.min().item():8.4f}, {tensor.max().item():8.4f}] {status}")

    return not has_nan and not has_inf


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============================================================
# Test Functions
# ============================================================

def test_data_loading():
    """Test 1: Data loading."""
    print_section("Test 1: Data Loading")

    from data.hercules_fusion import HerculesFusion
    from dataloader.hercules_dataset import hercules_cylinder_dataset, collate_fn_BEV

    # Load datasets
    val_pc_dataset = HerculesFusion(
        data_root=test_args.data_root,
        sequence_name=test_args.sequence,
        split='val'
    )

    val_cyl_dataset = hercules_cylinder_dataset(
        val_pc_dataset,
        grid_size=[480, 360, 32],
        fixed_volume_space=False
    )

    val_loader = DataLoader(
        val_cyl_dataset,
        batch_size=min(test_args.batch_size, len(val_cyl_dataset)),
        collate_fn=collate_fn_BEV,
        shuffle=False,
        num_workers=0
    )

    print(f"  Validation dataset size: {len(val_cyl_dataset)}")
    print(f"  Batch size: {test_args.batch_size}")

    # Load a batch
    for batch_idx, data in enumerate(val_loader):
        actual_bs = data[10].shape[0]
        print(f"\n  Sample batch ({actual_bs} items):")
        print(f"    [0] voxel_pos_l:   {data[0].shape if isinstance(data[0], torch.Tensor) else 'list'}")
        print(f"    [1] grid_ind_l:    list of {len(data[1])} arrays")
        print(f"    [2] fea_l:         list of {len(data[2])} arrays, "
              f"first shape={data[2][0].shape if len(data[2]) > 0 else 'empty'}")
        print(f"    [3] voxel_pos_r:   {data[3].shape if isinstance(data[3], torch.Tensor) else 'list'}")
        print(f"    [4] grid_ind_r:    list of {len(data[4])} arrays")
        print(f"    [5] fea_r:         list of {len(data[5])} arrays, "
              f"first shape={data[5][0].shape if len(data[5]) > 0 else 'empty'}")
        print(f"    [6] mono_left:     {data[6].shape}")
        print(f"    [7] mono_right:    {data[7].shape}")
        print(f"    [8] mono_rear:     {data[8].shape}")
        print(f"    [9] radar_2d:      {data[9].shape}")
        print(f"    [10] pose:         {data[10].shape}")
        print("  ✓ Data loading successful")
        break

    return val_loader


def test_model_creation(device):
    """Test 2: Model creation."""
    print_section("Test 2: Model Creation")

    from FusionModel import Fusionmodel

    model = Fusionmodel()
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print(f"  Sub-models:")
    print(f"    - LiDAR model:  {type(model.lidarmodel).__name__}")
    print(f"    - Image model:  {type(model.imagemodel).__name__}")
    print(f"    - Radar model:  {type(model.radarmodel).__name__}")
    print(f"    - Regression:   {type(model.regression).__name__}")

    print("  ✓ Model created successfully")
    return model


def test_checkpoint_loading(model, device):
    """Test 3: Checkpoint loading."""
    print_section("Test 3: Checkpoint Loading")

    if test_args.checkpoint is None:
        print("  (Skipping - no checkpoint specified, using random init)")
        return model

    if not os.path.exists(test_args.checkpoint):
        print(f"  WARNING: Checkpoint not found: {test_args.checkpoint}")
        print("  Continuing with random initialization")
        return model

    try:
        state_dict = torch.load(test_args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  ✓ Loaded: {test_args.checkpoint}")
    except Exception as e:
        print(f"  ERROR loading checkpoint: {e}")
        return model

    return model


def test_lidar_inference(model, batch, device, batch_size):
    """Test 4: LiDAR inference."""
    print_section("Test 4: LiDAR Branch Inference")

    all_ok = True

    # Left LiDAR
    print("  [Left LiDAR]")
    vox_tenl = [torch.from_numpy(i).to(device) for i in batch[1]]
    pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in batch[2]]

    try:
        with torch.no_grad():
            t0 = time.time()
            trans_l, rot_l = model([pt_fea_tenl, vox_tenl, batch_size])
            t1 = time.time()

        all_ok &= check_tensor("Translation", trans_l)
        all_ok &= check_tensor("Rotation", rot_l)
        print(f"    Inference time: {(t1-t0)*1000:.1f}ms")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        all_ok = False

    # Right LiDAR (Radar)
    print("\n  [Right LiDAR/Radar]")
    vox_tenr = [torch.from_numpy(i).to(device) for i in batch[4]]
    pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in batch[5]]

    try:
        with torch.no_grad():
            t0 = time.time()
            trans_r, rot_r = model([pt_fea_tenr, vox_tenr, batch_size])
            t1 = time.time()

        all_ok &= check_tensor("Translation", trans_r)
        all_ok &= check_tensor("Rotation", rot_r)
        print(f"    Inference time: {(t1-t0)*1000:.1f}ms")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        all_ok = False

    status = "✓ PASSED" if all_ok else "✗ FAILED"
    print(f"\n  [{status}] LiDAR inference")
    return all_ok


def test_camera_inference(model, batch, device):
    """Test 5: Camera inference."""
    print_section("Test 5: Camera Branch Inference")

    all_ok = True
    camera_names = ['mono_left', 'mono_right', 'mono_rear']
    camera_indices = [6, 7, 8]

    for name, idx in zip(camera_names, camera_indices):
        print(f"  [{name}]")
        img = torch.from_numpy(batch[idx]).float().to(device)

        try:
            with torch.no_grad():
                t0 = time.time()
                trans, rot = model([img])
                t1 = time.time()

            all_ok &= check_tensor(f"Translation ({name})", trans)
            all_ok &= check_tensor(f"Rotation ({name})", rot)
            print(f"    Inference time: {(t1-t0)*1000:.1f}ms\n")
        except Exception as e:
            print(f"    ✗ Error: {e}\n")
            all_ok = False

    status = "✓ PASSED" if all_ok else "✗ FAILED"
    print(f"  [{status}] Camera inference")
    return all_ok


def test_radar_inference(model, batch, device, batch_size):
    """Test 6: Radar inference."""
    print_section("Test 6: Radar 3D Point Cloud Inference")

    print("  [Radar as point cloud]")
    vox_tenr = [torch.from_numpy(i).to(device) for i in batch[4]]
    pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in batch[5]]

    all_ok = True
    try:
        with torch.no_grad():
            t0 = time.time()
            trans, rot = model([pt_fea_tenr, vox_tenr, batch_size])
            t1 = time.time()

        all_ok &= check_tensor("Translation (Radar)", trans)
        all_ok &= check_tensor("Rotation (Radar)", rot)
        print(f"    Inference time: {(t1-t0)*1000:.1f}ms")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        all_ok = False

    status = "✓ PASSED" if all_ok else "✗ FAILED"
    print(f"  [{status}] Radar inference")
    return all_ok


def test_multi_modal_fusion(model, batch, device, batch_size):
    """Test 7: Full multi-modal pipeline."""
    print_section("Test 7: Full Multi-Modal Pipeline")

    all_ok = True

    vox_tenl = [torch.from_numpy(i).to(device) for i in batch[1]]
    pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in batch[2]]

    vox_tenr = [torch.from_numpy(i).to(device) for i in batch[4]]
    pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in batch[5]]

    monoleft = torch.from_numpy(batch[6]).float().to(device)

    pose = torch.from_numpy(batch[10]).float().to(device)

    try:
        with torch.no_grad():
            t0 = time.time()

            # 3 forward passes
            trans_l, rot_l = model([pt_fea_tenl, vox_tenl, batch_size])
            trans_r, rot_r = model([pt_fea_tenr, vox_tenr, batch_size])
            trans_c, rot_c = model([monoleft])

            t1 = time.time()

        print(f"  LiDAR output:   trans shape={trans_l.shape}, rot shape={rot_l.shape}")
        print(f"  Radar output:   trans shape={trans_r.shape}, rot shape={rot_r.shape}")
        print(f"  Camera output:  trans shape={trans_c.shape}, rot shape={rot_c.shape}")
        print(f"  Ground truth:   pose shape={pose.shape}")
        print(f"  Total inference time (3 passes): {(t1-t0)*1000:.1f}ms")

        all_ok &= check_tensor("LiDAR Trans", trans_l)
        all_ok &= check_tensor("LiDAR Rot", rot_l)
        all_ok &= check_tensor("Radar Trans", trans_r)
        all_ok &= check_tensor("Radar Rot", rot_r)
        all_ok &= check_tensor("Camera Trans", trans_c)
        all_ok &= check_tensor("Camera Rot", rot_c)

    except Exception as e:
        print(f"  ✗ Error in multi-modal pipeline: {e}")
        all_ok = False

    status = "✓ PASSED" if all_ok else "✗ FAILED"
    print(f"  [{status}] Multi-modal fusion")
    return all_ok


def test_gradient_flow(model, batch, device, batch_size):
    """Test 8: Gradient flow."""
    print_section("Test 8: Gradient Flow & Backpropagation")

    all_ok = True

    vox_tenl = [torch.from_numpy(i).to(device) for i in batch[1]]
    pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in batch[2]]

    pose = torch.from_numpy(batch[10]).float().to(device)

    try:
        # Forward pass
        trans, rot = model([pt_fea_tenl, vox_tenl, batch_size])

        # Simple loss
        loss = trans.mean() + rot.mean()

        # Backward
        loss.backward()

        # Check gradients
        model_has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_nan_grad = torch.isnan(param.grad).any().item()
                has_inf_grad = torch.isinf(param.grad).any().item()

                if has_nan_grad or has_inf_grad:
                    print(f"  ✗ Gradient issue in {name}: NaN={has_nan_grad}, Inf={has_inf_grad}")
                    all_ok = False
                else:
                    model_has_grad = True

        if model_has_grad:
            print("  ✓ Gradients computed successfully")
            print(f"  ✓ Loss value: {loss.item():.6f}")
        else:
            print("  ✗ No gradients computed")
            all_ok = False

    except Exception as e:
        print(f"  ✗ Error in gradient flow: {e}")
        all_ok = False

    status = "✓ PASSED" if all_ok else "✗ FAILED"
    print(f"  [{status}] Gradient flow")
    return all_ok


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  HERCULES INFERENCE TEST")
    print("=" * 70)
    print(f"  Config: {test_args.config}")
    print(f"  Sequence: {test_args.sequence}")
    print(f"  Checkpoint: {test_args.checkpoint if test_args.checkpoint else '(random init)'}")
    print(f"  Batch size: {test_args.batch_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Run tests
    results = {}

    try:
        val_loader = test_data_loading()
        results['data_loading'] = True
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        results['data_loading'] = False
        return

    try:
        model = test_model_creation(device)
        results['model_creation'] = True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        results['model_creation'] = False
        return

    try:
        model = test_checkpoint_loading(model, device)
        results['checkpoint_loading'] = True
    except Exception as e:
        print(f"  ✗ Checkpoint loading failed: {e}")
        results['checkpoint_loading'] = False

    model.eval()

    # Get a batch for inference tests
    for batch in val_loader:
        batch_size = batch[10].shape[0]
        break

    results['lidar_inference'] = test_lidar_inference(model, batch, device, batch_size)
    results['camera_inference'] = test_camera_inference(model, batch, device)
    results['radar_inference'] = test_radar_inference(model, batch, device, batch_size)
    results['multi_modal'] = test_multi_modal_fusion(model, batch, device, batch_size)
    results['gradient_flow'] = test_gradient_flow(model, batch, device, batch_size)

    # Summary
    print_section("TEST SUMMARY")
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:25s}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n  ✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("\n  ✗✗✗ SOME TESTS FAILED ✗✗✗")

    print("=" * 70)


if __name__ == '__main__':
    main()
