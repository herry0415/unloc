#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Demo Test/Inference Script - Validates the model pipeline with synthetic data.

Tests:
1. Model loading (from checkpoint or random init)
2. Forward pass through all modalities (LiDAR, Camera, Radar)
3. Output dimension verification
4. NaN/Inf detection
5. Inference timing

Usage:
    cd <project_root>
    python scripts/demo_test.py                          # Random init
    python scripts/demo_test.py --checkpoint checkpoints/demo_best.pt  # Load checkpoint
"""

import os
import sys
import time
import argparse as demo_argparse  # renamed to avoid conflict with Combinedmodel.py

# ============================================================
# CRITICAL: Setup paths BEFORE any project imports
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Parse our own args FIRST, then override sys.argv for Combinedmodel.py
demo_parser = demo_argparse.ArgumentParser(description='Demo inference test')
demo_parser.add_argument('--checkpoint', type=str, default=None,
                         help='Path to model checkpoint (default: random init)')
demo_parser.add_argument('--batch_size', type=int, default=2,
                         help='Batch size for inference (default: 2)')
demo_args = demo_parser.parse_args()

# Now set sys.argv for Combinedmodel.py's internal argparse
DEMO_CONFIG = os.path.join(PROJECT_ROOT, 'config', 'demo_config.yaml')
sys.argv = [sys.argv[0], '-y', DEMO_CONFIG]

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn


def check_tensor(name, tensor):
    """Check a tensor for NaN/Inf and print info."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    status = "OK"
    if has_nan:
        status = "NaN DETECTED"
    elif has_inf:
        status = "Inf DETECTED"

    print(f"    {name:25s} shape={str(list(tensor.shape)):20s} "
          f"range=[{tensor.min().item():.4f}, {tensor.max().item():.4f}] "
          f"[{status}]")
    return not has_nan and not has_inf


def test_data_loading():
    """Test that the demo data loader works correctly."""
    print("\n--- Test 1: Data Loading ---")
    from dataloader.demo_dataset import DemoDataset, collate_fn_demo

    dataset = DemoDataset(num_samples=4, batch_size=2)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, collate_fn=collate_fn_demo)

    for batch in loader:
        print(f"  data[0]  (batch_id):   {type(batch[0])}")
        print(f"  data[1]  (vox_tenl):   {len(batch[1])} items, "
              f"shape={batch[1][0].shape}")
        print(f"  data[2]  (pt_fea_tenl): {len(batch[2])} items, "
              f"shape={batch[2][0].shape}")
        print(f"  data[4]  (vox_tenr):   {len(batch[4])} items, "
              f"shape={batch[4][0].shape}")
        print(f"  data[5]  (pt_fea_tenr): {len(batch[5])} items, "
              f"shape={batch[5][0].shape}")
        print(f"  data[6]  (mono_left):  shape={batch[6].shape}, "
              f"dtype={batch[6].dtype}")
        print(f"  data[7]  (mono_right): shape={batch[7].shape}")
        print(f"  data[8]  (mono_rear):  shape={batch[8].shape}")
        print(f"  data[9]  (radar):      shape={batch[9].shape}")
        print(f"  data[10] (labels):     shape={batch[10].shape}")
        print("  [OK] Data loading successful")
        return batch

    return None


def test_model_creation(device):
    """Test that the Fusionmodel can be created."""
    print("\n--- Test 2: Model Creation ---")
    from FusionModel import Fusionmodel

    model = Fusionmodel()
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Print sub-model info
    print(f"  Sub-models:")
    print(f"    - LiDAR model:  {type(model.lidarmodel).__name__}")
    print(f"    - Image model:  {type(model.imagemodel).__name__}")
    print(f"    - Radar model:  {type(model.radarmodel).__name__}")
    print(f"    - Regression:   {type(model.regression).__name__}")
    print("  [OK] Model created successfully")

    return model


def test_checkpoint_loading(model, checkpoint_path, device):
    """Test loading a checkpoint."""
    print(f"\n--- Test 3: Checkpoint Loading ---")
    if checkpoint_path is None:
        print("  Skipping (no checkpoint specified, using random init)")
        return model

    if not os.path.exists(checkpoint_path):
        print(f"  WARNING: Checkpoint not found: {checkpoint_path}")
        print("  Continuing with random initialization")
        return model

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"  Loaded: {checkpoint_path}")
    print("  [OK] Checkpoint loaded successfully")
    return model


def test_lidar_inference(model, batch, device, batch_size):
    """Test LiDAR branch inference."""
    print("\n--- Test 4: LiDAR Branch Inference ---")
    all_ok = True

    # Left LiDAR
    print("  [Left LiDAR]")
    vox_tenl = [torch.from_numpy(i).to(device) for i in batch[1]]
    pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in batch[2]]

    t0 = time.time()
    with torch.no_grad():
        trans_l, rot_l = model([pt_fea_tenl, vox_tenl, batch_size])
    t1 = time.time()

    all_ok &= check_tensor("Translation (left)", trans_l)
    all_ok &= check_tensor("Rotation (left)", rot_l)
    print(f"    Inference time: {(t1-t0)*1000:.1f}ms")

    # Right LiDAR
    print("  [Right LiDAR]")
    vox_tenr = [torch.from_numpy(i).to(device) for i in batch[4]]
    pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in batch[5]]

    t0 = time.time()
    with torch.no_grad():
        trans_r, rot_r = model([pt_fea_tenr, vox_tenr, batch_size])
    t1 = time.time()

    all_ok &= check_tensor("Translation (right)", trans_r)
    all_ok &= check_tensor("Rotation (right)", rot_r)
    print(f"    Inference time: {(t1-t0)*1000:.1f}ms")

    status = "OK" if all_ok else "FAIL"
    print(f"  [{status}] LiDAR inference {'passed' if all_ok else 'FAILED'}")
    return all_ok


def test_camera_inference(model, batch, device, batch_size):
    """Test Camera branch inference."""
    print("\n--- Test 5: Camera Branch Inference ---")
    all_ok = True

    camera_names = ['mono_left', 'mono_right', 'mono_rear']
    camera_indices = [6, 7, 8]

    for name, idx in zip(camera_names, camera_indices):
        print(f"  [{name}]")
        img = torch.from_numpy(batch[idx]).float().to(device)

        t0 = time.time()
        with torch.no_grad():
            trans, rot = model([img])
        t1 = time.time()

        all_ok &= check_tensor(f"Translation ({name})", trans)
        all_ok &= check_tensor(f"Rotation ({name})", rot)
        print(f"    Inference time: {(t1-t0)*1000:.1f}ms")

    status = "OK" if all_ok else "FAIL"
    print(f"  [{status}] Camera inference {'passed' if all_ok else 'FAILED'}")
    return all_ok


def test_radar_inference(model, batch, device, batch_size):
    """Test Radar branch inference."""
    print("\n--- Test 6: Radar Branch Inference ---")
    all_ok = True

    radar = torch.from_numpy(batch[9]).float().reshape(batch_size, 1, 512, 512).to(device)

    t0 = time.time()
    with torch.no_grad():
        trans, rot = model([radar])
    t1 = time.time()

    all_ok &= check_tensor("Translation (radar)", trans)
    all_ok &= check_tensor("Rotation (radar)", rot)
    print(f"    Inference time: {(t1-t0)*1000:.1f}ms")

    status = "OK" if all_ok else "FAIL"
    print(f"  [{status}] Radar inference {'passed' if all_ok else 'FAILED'}")
    return all_ok


def test_full_pipeline(model, batch, device, batch_size):
    """Test the complete forward pass with all modalities combined."""
    print("\n--- Test 7: Full Pipeline (All Modalities) ---")

    vox_tenl = [torch.from_numpy(i).to(device) for i in batch[1]]
    pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in batch[2]]
    vox_tenr = [torch.from_numpy(i).to(device) for i in batch[4]]
    pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in batch[5]]
    monoleft = torch.from_numpy(batch[6]).float().to(device)
    monoright = torch.from_numpy(batch[7]).float().to(device)
    monorear = torch.from_numpy(batch[8]).float().to(device)
    radarimage = torch.from_numpy(batch[9]).float().reshape(batch_size, 1, 512, 512).to(device)
    labels = torch.from_numpy(batch[10]).float()
    transgt = labels[:, 3:6].to(device)
    rotgt = labels[:, 0:3].to(device)

    criterion = nn.MSELoss()

    t0 = time.time()
    with torch.no_grad():
        # All 6 forward passes
        trans1, rot1 = model([pt_fea_tenl, vox_tenl, batch_size])
        trans2, rot2 = model([pt_fea_tenr, vox_tenr, batch_size])
        trans3, rot3 = model([monoleft])
        trans4, rot4 = model([monoright])
        trans5, rot5 = model([monorear])
        trans6, rot6 = model([radarimage])
    t1 = time.time()

    # Compute losses per modality
    loss_lidar_l = criterion(trans1, transgt).item() + criterion(rot1, rotgt).item()
    loss_lidar_r = criterion(trans2, transgt).item() + criterion(rot2, rotgt).item()
    loss_cam_l = criterion(trans3, transgt).item() + criterion(rot3, rotgt).item()
    loss_cam_r = criterion(trans4, transgt).item() + criterion(rot4, rotgt).item()
    loss_cam_re = criterion(trans5, transgt).item() + criterion(rot5, rotgt).item()
    loss_radar = criterion(trans6, transgt).item() + criterion(rot6, rotgt).item()

    total_time = (t1 - t0) * 1000
    per_sample = total_time / batch_size

    print(f"  Per-modality MSE loss:")
    print(f"    LiDAR Left:   {loss_lidar_l:.4f}")
    print(f"    LiDAR Right:  {loss_lidar_r:.4f}")
    print(f"    Camera Left:  {loss_cam_l:.4f}")
    print(f"    Camera Right: {loss_cam_r:.4f}")
    print(f"    Camera Rear:  {loss_cam_re:.4f}")
    print(f"    Radar:        {loss_radar:.4f}")
    print(f"  Total inference time: {total_time:.1f}ms "
          f"({per_sample:.1f}ms/sample)")

    # Check all outputs are valid
    all_outputs = [trans1, rot1, trans2, rot2, trans3, rot3,
                   trans4, rot4, trans5, rot5, trans6, rot6]
    has_nan = any(torch.isnan(t).any().item() for t in all_outputs)
    has_inf = any(torch.isinf(t).any().item() for t in all_outputs)

    if has_nan or has_inf:
        print("  [FAIL] Invalid values detected in outputs")
        return False

    # Check output dimensions
    for t, r in [(trans1, rot1), (trans3, rot3), (trans6, rot6)]:
        assert t.shape == (batch_size, 3), \
            f"Translation shape mismatch: {t.shape} != ({batch_size}, 3)"
        assert r.shape == (batch_size, 3), \
            f"Rotation shape mismatch: {r.shape} != ({batch_size}, 3)"

    print("  [OK] Full pipeline test passed")

    if torch.cuda.is_available():
        mem_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"  GPU Memory (peak): {mem_used:.2f} GB")

    return True


def test_gradient_flow(model, batch, device, batch_size):
    """Test that gradients flow properly through the model."""
    print("\n--- Test 8: Gradient Flow ---")
    model.train()

    monoleft = torch.from_numpy(batch[6]).float().to(device)
    labels = torch.from_numpy(batch[10]).float()
    transgt = labels[:, 3:6].to(device)
    rotgt = labels[:, 0:3].to(device)

    trans, rot = model([monoleft])
    loss = nn.MSELoss()(trans, transgt) + nn.MSELoss()(rot, rotgt)
    loss.backward()

    # Check that gradients exist and are not zero
    grad_ok = True
    zero_grad_count = 0
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is None:
                zero_grad_count += 1
            elif param.grad.abs().sum().item() == 0:
                zero_grad_count += 1

    model.zero_grad()
    model.eval()

    grad_ratio = (total_params - zero_grad_count) / total_params * 100
    print(f"  Parameters with gradients: {total_params - zero_grad_count}/{total_params} "
          f"({grad_ratio:.0f}%)")

    if grad_ratio < 50:
        print("  [WARN] Many parameters have zero gradients")
        return False
    else:
        print("  [OK] Gradient flow test passed")
        return True


def main():
    print("=" * 60)
    print("  DEMO INFERENCE TEST - Fusion Localization Pipeline")
    print("=" * 60)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("  WARNING: Using CPU")

    batch_size = demo_args.batch_size
    print(f"  Batch size: {batch_size}")

    results = {}

    # Test 1: Data Loading
    try:
        batch = test_data_loading()
        results['data_loading'] = batch is not None
    except Exception as e:
        print(f"  [FAIL] Data loading error: {e}")
        results['data_loading'] = False
        return results

    # Test 2: Model Creation
    try:
        model = test_model_creation(device)
        results['model_creation'] = True
    except Exception as e:
        print(f"  [FAIL] Model creation error: {e}")
        import traceback
        traceback.print_exc()
        results['model_creation'] = False
        return results

    # Test 3: Checkpoint Loading
    try:
        model = test_checkpoint_loading(model, demo_args.checkpoint, device)
        results['checkpoint_loading'] = True
    except Exception as e:
        print(f"  [FAIL] Checkpoint loading error: {e}")
        results['checkpoint_loading'] = False

    # Test 4: LiDAR Inference
    try:
        results['lidar_inference'] = test_lidar_inference(
            model, batch, device, batch_size)
    except Exception as e:
        print(f"  [FAIL] LiDAR inference error: {e}")
        import traceback
        traceback.print_exc()
        results['lidar_inference'] = False

    # Test 5: Camera Inference
    try:
        results['camera_inference'] = test_camera_inference(
            model, batch, device, batch_size)
    except Exception as e:
        print(f"  [FAIL] Camera inference error: {e}")
        import traceback
        traceback.print_exc()
        results['camera_inference'] = False

    # Test 6: Radar Inference
    try:
        results['radar_inference'] = test_radar_inference(
            model, batch, device, batch_size)
    except Exception as e:
        print(f"  [FAIL] Radar inference error: {e}")
        import traceback
        traceback.print_exc()
        results['radar_inference'] = False

    # Test 7: Full Pipeline
    try:
        results['full_pipeline'] = test_full_pipeline(
            model, batch, device, batch_size)
    except Exception as e:
        print(f"  [FAIL] Full pipeline error: {e}")
        import traceback
        traceback.print_exc()
        results['full_pipeline'] = False

    # Test 8: Gradient Flow
    try:
        results['gradient_flow'] = test_gradient_flow(
            model, batch, device, batch_size)
    except Exception as e:
        print(f"  [FAIL] Gradient flow error: {e}")
        import traceback
        traceback.print_exc()
        results['gradient_flow'] = False

    # ========== Summary ==========
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        icon = "OK" if passed else "!!"
        print(f"  [{icon}] {test_name:25s} {status}")
        if not passed:
            all_passed = False

    passed_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    print(f"\n  Result: {passed_count}/{total_count} tests passed")

    if all_passed:
        print("  Pipeline is ready for training with real data!")
    else:
        print("  Some tests failed. Check the logs above for details.")

    print(f"{'='*60}")
    return results


if __name__ == '__main__':
    main()
