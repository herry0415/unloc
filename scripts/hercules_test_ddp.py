#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HeRCULES Test/Inference Script (DDP Compatible)

Validates the model on HeRCULES test data with all 3 modalities.
Compatible with both single-GPU and multi-GPU trained models.

Tests:
1. Data loading (HerculesFusion + hercules_cylinder_dataset)
2. Model creation
3. Checkpoint loading (handles DDP module prefix)
4. LiDAR inference
5. Radar inference
6. Camera inference
7. Full multi-modal inference
8. Gradient flow

Usage (single GPU):
    python scripts/hercules_test_ddp.py --sequence Library --checkpoint checkpoints/hercules_best.pt

Usage (multi-GPU trained model on single GPU):
    python scripts/hercules_test_ddp.py --sequence Library --checkpoint checkpoints/hercules_best.pt --gpu 0
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

# Parse arguments first
test_parser = argparse.ArgumentParser(description='HeRCULES inference test (DDP compatible)')
test_parser.add_argument('--sequence', type=str, default='Library',
                         choices=['Library', 'Sports'],
                         help='Sequence to test on')
test_parser.add_argument('--data_root', type=str, default='/data/drj/HeRCULES/',
                         help='Path to HeRCULES dataset')
test_parser.add_argument('--config', type=str,
                         default=os.path.join(PROJECT_ROOT, 'config', 'hercules_fusion.yaml'),
                         help='Path to config file')
test_parser.add_argument('--checkpoint', type=str, default=None,
                         help='Path to model checkpoint (supports DDP-trained models)')
test_parser.add_argument('--batch_size', type=int, default=4,
                         help='Batch size for inference')
test_parser.add_argument('--gpu', type=int, default=0,
                         help='GPU ID to use for testing')
test_args = test_parser.parse_args()

# Now override sys.argv for Combinedmodel.py
sys.argv = [sys.argv[0], '-y', test_args.config]

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

print("=" * 80)
print("  HERCULES MULTI-MODAL FUSION TEST (DDP Compatible)")
print("=" * 80)
print(f"  Sequence: {test_args.sequence}")
print(f"  GPU: cuda:{test_args.gpu}")
print(f"  Batch size: {test_args.batch_size}")
if test_args.checkpoint:
    print(f"  Checkpoint: {test_args.checkpoint}")
print("=" * 80)


# ============================================================
# Device Setup
# ============================================================

def setup_device(gpu_id):
    """Setup device for testing."""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print(f"\n⚠ CUDA not available, using CPU")

    return device


# ============================================================
# DDP-Compatible Checkpoint Loading
# ============================================================

def load_checkpoint(model, checkpoint_path):
    """
    Load checkpoint compatible with both single-GPU and DDP-trained models.

    Handles the 'module.' prefix added by DDP automatically.
    """
    print(f"\n--- Loading Checkpoint ---")
    print(f"  Path: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"  ❌ ERROR: Checkpoint not found!")
        return False

    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        # Check if checkpoint was saved from DDP model (has 'module.' prefix)
        is_ddp_checkpoint = any(k.startswith('module.') for k in state_dict.keys())

        if is_ddp_checkpoint:
            # Remove 'module.' prefix for non-DDP model
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
            print(f"  ✓ Detected DDP checkpoint, removed 'module.' prefix")

        model.load_state_dict(state_dict)
        print(f"  ✓ Checkpoint loaded successfully")
        return True

    except Exception as e:
        print(f"  ❌ ERROR loading checkpoint: {e}")
        return False


# ============================================================
# Test Functions
# ============================================================

def test_data_loading():
    """Test 1: Load data and verify structure."""
    print("\n[Test 1/8] Data Loading...")
    try:
        from data.hercules_fusion import HerculesFusion
        from dataloader.hercules_dataset import hercules_cylinder_dataset, collate_fn_BEV

        val_dataset = HerculesFusion(
            data_root=test_args.data_root,
            sequence_name=test_args.sequence,
            split='val'
        )

        val_cyl_dataset = hercules_cylinder_dataset(
            val_dataset,
            grid_size=[480, 360, 32],
            fixed_volume_space=False
        )

        val_loader = DataLoader(
            val_cyl_dataset,
            batch_size=test_args.batch_size,
            collate_fn=collate_fn_BEV,
            num_workers=0,
            shuffle=False
        )

        # Get one batch
        for data in val_loader:
            assert len(data) == 11, f"Expected 11 elements, got {len(data)}"
            assert data[10].shape[0] == test_args.batch_size, "Batch size mismatch"
            print(f"  ✓ Loaded batch with shape: batch_size={data[10].shape[0]}")
            print(f"    - LiDAR voxel positions: {data[0].shape}")
            print(f"    - Camera image: {data[6].shape}")
            print(f"    - Labels: {data[10].shape}")
            break

        print("  ✓ Data loading test PASSED")
        return True, val_loader

    except Exception as e:
        print(f"  ❌ Data loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_creation(device):
    """Test 2: Create model."""
    print("\n[Test 2/8] Model Creation...")
    try:
        from FusionModel import Fusionmodel

        model = Fusionmodel()
        model.to(device)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created with {total_params:,} parameters")
        print("  ✓ Model creation test PASSED")
        return True, model

    except Exception as e:
        print(f"  ❌ Model creation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_checkpoint_loading(model, device):
    """Test 3: Load checkpoint."""
    print("\n[Test 3/8] Checkpoint Loading...")

    if not test_args.checkpoint:
        print("  ⊘ Skipped (no checkpoint provided)")
        return True

    try:
        model.to(device)
        success = load_checkpoint(model, test_args.checkpoint)
        if success:
            print("  ✓ Checkpoint loading test PASSED")
            return True
        else:
            print("  ❌ Checkpoint loading test FAILED")
            return False
    except Exception as e:
        print(f"  ❌ Checkpoint loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lidar_inference(model, val_loader, device):
    """Test 4: LiDAR inference."""
    print("\n[Test 4/8] LiDAR Inference...")
    try:
        with torch.no_grad():
            for data in val_loader:
                actual_bs = data[10].shape[0]
                vox_tenl = [torch.from_numpy(i).to(device) for i in data[1]]
                pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in data[2]]

                trans, rot = model([pt_fea_tenl, vox_tenl, actual_bs])

                assert trans.shape == (actual_bs, 3), f"Translation shape mismatch: {trans.shape}"
                assert rot.shape == (actual_bs, 3), f"Rotation shape mismatch: {rot.shape}"
                assert not torch.isnan(trans).any(), "NaN in translation output"
                assert not torch.isnan(rot).any(), "NaN in rotation output"

                print(f"  ✓ Output shapes: trans={trans.shape}, rot={rot.shape}")
                print(f"  ✓ No NaN values detected")
                print("  ✓ LiDAR inference test PASSED")
                return True

    except Exception as e:
        print(f"  ❌ LiDAR inference test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_camera_inference(model, val_loader, device):
    """Test 5: Camera inference."""
    print("\n[Test 5/8] Camera Inference...")
    try:
        with torch.no_grad():
            for data in val_loader:
                actual_bs = data[10].shape[0]
                monoleft = torch.from_numpy(data[6]).float().to(device)

                assert monoleft.shape[0] == actual_bs, "Batch size mismatch"
                assert monoleft.shape[1] == 3, f"Expected 3 channels, got {monoleft.shape[1]}"

                trans, rot = model([monoleft])

                assert trans.shape == (actual_bs, 3), f"Translation shape mismatch: {trans.shape}"
                assert rot.shape == (actual_bs, 3), f"Rotation shape mismatch: {rot.shape}"
                assert not torch.isnan(trans).any(), "NaN in translation output"
                assert not torch.isnan(rot).any(), "NaN in rotation output"

                print(f"  ✓ Camera input shape: {monoleft.shape}")
                print(f"  ✓ Output shapes: trans={trans.shape}, rot={rot.shape}")
                print("  ✓ Camera inference test PASSED")
                return True

    except Exception as e:
        print(f"  ❌ Camera inference test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_radar_inference(model, val_loader, device):
    """Test 6: Radar inference."""
    print("\n[Test 6/8] Radar Inference...")
    try:
        with torch.no_grad():
            for data in val_loader:
                actual_bs = data[10].shape[0]
                vox_tenr = [torch.from_numpy(i).to(device) for i in data[4]]
                pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in data[5]]

                trans, rot = model([pt_fea_tenr, vox_tenr, actual_bs])

                assert trans.shape == (actual_bs, 3), f"Translation shape mismatch: {trans.shape}"
                assert rot.shape == (actual_bs, 3), f"Rotation shape mismatch: {rot.shape}"
                assert not torch.isnan(trans).any(), "NaN in translation output"
                assert not torch.isnan(rot).any(), "NaN in rotation output"

                print(f"  ✓ Output shapes: trans={trans.shape}, rot={rot.shape}")
                print("  ✓ Radar inference test PASSED")
                return True

    except Exception as e:
        print(f"  ❌ Radar inference test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_modal_fusion(model, val_loader, device):
    """Test 7: Full multi-modal fusion."""
    print("\n[Test 7/8] Multi-Modal Fusion...")
    try:
        with torch.no_grad():
            for data in val_loader:
                actual_bs = data[10].shape[0]

                # LiDAR
                vox_tenl = [torch.from_numpy(i).to(device) for i in data[1]]
                pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in data[2]]
                trans_l, rot_l = model([pt_fea_tenl, vox_tenl, actual_bs])

                # Radar
                vox_tenr = [torch.from_numpy(i).to(device) for i in data[4]]
                pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in data[5]]
                trans_r, rot_r = model([pt_fea_tenr, vox_tenr, actual_bs])

                # Camera
                monoleft = torch.from_numpy(data[6]).float().to(device)
                trans_c, rot_c = model([monoleft])

                print(f"  ✓ LiDAR: trans={trans_l.shape}, rot={rot_l.shape}")
                print(f"  ✓ Radar: trans={trans_r.shape}, rot={rot_r.shape}")
                print(f"  ✓ Camera: trans={trans_c.shape}, rot={rot_c.shape}")
                print("  ✓ Multi-modal fusion test PASSED")
                return True

    except Exception as e:
        print(f"  ❌ Multi-modal fusion test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow(model, val_loader, device):
    """Test 8: Gradient flow."""
    print("\n[Test 8/8] Gradient Flow...")
    try:
        model.train()
        criterion = nn.MSELoss()

        for data in val_loader:
            actual_bs = data[10].shape[0]

            vox_tenl = [torch.from_numpy(i).to(device) for i in data[1]]
            pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in data[2]]
            trans, rot = model([pt_fea_tenl, vox_tenl, actual_bs])

            labels = torch.from_numpy(data[10]).float().to(device)
            rotgt = labels[:, 0:3]
            transgt = labels[:, 3:6]

            loss = criterion(trans, transgt) + criterion(rot, rotgt)
            loss.backward()

            # Check for gradients
            has_grads = False
            for p in model.parameters():
                if p.grad is not None and p.grad.abs().sum() > 0:
                    has_grads = True
                    break

            if has_grads:
                print(f"  ✓ Gradients computed successfully")
                print("  ✓ Gradient flow test PASSED")
                return True
            else:
                print(f"  ❌ No gradients computed")
                return False

    except Exception as e:
        print(f"  ❌ Gradient flow test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# Main Test Suite
# ============================================================

def main():
    device = setup_device(test_args.gpu)

    # Test 1: Data loading
    success1, val_loader = test_data_loading()
    if not success1:
        return 1

    # Test 2: Model creation
    success2, model = test_model_creation(device)
    if not success2:
        return 1

    # Test 3: Checkpoint loading
    success3 = test_checkpoint_loading(model, device)
    model = model.to(device)

    # Test 4-7: Inference tests
    success4 = test_lidar_inference(model, val_loader, device)
    success5 = test_camera_inference(model, val_loader, device)
    success6 = test_radar_inference(model, val_loader, device)
    success7 = test_multi_modal_fusion(model, val_loader, device)

    # Test 8: Gradient flow
    success8 = test_gradient_flow(model, val_loader, device)

    # Summary
    print("\n" + "=" * 80)
    print("  TEST SUMMARY")
    print("=" * 80)
    print(f"  [1/8] Data Loading:              {'✓ PASS' if success1 else '❌ FAIL'}")
    print(f"  [2/8] Model Creation:            {'✓ PASS' if success2 else '❌ FAIL'}")
    print(f"  [3/8] Checkpoint Loading:        {'✓ PASS' if success3 else '❌ FAIL'}")
    print(f"  [4/8] LiDAR Inference:           {'✓ PASS' if success4 else '❌ FAIL'}")
    print(f"  [5/8] Camera Inference:          {'✓ PASS' if success5 else '❌ FAIL'}")
    print(f"  [6/8] Radar Inference:           {'✓ PASS' if success6 else '❌ FAIL'}")
    print(f"  [7/8] Multi-Modal Fusion:        {'✓ PASS' if success7 else '❌ FAIL'}")
    print(f"  [8/8] Gradient Flow:             {'✓ PASS' if success8 else '❌ FAIL'}")
    print("=" * 80)

    all_passed = all([success1, success2, success3, success4, success5, success6, success7, success8])
    if all_passed:
        print("  🎉 ALL TESTS PASSED!")
        print("=" * 80)
        return 0
    else:
        print("  ❌ SOME TESTS FAILED")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    exit(main())
