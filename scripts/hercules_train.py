#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HeRCULES Multi-Modal Fusion Training Script

Trains FusionModel on HeRCULES dataset with 3 forward passes:
1. LiDAR point cloud
2. Radar point cloud (transformed to LiDAR frame)
3. Camera (stereo_left only)

The 3 losses are summed: Loss = loss_lidar + loss_radar + loss_camera

Usage:
    cd <project_root>
    python scripts/hercules_train.py --sequence Library --epochs 50 --batch_size 4

Based on demo_train.py and TrainModel.py patterns.
"""

import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime

# ============================================================
# CRITICAL: Setup paths BEFORE any project imports
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Parse arguments FIRST (before sys.argv override)
parser = argparse.ArgumentParser(description='HeRCULES training')
parser.add_argument('--sequence', type=str, default='Library',
                    choices=['Library', 'Sports'],
                    help='HeRCULES sequence to train on')
parser.add_argument('--data_root', type=str, default='/data/drj/HeRCULES/',
                    help='Path to HeRCULES dataset root')
parser.add_argument('--config', type=str,
                    default=os.path.join(PROJECT_ROOT, 'config', 'hercules_fusion.yaml'),
                    help='Path to config file')
parser.add_argument('--batch_size', type=int, default=2,
                    help='Batch size for training')
parser.add_argument('--val_batch_size', type=int, default=2,
                    help='Batch size for validation')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to checkpoint to load')
parser.add_argument('--output_dir', type=str, default='checkpoints',
                    help='Directory for checkpoints')
args = parser.parse_args()

# Now override sys.argv for Combinedmodel.py
sys.argv = [sys.argv[0], '-y', args.config]

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# ============================================================
# Loss Function
# ============================================================
class AtLocCriterion(nn.Module):
    """6DoF pose regression loss with learnable uncertainty weighting."""

    def __init__(self, t_loss_fn=nn.MSELoss(), q_loss_fn=nn.MSELoss(),
                 sax=0.0, saq=0.0, learn_beta=False):
        super(AtLocCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, trans, rot, rotgt, transgt):
        """
        Args:
            trans: predicted translation (B, 3)
            rot: predicted rotation/log_quat (B, 3)
            rotgt: ground truth rotation (B, 3)
            transgt: ground truth translation (B, 3)
        """
        loss = (torch.exp(-self.sax) * self.t_loss_fn(trans, transgt) + self.sax +
                torch.exp(-self.saq) * self.q_loss_fn(rot, rotgt) + self.saq)
        return loss


# ============================================================
# Environment Setup
# ============================================================

def check_environment():
    """Pre-flight checks."""
    print("=" * 70)
    print("  HERCULES MULTI-MODAL FUSION TRAINING")
    print("=" * 70)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory: {gpu_mem:.1f} GB")
    else:
        device = torch.device('cpu')
        print("  WARNING: CUDA not available, using CPU")

    print(f"  PyTorch: {torch.__version__}")
    print(f"  Config: {args.config}")
    print(f"  Sequence: {args.sequence}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print("=" * 70)

    if not os.path.exists(args.config):
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    return device


# ============================================================
# Data Loading
# ============================================================

def build_dataloaders():
    """Build HeRCULES data loaders."""
    from data.hercules_fusion import HerculesFusion
    from dataloader.hercules_dataset import hercules_cylinder_dataset, collate_fn_BEV
    from torch.utils.data import DataLoader

    print("\n--- Building Data Loaders ---")

    # Build training dataset
    train_pc_dataset = HerculesFusion(
        data_root=args.data_root,
        sequence_name=args.sequence,
        split='train'
    )
    train_cyl_dataset = hercules_cylinder_dataset(
        train_pc_dataset,
        grid_size=[480, 360, 32],
        fixed_volume_space=False
    )
    train_loader = DataLoader(
        train_cyl_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn_BEV,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    # Build validation dataset
    val_pc_dataset = HerculesFusion(
        data_root=args.data_root,
        sequence_name=args.sequence,
        split='val'
    )
    val_cyl_dataset = hercules_cylinder_dataset(
        val_pc_dataset,
        grid_size=[480, 360, 32],
        fixed_volume_space=False
    )
    val_loader = DataLoader(
        val_cyl_dataset,
        batch_size=args.val_batch_size,
        collate_fn=collate_fn_BEV,
        shuffle=False,
        num_workers=4
    )

    print(f"  Train samples: {len(train_cyl_dataset)}")
    print(f"  Val samples: {len(val_cyl_dataset)}")
    print(f"  Train batches per epoch: {len(train_loader)}")

    return train_loader, val_loader


# ============================================================
# Model Building
# ============================================================

def build_model(device):
    """Build Fusion model."""
    from FusionModel import Fusionmodel

    print("\n--- Building Fusion Model ---")
    model = Fusionmodel()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer,
                    device, epoch, num_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    lidar_loss_sum = 0.0
    radar_loss_sum = 0.0
    camera_loss_sum = 0.0
    num_batches = 0

    for i_iter, data in enumerate(train_loader):
        # ========== Extract data from 11-tuple ==========
        actual_bs = data[10].shape[0]

        # LiDAR data
        train_vox_tenl = [torch.from_numpy(i).to(device) for i in data[1]]
        train_pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(device)
                             for i in data[2]]

        # Radar data
        train_vox_tenr = [torch.from_numpy(i).to(device) for i in data[4]]
        train_pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(device)
                             for i in data[5]]

        # Camera data
        monoleft = torch.from_numpy(data[6]).float().to(device)

        # Labels (6DoF: rot(3) + trans(3))
        labels = torch.from_numpy(data[10]).float().to(device)
        rotgt = labels[:, 0:3]
        transgt = labels[:, 3:6]

        optimizer.zero_grad()

        # ========== Forward Pass 1: LiDAR ==========
        trans_lidar, rot_lidar = model([train_pt_fea_tenl, train_vox_tenl, actual_bs])
        loss_lidar = criterion(trans_lidar, rot_lidar, rotgt, transgt)

        # ========== Forward Pass 2: Radar ==========
        trans_radar, rot_radar = model([train_pt_fea_tenr, train_vox_tenr, actual_bs])
        loss_radar = criterion(trans_radar, rot_radar, rotgt, transgt)

        # ========== Forward Pass 3: Camera ==========
        trans_camera, rot_camera = model([monoleft])
        loss_camera = criterion(trans_camera, rot_camera, rotgt, transgt)

        # ========== Backward ==========
        loss = loss_lidar + loss_radar + loss_camera
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate statistics
        total_loss += loss.item()
        lidar_loss_sum += loss_lidar.item()
        radar_loss_sum += loss_radar.item()
        camera_loss_sum += loss_camera.item()
        num_batches += 1

        # Print progress
        if (i_iter + 1) % max(1, len(train_loader) // 10) == 0:
            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch + 1}/{num_epochs} | Batch {i_iter + 1}/{len(train_loader)} | "
                  f"Loss: {avg_loss:.4f} "
                  f"(L:{lidar_loss_sum/num_batches:.4f} "
                  f"R:{radar_loss_sum/num_batches:.4f} "
                  f"C:{camera_loss_sum/num_batches:.4f})")

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data in val_loader:
            actual_bs = data[10].shape[0]

            # Extract data
            vox_tenl = [torch.from_numpy(i).to(device) for i in data[1]]
            pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(device)
                          for i in data[2]]

            vox_tenr = [torch.from_numpy(i).to(device) for i in data[4]]
            pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(device)
                          for i in data[5]]

            monoleft = torch.from_numpy(data[6]).float().to(device)

            labels = torch.from_numpy(data[10]).float().to(device)
            rotgt = labels[:, 0:3]
            transgt = labels[:, 3:6]

            # Forward passes
            trans_l, rot_l = model([pt_fea_tenl, vox_tenl, actual_bs])
            loss_l = criterion(trans_l, rot_l, rotgt, transgt)

            trans_r, rot_r = model([pt_fea_tenr, vox_tenr, actual_bs])
            loss_r = criterion(trans_r, rot_r, rotgt, transgt)

            trans_c, rot_c = model([monoleft])
            loss_c = criterion(trans_c, rot_c, rotgt, transgt)

            loss = loss_l + loss_r + loss_c
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


# ============================================================
# Main Training Loop
# ============================================================

def main():
    device = check_environment()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build data loaders
    train_loader, val_loader = build_dataloaders()

    # Build model
    model = build_model(device)

    # Setup loss and optimizer
    print("\n--- Setting up Loss and Optimizer ---")
    criterion = AtLocCriterion(
        t_loss_fn=nn.MSELoss(),
        q_loss_fn=nn.MSELoss(),
        sax=0.0,
        saq=0.0,
        learn_beta=True
    )
    criterion.to(device)

    optimizer = Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr
    )
    print(f"  Loss function: AtLocCriterion (with learnable uncertainty)")
    print(f"  Optimizer: Adam (lr={args.lr})")

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\n--- Loading Checkpoint ---")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  Loaded: {args.checkpoint}")

    # Training loop
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                      device, epoch, args.epochs)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {elapsed:.1f}s")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            checkpoint_path = os.path.join(args.output_dir, 'hercules_best.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Best model saved: {checkpoint_path}")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            periodic_path = os.path.join(args.output_dir, f'hercules_epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), periodic_path)
            print(f"  ✓ Periodic checkpoint: {periodic_path}")

    print("\n" + "=" * 70)
    print(f"  Training Complete!")
    print(f"  Best model at epoch {best_epoch} with val loss {best_val_loss:.6f}")
    print(f"  Checkpoint: {os.path.join(args.output_dir, 'hercules_best.pt')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
