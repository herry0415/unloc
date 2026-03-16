#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HeRCULES Single-Modality Training Script (DDP Multi-GPU Version)

Trains FusionModel on HeRCULES dataset with a SINGLE modality (LiDAR or Radar).
Only 1 forward + backward pass per iteration (vs 3 in hercules_train_ddp.py).

Benefits:
- Compare single-modality vs multi-modal performance
- 1/3 training time and GPU memory
- Evaluate individual sensor localization ability

Usage (LiDAR, single GPU 1):
    python scripts/hercules_train_single.py --modality lidar --gpu 1 --num_gpus 1 --batch_size 4 --epochs 40 \
        --output_dir checkpoints/lidar_only

Usage (Radar, single GPU 1):
    python scripts/hercules_train_single.py --modality radar --gpu 1 --num_gpus 1 --batch_size 6 --epochs 40 \
        --output_dir checkpoints/radar_only

Resume from checkpoint:
    python scripts/hercules_train_single.py --modality lidar --gpu 1 --num_gpus 1 --batch_size 4 --epochs 40 \
        --checkpoint checkpoints/lidar_only/hercules_best_lidar.pt \
        --output_dir checkpoints/lidar_only
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
parser = argparse.ArgumentParser(description='HeRCULES single-modality DDP training')
parser.add_argument('--modality', type=str, default='lidar',
                    choices=['lidar', 'radar'],
                    help='Which modality to train')
parser.add_argument('--sequence', type=str, default='Library',
                    choices=['Library', 'Sports'],
                    help='HeRCULES sequence to train on')
parser.add_argument('--data_root', type=str, default='/data/drj/HeRCULES/',
                    help='Path to HeRCULES dataset root')
parser.add_argument('--config', type=str,
                    default=os.path.join(PROJECT_ROOT, 'config', 'hercules_fusion.yaml'),
                    help='Path to config file')
parser.add_argument('--batch_size', type=int, default=2,
                    help='Batch size per GPU (total batch = batch_size * num_gpus)')
parser.add_argument('--val_batch_size', type=int, default=2,
                    help='Batch size for validation per GPU')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning rate')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to checkpoint to load')
parser.add_argument('--output_dir', type=str, default='checkpoints',
                    help='Directory for checkpoints')
# GPU parameters
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU ID to use for single-GPU training')
parser.add_argument('--num_gpus', type=int, default=1, choices=[1, 2, 3, 4],
                    help='Number of GPUs to use for distributed training')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of data loading workers per GPU')
# Required for torch.distributed.launch compatibility
parser.add_argument('--local_rank', type=int, default=0,
                    help='Local rank for distributed training (auto-set by launcher)')

args = parser.parse_args()

# Now override sys.argv for Combinedmodel.py
sys.argv = [sys.argv[0], '-y', args.config]

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.distributed as dist
try:
    from torch.nn import DistributedDataParallel as DDP
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

# ============================================================
# Distributed Training Utilities
# ============================================================

def setup_distributed(num_gpus):
    """Initialize distributed training."""
    if num_gpus == 1:
        # Single GPU mode - use --gpu to select device
        local_rank = args.gpu
        return False, local_rank, 1

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return True, rank, world_size


def cleanup_distributed(is_distributed):
    """Clean up distributed training."""
    if is_distributed:
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if current process is main process (rank 0)."""
    return rank == 0


def log_print(msg, rank=0):
    """Print only on rank 0."""
    if is_main_process(rank):
        print(msg)


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
        loss = (torch.exp(-self.sax) * self.t_loss_fn(trans, transgt) + self.sax +
                torch.exp(-self.saq) * self.q_loss_fn(rot, rotgt) + self.saq)
        return loss


# ============================================================
# Environment Setup
# ============================================================

def check_environment(rank, is_distributed):
    """Pre-flight checks before training."""
    if is_main_process(rank):
        log_print("=" * 70)
        log_print(f"  HERCULES SINGLE-MODALITY TRAINING [{args.modality.upper()}] (DDP)")
        log_print("=" * 70)

    if torch.cuda.is_available():
        if is_distributed:
            local_rank = args.local_rank if hasattr(args, 'local_rank') else rank
        else:
            local_rank = args.gpu
        device = torch.device(f'cuda:{local_rank}')

        if is_main_process(rank):
            log_print(f"  GPU: {torch.cuda.get_device_name(device)}")
            log_print(f"  CUDA Version: {torch.version.cuda}")
            gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
            log_print(f"  GPU Memory: {gpu_mem:.1f} GB per GPU")
    else:
        device = torch.device('cpu')
        if is_main_process(rank):
            log_print("  WARNING: CUDA not available, using CPU")

    if is_main_process(rank):
        log_print(f"  PyTorch: {torch.__version__}")
        log_print(f"  Config: {args.config}")
        log_print(f"  Modality: {args.modality.upper()}")
        log_print(f"  Sequence: {args.sequence}")
        log_print(f"  Batch size per GPU: {args.batch_size}")
        log_print(f"  Total batch size: {args.batch_size * (1 if not is_distributed else dist.get_world_size())}")
        log_print(f"  Learning rate: {args.lr}")
        log_print(f"  Epochs: {args.epochs}")
        log_print(f"  Num GPUs: {args.num_gpus}")
        if is_distributed:
            log_print(f"  Distributed mode: DDP with {dist.get_world_size()} processes")
        log_print("=" * 70)

    if not os.path.exists(args.config):
        log_print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)

    return device


# ============================================================
# Data Loading
# ============================================================

def build_dataloaders(rank, world_size, is_distributed):
    """Build HeRCULES data loaders with DDP support."""
    from data.hercules_fusion import HerculesFusion
    from dataloader.hercules_dataset import hercules_cylinder_dataset, collate_fn_BEV
    from torch.utils.data import DataLoader, DistributedSampler

    if is_main_process(rank):
        log_print("\n--- Building Data Loaders (DDP) ---")

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

    if is_distributed:
        train_sampler = DistributedSampler(
            train_cyl_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        train_loader = DataLoader(
            train_cyl_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn_BEV,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_cyl_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_BEV,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True
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

    if is_distributed:
        val_sampler = DistributedSampler(
            val_cyl_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        val_loader = DataLoader(
            val_cyl_dataset,
            batch_size=args.val_batch_size,
            sampler=val_sampler,
            collate_fn=collate_fn_BEV,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        val_loader = DataLoader(
            val_cyl_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            collate_fn=collate_fn_BEV,
            num_workers=args.num_workers,
            pin_memory=True
        )

    if is_main_process(rank):
        log_print(f"  Train samples: {len(train_cyl_dataset)}")
        log_print(f"  Val samples: {len(val_cyl_dataset)}")
        if is_distributed:
            log_print(f"  Train batches per epoch (per GPU): {len(train_loader)}")
        else:
            log_print(f"  Train batches per epoch: {len(train_loader)}")

    return train_loader, val_loader


# ============================================================
# Model Building
# ============================================================

def build_model(device, rank, is_distributed):
    """Build Fusion model with DDP wrapper."""
    from FusionModel import Fusionmodel

    if is_main_process(rank):
        log_print("\n--- Building Fusion Model ---")

    model = Fusionmodel()
    model.to(device)

    if is_distributed:
        local_rank = args.local_rank if hasattr(args, 'local_rank') else rank
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    if is_main_process(rank):
        if is_distributed:
            total_params = sum(p.numel() for p in model.module.parameters())
            trainable_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
        else:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        log_print(f"  Total parameters: {total_params:,}")
        log_print(f"  Trainable parameters: {trainable_params:,}")
        if is_distributed:
            local_rank = args.local_rank if hasattr(args, 'local_rank') else rank
            log_print(f"  DDP: Enabled (device: cuda:{local_rank})")

    return model


# ============================================================
# Training Loop
# ============================================================

def train_one_epoch(model, train_loader, criterion, optimizer,
                    device, epoch, num_epochs, rank, is_distributed):
    """Train for one epoch with single modality."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    if is_distributed:
        train_loader.sampler.set_epoch(epoch)

    # Create progress bar (only on rank 0)
    if is_main_process(rank):
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'Epoch {epoch + 1}/{num_epochs} [{args.modality.upper()}]',
            ncols=120,
            ascii=True,
            leave=True
        )
        iterator = pbar
    else:
        pbar = None
        iterator = enumerate(train_loader)

    for i_iter, data in iterator:
        # ========== Extract data from 11-tuple ==========
        actual_bs = data[10].shape[0]

        # Labels: process_poses outputs [trans_normalized(3), log_quat(3)]
        labels = torch.from_numpy(data[10]).float().to(device)
        transgt = labels[:, 0:3]  # normalized translation
        rotgt = labels[:, 3:6]    # log quaternion

        optimizer.zero_grad()

        # ========== Single forward + backward ==========
        if args.modality == 'lidar':
            train_vox_ten = [torch.from_numpy(i).to(device) for i in data[1]]
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(device)
                                for i in data[2]]
            trans, rot = model([train_pt_fea_ten, train_vox_ten, actual_bs])
        elif args.modality == 'radar':
            train_vox_ten = [torch.from_numpy(i).to(device) for i in data[4]]
            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(device)
                                for i in data[5]]
            trans, rot = model([train_pt_fea_ten, train_vox_ten, actual_bs])

        loss = criterion(trans, rot, rotgt, transgt)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate statistics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar (only on rank 0)
        if is_main_process(rank) and pbar is not None:
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'Loss': f'{avg_loss:.6f}'})

    if pbar is not None:
        pbar.close()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, val_loader, criterion, device, rank, is_distributed):
    """Validate the model with single modality."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Create progress bar (only on rank 0)
    if is_main_process(rank):
        pbar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc=f'Validating [{args.modality.upper()}]',
            ncols=120,
            ascii=True,
            leave=True
        )
        iterator = pbar
    else:
        pbar = None
        iterator = enumerate(val_loader)

    with torch.no_grad():
        for i_iter, data in iterator:
            actual_bs = data[10].shape[0]

            # Labels
            labels = torch.from_numpy(data[10]).float().to(device)
            transgt = labels[:, 0:3]
            rotgt = labels[:, 3:6]

            # Single forward pass
            if args.modality == 'lidar':
                vox_ten = [torch.from_numpy(i).to(device) for i in data[1]]
                pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(device)
                              for i in data[2]]
                trans, rot = model([pt_fea_ten, vox_ten, actual_bs])
            elif args.modality == 'radar':
                vox_ten = [torch.from_numpy(i).to(device) for i in data[4]]
                pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(device)
                              for i in data[5]]
                trans, rot = model([pt_fea_ten, vox_ten, actual_bs])

            loss = criterion(trans, rot, rotgt, transgt)
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            if is_main_process(rank) and pbar is not None:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'Loss': f'{avg_loss:.6f}'})

    if pbar is not None:
        pbar.close()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Synchronize validation loss across all processes
    if is_distributed:
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor / dist.get_world_size()).item()

    return avg_loss


# ============================================================
# Main Training Loop
# ============================================================

def main():
    # Initialize distributed training
    is_distributed, rank, world_size = setup_distributed(args.num_gpus)

    # Setup device
    device = check_environment(rank, is_distributed)

    # Create output directory (only on rank 0)
    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)

    # Build data loaders
    train_loader, val_loader = build_dataloaders(rank, world_size, is_distributed)

    # Build model
    model = build_model(device, rank, is_distributed)

    # Setup loss and optimizer
    if is_main_process(rank):
        log_print("\n--- Setting up Loss and Optimizer ---")

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

    if is_main_process(rank):
        log_print(f"  Loss function: AtLocCriterion (with learnable uncertainty)")
        log_print(f"  Optimizer: Adam (lr={args.lr})")

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        if is_main_process(rank):
            log_print(f"\n--- Loading Checkpoint ---")
        state_dict = torch.load(args.checkpoint, map_location=device)
        if is_distributed:
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        if is_main_process(rank):
            log_print(f"  Loaded: {args.checkpoint}")

    # Training loop
    if is_main_process(rank):
        log_print("\n" + "=" * 100)
        log_print(f"  STARTING TRAINING LOOP [{args.modality.upper()} ONLY]")
        log_print("=" * 100)

    best_val_loss = float('inf')
    best_epoch = 0
    epoch_times = []

    for epoch in range(args.epochs):
        if is_main_process(rank):
            log_print(f"\n{'='*100}")
            log_print(f"  Epoch {epoch + 1}/{args.epochs} [{args.modality.upper()}]")
            log_print(f"{'='*100}")

        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                      device, epoch, args.epochs, rank, is_distributed)

        # Validate
        val_loss = validate(model, val_loader, criterion, device, rank, is_distributed)

        elapsed = time.time() - start_time
        epoch_times.append(elapsed)
        epoch_time_min = elapsed / 60.0

        # Calculate and show estimated remaining time
        if is_main_process(rank):
            if len(epoch_times) > 0:
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                remaining_epochs = args.epochs - (epoch + 1)
                estimated_remaining_hours = (avg_epoch_time * remaining_epochs) / 3600.0
            else:
                estimated_remaining_hours = 0

            log_print(f"\n{'─'*100}")
            log_print(f"  Epoch {epoch + 1}/{args.epochs} [{args.modality.upper()}] Complete")
            log_print(f"{'─'*100}")
            log_print(f"  Train Loss: {train_loss:.6f}  |  Val Loss: {val_loss:.6f}")
            log_print(f"  Epoch Time: {epoch_time_min:.2f} min  |  Est. Remaining: {estimated_remaining_hours:.1f} hours")
            log_print(f"{'─'*100}\n")

            # Save best checkpoint (modality-specific naming)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                checkpoint_path = os.path.join(args.output_dir, f'hercules_best_{args.modality}.pt')
                if is_distributed:
                    torch.save(model.module.state_dict(), checkpoint_path)
                else:
                    torch.save(model.state_dict(), checkpoint_path)
                log_print(f"  New best model! Val Loss: {val_loss:.6f}\n")

            # Periodic checkpoint (every 5 epochs, modality-specific naming)
            if (epoch + 1) % 5 == 0:
                periodic_path = os.path.join(args.output_dir, f'hercules_epoch_{epoch + 1}_{args.modality}.pt')
                if is_distributed:
                    torch.save(model.module.state_dict(), periodic_path)
                else:
                    torch.save(model.state_dict(), periodic_path)
                log_print(f"  Periodic checkpoint: hercules_epoch_{epoch + 1}_{args.modality}.pt\n")

    if is_main_process(rank):
        log_print("\n" + "=" * 100)
        log_print(f"  TRAINING COMPLETE [{args.modality.upper()}]!")
        log_print(f"  Best model at epoch {best_epoch} with val loss {best_val_loss:.6f}")
        log_print(f"  Total training time: {sum(epoch_times)/3600.0:.1f} hours")
        log_print(f"  Checkpoint: {os.path.join(args.output_dir, f'hercules_best_{args.modality}.pt')}")
        log_print("=" * 100)

    # Cleanup distributed training
    cleanup_distributed(is_distributed)


if __name__ == '__main__':
    main()
