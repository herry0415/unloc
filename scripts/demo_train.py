#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Demo Training Script - Tests the complete multi-modal fusion training pipeline
using synthetic data. No real RobotCar data required.

Usage:
    cd <project_root>
    python scripts/demo_train.py

Based on TrainModel.py but simplified for quick pipeline validation.
"""

import os
import sys
import time
import numpy as np

# ============================================================
# CRITICAL: Setup paths BEFORE any project imports
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Combinedmodel.py uses argparse.parse_args() in localizationmodel.__init__()
# We must set sys.argv to provide the config path before the model is constructed
DEMO_CONFIG = os.path.join(PROJECT_ROOT, 'config', 'demo_config.yaml')
sys.argv = [sys.argv[0], '-y', DEMO_CONFIG]

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn

# ============================================================
# Loss function (same as TrainModel.py)
# ============================================================
class AtLocCriterion(nn.Module):
    """Loss for 6DoF pose regression with learnable uncertainty weights."""
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


def check_environment():
    """Pre-flight checks before training."""
    print("=" * 60)
    print("  DEMO TRAINING - Fusion Localization Pipeline")
    print("=" * 60)

    # Check CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory: {gpu_mem:.1f} GB")
    else:
        device = torch.device('cpu')
        print("  WARNING: CUDA not available, using CPU")
        print("  Note: Model has .cuda() calls that may fail without GPU")

    print(f"  PyTorch: {torch.__version__}")
    print(f"  Config: {DEMO_CONFIG}")
    print(f"  Project Root: {PROJECT_ROOT}")

    # Check config file exists
    if not os.path.exists(DEMO_CONFIG):
        print(f"\n  ERROR: Config file not found: {DEMO_CONFIG}")
        print("  Please ensure config/demo_config.yaml exists")
        sys.exit(1)

    # Check spconv
    try:
        import spconv
        print(f"  spconv: available")
    except ImportError:
        print("  WARNING: spconv not found - LiDAR model may fail")

    print("=" * 60)
    return device


def build_data_loaders(train_batch_size, val_batch_size):
    """Build demo data loaders with synthetic data."""
    from builder import data_builder

    # These configs are only needed for the data_builder.build() interface
    # In demo mode, the actual values don't matter much
    dataset_config = {
        'dataset_type': 'cylinder_dataset',
        'pc_dataset_type': 'SemKITTI_sk',
        'ignore_label': 0,
        'return_test': False,
        'fixed_volume_space': True,
        'label_mapping': '',
        'max_volume_space': [50, 3.1415926, 2],
        'min_volume_space': [0, -3.1415926, -4],
    }
    train_dataloader_config = {
        'data_path': 'demo',
        'imageset': 'train',
        'return_ref': True,
        'batch_size': train_batch_size,
        'shuffle': True,
        'num_workers': 0,
    }
    val_dataloader_config = {
        'data_path': 'demo',
        'imageset': 'val',
        'return_ref': True,
        'batch_size': val_batch_size,
        'shuffle': False,
        'num_workers': 0,
    }

    train_loader, val_loader = data_builder.build(
        dataset_config, train_dataloader_config, val_dataloader_config,
        use_demo=True
    )
    return train_loader, val_loader


def build_model(device):
    """Build and initialize the Fusionmodel."""
    from FusionModel import Fusionmodel

    print("\n--- Building Fusion Model ---")
    model = Fusionmodel()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def train_one_epoch(model, train_loader, criterion, optimizer,
                    device, epoch, num_epochs, batch_size):
    """Run one training epoch."""
    model.train()
    train_losses = []

    for i_iter, data in enumerate(train_loader):
        # ========== Extract data ==========
        train_vox_tenl = [torch.from_numpy(i).to(device) for i in data[1]]
        train_pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in data[2]]
        train_pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in data[5]]
        train_vox_tenr = [torch.from_numpy(i).to(device) for i in data[4]]
        monoleft = torch.from_numpy(data[6]).float()
        monoright = torch.from_numpy(data[7]).float()
        monorear = torch.from_numpy(data[8]).float()
        radarimage = torch.from_numpy(data[9]).float().reshape(batch_size, 1, 512, 512)
        labels = torch.from_numpy(data[10]).float()
        transgt = labels[:, 3:6]
        rotgt = labels[:, 0:3]

        optimizer.zero_grad()

        # ========== Forward passes for each modality ==========
        # Collect valid losses; skip modalities that produce NaN
        losses = []
        nan_modalities = []

        def safe_forward(name, forward_fn, *args):
            """Run forward pass and return loss, or None if NaN."""
            trans, rot = forward_fn(*args)
            loss = criterion(trans, rot, rotgt.to(device), transgt.to(device))
            if torch.isnan(loss) or torch.isinf(loss):
                nan_modalities.append(name)
                return None, trans, rot
            return loss, trans, rot

        # 1. LiDAR Left
        loss1, trans1, rot1 = safe_forward(
            "LiDAR_L", lambda: model([train_pt_fea_tenl, train_vox_tenl, batch_size]))
        if loss1 is not None:
            losses.append(loss1)

        # 2. LiDAR Right
        loss2, trans2, rot2 = safe_forward(
            "LiDAR_R", lambda: model([train_pt_fea_tenr, train_vox_tenr, batch_size]))
        if loss2 is not None:
            losses.append(loss2)

        # 3. Camera Left
        loss3, trans3, rot3 = safe_forward(
            "Cam_L", lambda: model([monoleft.to(device)]))
        if loss3 is not None:
            losses.append(loss3)

        # 4. Camera Right
        loss4, trans4, rot4 = safe_forward(
            "Cam_R", lambda: model([monoright.to(device)]))
        if loss4 is not None:
            losses.append(loss4)

        # 5. Camera Rear
        loss5, trans5, rot5 = safe_forward(
            "Cam_Re", lambda: model([monorear.to(device)]))
        if loss5 is not None:
            losses.append(loss5)

        # 6. Radar
        loss6, trans6, rot6 = safe_forward(
            "Radar", lambda: model([radarimage.to(device)]))
        if loss6 is not None:
            losses.append(loss6)

        if len(losses) == 0:
            print(f"  Batch {i_iter+1}: ALL modalities NaN, skipping")
            continue

        if nan_modalities:
            print(f"  Batch {i_iter+1}: NaN in {nan_modalities}, "
                  f"training with {len(losses)}/6 modalities")

        # ========== Backward + optimize ==========
        total_loss = sum(losses)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = total_loss.item()
        train_losses.append(loss_val)

        # ========== Logging ==========
        # Log first batch details (shapes, values)
        if i_iter == 0 and epoch == 0:
            print(f"\n  [First Batch - Data Shapes]")
            print(f"    LiDAR Left:  vox={train_vox_tenl[0].shape}, "
                  f"feat={train_pt_fea_tenl[0].shape}")
            print(f"    LiDAR Right: vox={train_vox_tenr[0].shape}, "
                  f"feat={train_pt_fea_tenr[0].shape}")
            print(f"    Camera:      {monoleft.shape}")
            print(f"    Radar:       {radarimage.shape}")
            print(f"    Labels:      {labels.shape}")
            print(f"  [First Batch - Model Output Shapes]")
            print(f"    Translation: {trans1.shape}")
            print(f"    Rotation:    {rot1.shape}")
            print()

        lidar_loss = (loss1.item() if loss1 is not None else 0) + \
                     (loss2.item() if loss2 is not None else 0)
        cam_loss = (loss3.item() if loss3 is not None else 0) + \
                   (loss4.item() if loss4 is not None else 0) + \
                   (loss5.item() if loss5 is not None else 0)
        radar_loss = loss6.item() if loss6 is not None else 0

        # Check for NaN in total
        if torch.isnan(total_loss):
            print(f"  ERROR: NaN total loss at batch {i_iter+1}!")
            continue

        lidar_loss = loss1.item() + loss2.item()
        cam_loss = loss3.item() + loss4.item() + loss5.item()
        radar_loss = loss6.item()
        print(f"  Batch {i_iter+1}/{len(train_loader)} | "
              f"Loss: {loss_val:.4f} "
              f"[LiDAR: {lidar_loss:.3f} | Cam: {cam_loss:.3f} | "
              f"Radar: {radar_loss:.3f}]")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.mean(train_losses)


def validate(model, val_loader, criterion, device, batch_size):
    """Run validation."""
    model.eval()
    val_losses = []

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_loader):
            val_vox_tenl = [torch.from_numpy(i).to(device) for i in data[1]]
            val_pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in data[2]]
            val_pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in data[5]]
            val_vox_tenr = [torch.from_numpy(i).to(device) for i in data[4]]
            monoleft = torch.from_numpy(data[6]).float()
            monoright = torch.from_numpy(data[7]).float()
            monorear = torch.from_numpy(data[8]).float()
            radarimage = torch.from_numpy(data[9]).float().reshape(batch_size, 1, 512, 512)
            labels = torch.from_numpy(data[10]).float()
            transgt = labels[:, 3:6]
            rotgt = labels[:, 0:3]

            # Forward all modalities
            trans1, rot1 = model([val_pt_fea_tenl, val_vox_tenl, batch_size])
            trans2, rot2 = model([val_pt_fea_tenr, val_vox_tenr, batch_size])
            loss1 = criterion(trans1, rot1, rotgt.to(device), transgt.to(device))
            loss2 = criterion(trans2, rot2, rotgt.to(device), transgt.to(device))

            trans3, rot3 = model([monoleft.to(device)])
            trans4, rot4 = model([monoright.to(device)])
            trans5, rot5 = model([monorear.to(device)])
            loss3 = criterion(trans3, rot3, rotgt.to(device), transgt.to(device))
            loss4 = criterion(trans4, rot4, rotgt.to(device), transgt.to(device))
            loss5 = criterion(trans5, rot5, rotgt.to(device), transgt.to(device))

            trans6, rot6 = model([radarimage.to(device)])
            loss6 = criterion(trans6, rot6, rotgt.to(device), transgt.to(device))

            val_loss_total = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
            val_losses.append(val_loss_total.item())

    return np.mean(val_losses)


def main():
    # ========== Environment check ==========
    device = check_environment()

    # ========== Hyperparameters ==========
    NUM_EPOCHS = 2
    LEARNING_RATE = 0.00001
    TRAIN_BATCH_SIZE = 2
    VAL_BATCH_SIZE = 2

    # ========== Build data loaders ==========
    print("\n--- Building Demo Data Loaders ---")
    train_loader, val_loader = build_data_loaders(TRAIN_BATCH_SIZE, VAL_BATCH_SIZE)

    # ========== Build model ==========
    model = build_model(device)

    # ========== Loss and optimizer ==========
    training_loss = AtLocCriterion(sax=0.0, saq=0.0, learn_beta=True).to(device)
    val_criterion = AtLocCriterion().to(device)

    param_list = [{'params': model.parameters()}]
    param_list.append({'params': [training_loss.sax, training_loss.saq]})
    optimizer = torch.optim.Adam(param_list, lr=LEARNING_RATE, weight_decay=0.0005)

    # ========== Checkpoint directory ==========
    save_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    # ========== Training loop ==========
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")

        # Training phase
        avg_train_loss = train_one_epoch(
            model, train_loader, training_loss, optimizer,
            device, epoch, NUM_EPOCHS, TRAIN_BATCH_SIZE
        )

        if avg_train_loss is None:
            print("Training aborted due to NaN loss")
            return

        # Validation phase
        avg_val_loss = validate(
            model, val_loader, val_criterion, device, VAL_BATCH_SIZE
        )

        epoch_time = time.time() - epoch_start
        print(f"\n  Epoch {epoch+1} Summary:")
        print(f"    Train Loss: {avg_train_loss:.4f}")
        print(f"    Val Loss:   {avg_val_loss:.4f}")
        print(f"    Time:       {epoch_time:.1f}s")

        # GPU memory info
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"    GPU Memory: {mem_used:.2f} GB (peak)")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, 'demo_best.pt')
            torch.save(model.state_dict(), save_path)
            print(f"    Model saved: {save_path}")

    # ========== Summary ==========
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  DEMO TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total Time:        {total_time:.1f}s")
    print(f"  Best Val Loss:     {best_val_loss:.4f}")
    print(f"  Checkpoint:        {os.path.join(save_dir, 'demo_best.pt')}")
    print(f"{'='*60}")

    # ========== Validation checklist ==========
    print(f"\n  Validation Checklist:")
    print(f"    [{'OK' if avg_train_loss is not None else 'FAIL'}] Forward pass completed")
    print(f"    [{'OK' if not np.isnan(avg_train_loss) else 'FAIL'}] No NaN in loss")
    print(f"    [{'OK' if avg_train_loss < 100 else 'WARN'}] Loss in reasonable range")
    print(f"    [{'OK' if os.path.exists(os.path.join(save_dir, 'demo_best.pt')) else 'FAIL'}] "
          f"Checkpoint saved")
    print()


if __name__ == '__main__':
    main()
