# 分布式训练代码改动详解 - 改前改后对比

## 目录
1. [参数解析](#1-参数解析)
2. [分布式初始化](#2-分布式初始化)
3. [数据加载](#3-数据加载)
4. [模型包装](#4-模型包装)
5. [训练循环](#5-训练循环)
6. [模型保存加载](#6-模型保存加载)
7. [完整函数对比](#7-完整函数对比)

---

## 1. 参数解析

### 改前 (单卡训练)
```python
parser = argparse.ArgumentParser(description='HeRCULES training')
parser.add_argument('--sequence', type=str, default='Library')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--val_batch_size', type=int, default=2)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='checkpoints')
```

### 改后 (分布式训练)
```python
parser = argparse.ArgumentParser(description='HeRCULES DDP training')
parser.add_argument('--sequence', type=str, default='Library')
parser.add_argument('--batch_size', type=int, default=2,
                    help='Batch size per GPU')          # ← 改：每个GPU的batch_size
parser.add_argument('--val_batch_size', type=int, default=2,
                    help='Batch size per GPU')          # ← 改
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='checkpoints')

# ★ 新增：DDP参数
parser.add_argument('--num_gpus', type=int, default=1, choices=[1, 2, 3, 4],
                    help='Number of GPUs to use for distributed training')
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of data loading workers per GPU')
parser.add_argument('--local_rank', type=int, default=0,
                    help='Local rank for distributed training (auto-set by launcher)')
```

---

## 2. 分布式初始化

### 改前 (单卡训练)
```python
# 无需任何初始化代码
```

### 改后 (分布式训练)
```python
# ★ 新增：分布式工具函数
def setup_distributed(num_gpus):
    """Initialize distributed training."""
    if num_gpus == 1:
        # 单GPU模式 - 兼容性模式
        local_rank = args.local_rank if hasattr(args, 'local_rank') else 0
        return False, local_rank, 1

    # 多GPU DDP模式
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
```

---

## 3. 数据加载

### 改前 (单卡训练)
```python
def build_dataloaders():
    """Build HeRCULES data loaders."""
    from data.hercules_fusion import HerculesFusion
    from dataloader.hercules_dataset import hercules_cylinder_dataset, collate_fn_BEV
    from torch.utils.data import DataLoader

    print("\n--- Building Data Loaders ---")

    # 训练集
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
        shuffle=True,              # ← 直接shuffle
        collate_fn=collate_fn_BEV,
        num_workers=4,             # ← 固定4个worker
        drop_last=True
    )

    # 验证集
    val_pc_dataset = HerculesFusion(...)
    val_cyl_dataset = hercules_cylinder_dataset(...)
    val_loader = DataLoader(
        val_cyl_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn_BEV,
        num_workers=4,
        drop_last=False
    )

    print(f"  Train samples: {len(train_cyl_dataset)}")
    print(f"  Val samples: {len(val_cyl_dataset)}")
    print(f"  Train batches per epoch: {len(train_loader)}")

    return train_loader, val_loader
```

### 改后 (分布式训练)
```python
def build_dataloaders(rank, world_size, is_distributed):
    """Build HeRCULES data loaders with DDP support."""
    from data.hercules_fusion import HerculesFusion
    from dataloader.hercules_dataset import hercules_cylinder_dataset, collate_fn_BEV
    from torch.utils.data import DataLoader, DistributedSampler  # ← 新增import

    if is_main_process(rank):
        log_print("\n--- Building Data Loaders (DDP) ---")

    # 训练集
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

    # ★ 关键改动：根据is_distributed选择sampler
    if is_distributed:
        # DDP模式：使用DistributedSampler分割数据
        train_sampler = DistributedSampler(
            train_cyl_dataset,
            num_replicas=world_size,  # GPU总数
            rank=rank,                 # 当前GPU ID
            shuffle=True,
            drop_last=True
        )
        train_loader = DataLoader(
            train_cyl_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,     # ← 用sampler替代shuffle
            collate_fn=collate_fn_BEV,
            num_workers=args.num_workers,  # ← 使用参数化的num_workers
            pin_memory=True             # ← 新增，加速数据传输
        )
    else:
        # 单GPU模式：使用普通DataLoader
        train_loader = DataLoader(
            train_cyl_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_BEV,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True
        )

    # 验证集 (类似)
    val_pc_dataset = HerculesFusion(...)
    val_cyl_dataset = hercules_cylinder_dataset(...)

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
```

**改动要点：**
- ✅ 新增`rank`, `world_size`, `is_distributed`参数
- ✅ 使用`DistributedSampler`而不是`shuffle=True`
- ✅ 条件判断：DDP模式vs单GPU模式
- ✅ 每个GPU的数据分割由`DistributedSampler`自动处理

---

## 4. 模型包装

### 改前 (单卡训练)
```python
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
```

### 改后 (分布式训练)
```python
def build_model(device, rank, is_distributed):
    """Build Fusion model with DDP wrapper."""
    from FusionModel import Fusionmodel

    if is_main_process(rank):
        log_print("\n--- Building Fusion Model ---")

    model = Fusionmodel()
    model.to(device)

    # ★ 关键：根据is_distributed包装模型
    if is_distributed:
        local_rank = args.local_rank if hasattr(args, 'local_rank') else rank
        model = DDP(model, device_ids=[local_rank],
                   output_device=local_rank,
                   find_unused_parameters=True)  # ← spconv需要这个参数

    if is_main_process(rank):
        # ★ 改动：参数计数时需要判断是否使用.module
        if is_distributed:
            total_params = sum(p.numel() for p in model.module.parameters())
            trainable_params = sum(p.numel() for p in model.module.parameters()
                                  if p.requires_grad)
        else:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters()
                                  if p.requires_grad)

        log_print(f"  Total parameters: {total_params:,}")
        log_print(f"  Trainable parameters: {trainable_params:,}")
        if is_distributed:
            local_rank = args.local_rank if hasattr(args, 'local_rank') else rank
            log_print(f"  DDP: Enabled (device: cuda:{local_rank})")

    return model
```

**改动要点：**
- ✅ 新增`rank`, `is_distributed`参数
- ✅ 用`DDP`包装模型（当`is_distributed=True`）
- ✅ 参数访问改为`model.module.parameters()`（DDP模式）
- ✅ `find_unused_parameters=True`用于spconv兼容

---

## 5. 训练循环

### 改前 (单卡训练)
```python
def train_one_epoch(model, train_loader, criterion, optimizer,
                    device, epoch, num_epochs):
    """Train for one epoch with tqdm progress bar."""
    model.train()
    total_loss = 0.0
    lidar_loss_sum = 0.0
    radar_loss_sum = 0.0
    camera_loss_sum = 0.0
    num_batches = 0

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f'Epoch {epoch + 1}/{num_epochs}',
        ncols=120,
        ascii=True,
        leave=True
    )

    for i_iter, data in pbar:
        actual_bs = data[10].shape[0]

        # 提取数据
        train_vox_tenl = [torch.from_numpy(i).to(device) for i in data[1]]
        train_pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(device)
                             for i in data[2]]
        train_vox_tenr = [torch.from_numpy(i).to(device) for i in data[4]]
        train_pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(device)
                             for i in data[5]]
        monoleft = torch.from_numpy(data[6]).float().to(device)
        labels = torch.from_numpy(data[10]).float().to(device)
        transgt = labels[:, 0:3]
        rotgt = labels[:, 3:6]

        optimizer.zero_grad()

        # ★ 原方式：合并后一次backward
        trans_lidar, rot_lidar = model([train_pt_fea_tenl, train_vox_tenl, actual_bs])
        loss_lidar = criterion(trans_lidar, rot_lidar, rotgt, transgt)

        trans_radar, rot_radar = model([train_pt_fea_tenr, train_vox_tenr, actual_bs])
        loss_radar = criterion(trans_radar, rot_radar, rotgt, transgt)

        trans_camera, rot_camera = model([monoleft])
        loss_camera = criterion(trans_camera, rot_camera, rotgt, transgt)

        loss = loss_lidar + loss_radar + loss_camera
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        lidar_loss_sum += loss_lidar.item()
        radar_loss_sum += loss_radar.item()
        camera_loss_sum += loss_camera.item()
        num_batches += 1

        avg_loss = total_loss / num_batches
        avg_lidar = lidar_loss_sum / num_batches
        avg_radar = radar_loss_sum / num_batches
        avg_camera = camera_loss_sum / num_batches

        pbar.set_postfix({
            'Loss': f'{avg_loss:.6f}',
            'L': f'{avg_lidar:.4f}',
            'R': f'{avg_radar:.4f}',
            'C': f'{avg_camera:.4f}'
        })

    pbar.close()
    avg_loss = total_loss / num_batches
    return avg_loss
```

### 改后 (分布式训练)
```python
def train_one_epoch(model, train_loader, criterion, optimizer,
                    device, epoch, num_epochs, rank, is_distributed):  # ★ 新增参数
    """Train for one epoch with tqdm progress bar."""
    model.train()
    total_loss = 0.0
    lidar_loss_sum = 0.0
    radar_loss_sum = 0.0
    camera_loss_sum = 0.0
    num_batches = 0

    # ★ 新增：DDP模式下需要设置sampler的epoch
    if is_distributed:
        train_loader.sampler.set_epoch(epoch)

    # ★ 改动：只在rank 0显示进度条
    if is_main_process(rank):
        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'Epoch {epoch + 1}/{num_epochs}',
            ncols=120,
            ascii=True,
            leave=True
        )
        iterator = pbar
    else:
        pbar = None
        iterator = enumerate(train_loader)

    for i_iter, data in iterator:
        actual_bs = data[10].shape[0]

        # 提取数据 (代码相同，略)
        train_vox_tenl = [torch.from_numpy(i).to(device) for i in data[1]]
        # ... (其他数据提取)

        optimizer.zero_grad()

        # ★ 改动：3个forward + 3个独立backward（避免spconv inplace冲突）
        trans_lidar, rot_lidar = model([train_pt_fea_tenl, train_vox_tenl, actual_bs])
        loss_lidar = criterion(trans_lidar, rot_lidar, rotgt, transgt)
        loss_lidar.backward()  # ← 独立backward

        trans_radar, rot_radar = model([train_pt_fea_tenr, train_vox_tenr, actual_bs])
        loss_radar = criterion(trans_radar, rot_radar, rotgt, transgt)
        loss_radar.backward()  # ← 独立backward

        trans_camera, rot_camera = model([monoleft])
        loss_camera = criterion(trans_camera, rot_camera, rotgt, transgt)
        loss_camera.backward()  # ← 独立backward

        # 梯度已累积，统一更新
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ★ 改动：累加loss（现在是三个.item()的和）
        loss = loss_lidar.item() + loss_radar.item() + loss_camera.item()
        total_loss += loss
        lidar_loss_sum += loss_lidar.item()
        radar_loss_sum += loss_radar.item()
        camera_loss_sum += loss_camera.item()
        num_batches += 1

        # ★ 改动：只在rank 0更新进度条
        if is_main_process(rank) and pbar is not None:
            avg_loss = total_loss / num_batches
            avg_lidar = lidar_loss_sum / num_batches
            avg_radar = radar_loss_sum / num_batches
            avg_camera = camera_loss_sum / num_batches

            pbar.set_postfix({
                'Loss': f'{avg_loss:.6f}',
                'L': f'{avg_lidar:.4f}',
                'R': f'{avg_radar:.4f}',
                'C': f'{avg_camera:.4f}'
            })

    if pbar is not None:
        pbar.close()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss
```

**改动要点：**
- ✅ `train_loader.sampler.set_epoch(epoch)` - 每个epoch重新shuffle
- ✅ 条件化进度条 - 只在rank 0显示
- ✅ 3个独立backward替代1个合并backward
- ✅ 进度条更新也条件化

---

## 6. 模型保存加载

### 改前 (单卡训练)
```python
# 保存最佳模型
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_epoch = epoch + 1
    checkpoint_path = os.path.join(args.output_dir, 'hercules_best.pt')
    torch.save(model.state_dict(), checkpoint_path)  # ← 直接保存
    print(f"  ⭐ New best model! Val Loss: {val_loss:.6f}\n")

# 周期性保存
if (epoch + 1) % 10 == 0:
    periodic_path = os.path.join(args.output_dir, f'hercules_epoch_{epoch + 1}.pt')
    torch.save(model.state_dict(), periodic_path)
    print(f"  💾 Periodic checkpoint: hercules_epoch_{epoch + 1}.pt\n")
```

### 改后 (分布式训练)
```python
# ★ 改动：只在rank 0保存
if is_main_process(rank):
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        checkpoint_path = os.path.join(args.output_dir, 'hercules_best.pt')
        if is_distributed:
            # ★ 用.module提取原模型参数
            torch.save(model.module.state_dict(), checkpoint_path)
        else:
            torch.save(model.state_dict(), checkpoint_path)
        log_print(f"  ⭐ New best model! Val Loss: {val_loss:.6f}\n")

    # 周期性保存
    if (epoch + 1) % 5 == 0:  # ← 改为5个epoch保存一次
        periodic_path = os.path.join(args.output_dir, f'hercules_epoch_{epoch + 1}.pt')
        if is_distributed:
            torch.save(model.module.state_dict(), periodic_path)
        else:
            torch.save(model.state_dict(), periodic_path)
        log_print(f"  💾 Periodic checkpoint: hercules_epoch_{epoch + 1}.pt\n")
```

### 模型加载 (改后)
```python
# 加载检查点
if args.checkpoint and os.path.exists(args.checkpoint):
    if is_main_process(rank):
        log_print(f"\n--- Loading Checkpoint ---")

    state_dict = torch.load(args.checkpoint, map_location=device)

    if is_distributed:
        model.module.load_state_dict(state_dict)  # ← 加载到.module
    else:
        model.load_state_dict(state_dict)

    if is_main_process(rank):
        log_print(f"  Loaded: {args.checkpoint}")
```

**改动要点：**
- ✅ 只在rank 0保存（避免重复）
- ✅ DDP模式使用`model.module.state_dict()`
- ✅ 单GPU模式直接`model.state_dict()`
- ✅ 加载时也使用条件化加载

---

## 7. 完整函数对比

### 环境检测

#### 改前
```python
def check_environment():
    """Pre-flight checks before training."""
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
```

#### 改后
```python
def check_environment(rank, is_distributed):
    """Pre-flight checks before training."""
    # ★ 改动：使用log_print，只在rank 0输出
    if is_main_process(rank):
        log_print("=" * 70)
        log_print("  HERCULES MULTI-MODAL FUSION TRAINING (DDP)")
        log_print("=" * 70)

    if torch.cuda.is_available():
        # ★ 改动：使用local_rank进行GPU分配
        local_rank = args.local_rank if hasattr(args, 'local_rank') else rank
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
        log_print(f"  Sequence: {args.sequence}")
        log_print(f"  Batch size per GPU: {args.batch_size}")
        # ★ 新增：显示总batch_size
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
```

### 验证函数

#### 改前
```python
def validate(model, val_loader, criterion, device):
    """Validate the model with progress bar."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc='Validating',
        ncols=120,
        ascii=True,
        leave=True
    )

    with torch.no_grad():
        for i_iter, data in pbar:
            # ... 数据提取和前向传播 ...
            loss = loss_l + loss_r + loss_c
            total_loss += loss.item()
            num_batches += 1

            avg_loss = total_loss / num_batches
            pbar.set_postfix({'Loss': f'{avg_loss:.6f}'})

    pbar.close()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss
```

#### 改后
```python
def validate(model, val_loader, criterion, device, rank, is_distributed):  # ★ 新增参数
    """Validate the model with progress bar."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # ★ 改动：只在rank 0显示进度条
    if is_main_process(rank):
        pbar = tqdm(
            enumerate(val_loader),
            total=len(val_loader),
            desc='Validating',
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
            # ... 数据提取和前向传播 ...
            loss = loss_l + loss_r + loss_c
            total_loss += loss.item()
            num_batches += 1

            # ★ 改动：只在rank 0更新进度条
            if is_main_process(rank) and pbar is not None:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'Loss': f'{avg_loss:.6f}'})

    if pbar is not None:
        pbar.close()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # ★ 新增：同步所有GPU的loss，用于checkpoint决策
    if is_distributed:
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor / dist.get_world_size()).item()

    return avg_loss
```

---

## 总结表：关键改动清单

| 功能点 | 改前 | 改后 | 必要性 |
|------|------|------|--------|
| 导入DDP库 | ❌ | ✅ `import torch.distributed as dist` | ✅ 必要 |
| 新增参数 | ❌ | ✅ `--num_gpus`, `--local_rank`, `--num_workers` | ✅ 必要 |
| 初始化分布式 | ❌ | ✅ `setup_distributed()` | ✅ 必要 |
| 数据采样器 | `shuffle=True` | ✅ `DistributedSampler` | ✅ 必要 |
| 模型包装 | ❌ 无 | ✅ `DDP(model, ...)` | ✅ 必要 |
| 参数访问 | `model.parameters()` | ✅ `model.module.parameters()` (DDP) | ✅ 必要 |
| 进度条 | 所有进程 | ✅ 仅rank 0 | ✅ 必要 |
| 模型保存 | 所有进程 | ✅ 仅rank 0 + `.module.state_dict()` | ✅ 必要 |
| 验证同步 | 无 | ✅ `dist.all_reduce()` | ✅ 必要 |
| Forward方式 | `loss.backward()` | ✅ 3个独立backward | ⚠️ spconv优化 |
| Cleanup | ❌ 无 | ✅ `cleanup_distributed()` | ✅ 必要 |

