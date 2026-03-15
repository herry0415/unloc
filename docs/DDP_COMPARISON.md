# 单卡训练 vs 分布式训练(DDP)对比

## 核心答案
**模型代码(FusionModel.py)不需要改！** ✅
- FusionModel.py 保持完全不变
- 只需要在**训练脚本**中添加DDP逻辑
- DDP通过`torch.nn.parallel.DistributedDataParallel`包装模型，对模型代码透明

---

## 详细对比表

### 1. 导入和初始化

| 项目 | 单卡训练(hercules_train.py) | 分布式训练(hercules_train_ddp.py) |
|------|-------------------------|--------------------------|
| 导入 | `import torch` | `import torch.distributed as dist` <br> `from torch.nn import DistributedDataParallel as DDP` |
| 初始化 | 无需初始化 | `dist.init_process_group(backend='nccl')` |
| 清理 | 无需清理 | `dist.destroy_process_group()` |

---

### 2. 参数和配置

| 项目 | 单卡训练 | 分布式训练 |
|------|--------|----------|
| 新增参数 | - | `--num_gpus` (1/2/3/4) <br> `--local_rank` (auto-set by launcher) <br> `--num_workers` (per GPU) |
| batch_size含义 | 总batch_size | **每个GPU的batch_size** <br> 总batch = batch_size × num_gpus |
| 其他参数 | 不变 | 不变 |

**使用示例对比：**
```bash
# 单卡训练
python scripts/hercules_train.py --batch_size 2

# 分布式训练(1张GPU, 等价)
python scripts/hercules_train_ddp.py --batch_size 2 --num_gpus 1

# 分布式训练(2张GPU)
python scripts/hercules_train_ddp.py --batch_size 2 --num_gpus 2  # 总batch=4
```

---

### 3. 环境检测和设备管理

#### 单卡训练:
```python
def check_environment():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # 固定GPU 0
    else:
        device = torch.device('cpu')
```

#### 分布式训练:
```python
def setup_distributed(num_gpus):
    if num_gpus == 1:
        return False, local_rank, 1  # 单卡模式

    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')  # 初始化DDP

    rank = dist.get_rank()           # 进程ID
    world_size = dist.get_world_size()  # 总进程数
    return True, rank, world_size

def check_environment(rank, is_distributed):
    # 获取local_rank用于GPU分配
    local_rank = args.local_rank
    device = torch.device(f'cuda:{local_rank}')
```

**区别：**
- 单卡：直接用GPU 0
- DDP：通过rank/local_rank动态分配GPU

---

### 4. 数据加载器

#### 单卡训练:
```python
train_loader = DataLoader(
    train_cyl_dataset,
    batch_size=args.batch_size,
    shuffle=True,           # 直接shuffle
    collate_fn=collate_fn_BEV,
    num_workers=4,
    drop_last=True
)
```

#### 分布式训练:
```python
if is_distributed:
    # 使用DistributedSampler分配数据到不同GPU
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
        sampler=train_sampler,     # 用sampler代替shuffle
        collate_fn=collate_fn_BEV,
        num_workers=args.num_workers,
        pin_memory=True
    )
else:
    # num_gpus=1时，用普通DataLoader(不用DistributedSampler)
    train_loader = DataLoader(...)
```

**关键点：**
- `DistributedSampler`确保每个GPU获得不同的数据子集
- `shuffle=True`变为`sampler`参数
- 每个epoch需要`sampler.set_epoch(epoch)`重新shuffle

---

### 5. 模型包装

#### 单卡训练:
```python
def build_model(device):
    model = Fusionmodel()
    model.to(device)
    return model
```

#### 分布式训练:
```python
def build_model(device, rank, is_distributed):
    model = Fusionmodel()
    model.to(device)

    if is_distributed:
        # 关键：用DDP包装模型
        model = DDP(model, device_ids=[local_rank],
                   output_device=local_rank,
                   find_unused_parameters=True)

    return model
```

**DDP包装的影响：**
- 模型结构不变
- 参数访问方式改变：`model.module.parameters()` (DDP情况)
- 不同GPU上运行相同模型，共享参数梯度

---

### 6. 训练循环中的前向后向

#### 单卡训练:
```python
def train_one_epoch(model, train_loader, ...):
    for i_iter, data in enumerate(train_loader):
        # 3个forward pass，合并后一起backward
        trans_lidar, rot_lidar = model([...])
        loss_lidar = criterion(...)

        trans_radar, rot_radar = model([...])
        loss_radar = criterion(...)

        trans_camera, rot_camera = model([...])
        loss_camera = criterion(...)

        # 一次累积backward
        loss = loss_lidar + loss_radar + loss_camera
        loss.backward()
        optimizer.step()
```

#### 分布式训练:
```python
def train_one_epoch(model, train_loader, ..., is_distributed):
    if is_distributed:
        train_loader.sampler.set_epoch(epoch)  # 关键：重新shuffle

    for i_iter, data in iterator:
        # 改为：3个forward + 3个独立backward
        # (避免spconv的inplace冲突)

        trans_lidar, rot_lidar = model([...])
        loss_lidar = criterion(...)
        loss_lidar.backward()        # 独立backward

        trans_radar, rot_radar = model([...])
        loss_radar = criterion(...)
        loss_radar.backward()        # 独立backward

        trans_camera, rot_camera = model([...])
        loss_camera = criterion(...)
        loss_camera.backward()       # 独立backward

        # 梯度累积，统一更新
        optimizer.step()
```

**区别：**
- 单卡：`loss.backward()` 一次
- DDP：3次独立`backward()`来累积梯度
  - 原因：spconv库inplace操作冲突
  - 结果相同，但分开处理更稳定

---

### 7. 进度条和日志输出

#### 单卡训练:
```python
def train_one_epoch(model, train_loader, ...):
    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        ...
    )

    for i_iter, data in pbar:
        # ... 训练代码
        pbar.set_postfix({...})
```

#### 分布式训练:
```python
def train_one_epoch(model, train_loader, ..., rank, is_distributed):
    if is_main_process(rank):        # 只有rank 0显示进度条
        pbar = tqdm(...)
        iterator = pbar
    else:
        pbar = None
        iterator = enumerate(train_loader)

    for i_iter, data in iterator:
        # ... 训练代码
        if is_main_process(rank) and pbar is not None:
            pbar.set_postfix({...})
```

**关键函数：**
```python
def is_main_process(rank):
    return rank == 0

def log_print(msg, rank=0):
    if is_main_process(rank):
        print(msg)
```

**原因：** 多GPU环境下，每个进程都会输出，会刷屏。只让rank 0输出。

---

### 8. 验证和同步

#### 单卡训练:
```python
def validate(model, val_loader, criterion, device):
    # 直接返回该GPU上的loss
    avg_loss = total_loss / num_batches
    return avg_loss
```

#### 分布式训练:
```python
def validate(model, val_loader, criterion, device, rank, is_distributed):
    avg_loss = total_loss / num_batches

    if is_distributed:
        # 关键：同步所有GPU的loss
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor / dist.get_world_size()).item()

    return avg_loss
```

**原因：** 验证时需要用全局loss做checkpoint决策，不能只用某一个GPU的loss

---

### 9. 模型保存

#### 单卡训练:
```python
if val_loss < best_val_loss:
    checkpoint_path = os.path.join(args.output_dir, 'hercules_best.pt')
    torch.save(model.state_dict(), checkpoint_path)  # 直接保存
```

#### 分布式训练:
```python
if is_main_process(rank):  # 只在rank 0保存
    if val_loss < best_val_loss:
        checkpoint_path = os.path.join(args.output_dir, 'hercules_best.pt')
        if is_distributed:
            torch.save(model.module.state_dict(), checkpoint_path)  # .module
        else:
            torch.save(model.state_dict(), checkpoint_path)
```

**关键点：**
- 只有rank 0保存（避免多个进程重复保存）
- DDP模型需要用`.module.state_dict()`提取原模型参数

---

### 10. 模型加载

#### 单卡训练:
```python
state_dict = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(state_dict)
```

#### 分布式训练:
```python
state_dict = torch.load(args.checkpoint, map_location=device)
if is_distributed:
    model.module.load_state_dict(state_dict)  # 加载到.module
else:
    model.load_state_dict(state_dict)

# 同时需要处理DDP checkpoint的'module.'前缀
if any(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k[7:]: v for k, v in state_dict.items()}
```

---

## 性能对比

| 指标 | 单卡 | DDP 2卡 | DDP 4卡 |
|------|-----|--------|--------|
| 每epoch时间 | ~11分钟 | ~6分钟/卡 | ~3分钟/卡 |
| batch_size/卡 | 2 | 2 | 2 |
| 总batch_size | 2 | 4 | 8 |
| 梯度有效性 | ✅ | ✅（同步后） | ✅（同步后） |

---

## 核心差异总结

| 维度 | 单卡训练 | 分布式训练 |
|------|--------|----------|
| **模型改动** | ❌ 不需要 | ❌ 不需要 |
| **数据分配** | 单个loader | DistributedSampler分割 |
| **梯度更新** | 每batch一次 | 多卡同步后更新 |
| **日志输出** | 全部进程输出 | 仅rank 0输出 |
| **参数检查点** | `model.state_dict()` | `model.module.state_dict()` |
| **初始化代码** | 无 | `dist.init_process_group()` |
| **cleanup** | 无 | `dist.destroy_process_group()` |

---

## 测试脚本

两个脚本都兼容，使用checkpoint时会自动处理：

```python
# hercules_test.py中的自动处理
state_dict = torch.load(checkpoint, map_location=device)
if any(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k[7:]: v for k, v in state_dict.items()}
    print("Detected DDP checkpoint, removed 'module.' prefix")
model.load_state_dict(state_dict)
```

✅ 无论从单卡还是DDP训练的checkpoint都能正确加载！

---

## 最佳实践建议

1. **快速实验** → 用单卡版本 (`hercules_train.py`)
   - 更简单，bug易发现
   - batch_size = 2，显存占用少

2. **大规模训练** → 用DDP版本 (`hercules_train_ddp.py`)
   - 多GPU并行，速度快
   - batch_size = 2/GPU，总batch_size更大，收敛更好

3. **切换策略**
   - 单卡验证收敛逻辑 → DDP训练
   - DDP训练产生的checkpoint可直接用单卡test加载
