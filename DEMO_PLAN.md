# 🎯 Demo 训练流程验证计划

## 📊 当前项目状态分析

### ✅ 已有
- ✓ 完整的神经网络架构 (FusionModel.py)
- ✓ 三个分支模型 (LiDAR, Camera, Radar)
- ✓ 训练脚本框架 (TrainModel.py)
- ✓ 预训练权重 (Models/BaseModel/*.pt)
- ✓ 配置系统 (config/config.py)

### ❌ 问题
- ✗ **硬编码数据路径** 指向不存在的位置 (data_builder.py: 行18, 45)
- ✗ **RobotCar 数据集** 缺失 (需要真实数据)
- ✗ **数据加载器** 依赖真实数据集格式
- ✗ **collate 函数** 依赖特定的数据结构
- ✗ **数据格式不清** 具体的11个元素结构不明确

### 🔴 核心痛点
目前无法运行训练，因为：
1. 数据加载器会在初始化时失败（路径不存在）
2. 即使路径存在，RobotCar 类也会查找真实文件

---

## 🎪 Demo 方案（3个层次）

### **方案 A: 最小化 Demo（推荐）✅**
创建**虚拟数据生成器**，完全不依赖真实数据

#### 优点
- ✓ 快速实现
- ✓ 不需要任何真实数据
- ✓ 能验证整个训练流程
- ✓ 能测试所有模块的数据维度

#### 步骤
1. **创建** `DemoDataset` 类 (dataloader/demo_dataset.py)
   - 继承 `torch.utils.data.Dataset`
   - 生成随机数据，符合11元组格式

2. **修改** `data_builder.py`
   - 添加 `use_demo=True` 参数
   - 当 demo 模式时，使用 DemoDataset 替代 RobotCar

3. **创建** `demo_train.py` 脚本
   - 简化版的 TrainModel.py
   - 只跑 5-10 个 batch
   - 输出清晰的日志

4. **创建** `demo_test.py` 脚本
   - 测试推理过程
   - 验证输出维度

#### 代码架构
```
dataloader/
├── demo_dataset.py      (新建) ← 虚拟数据生成
└── dataset_semantickitti.py

builder/
├── data_builder.py      (改动) ← 支持 demo 模式

scripts/
├── demo_train.py        (新建) ← 简化训练脚本
└── demo_test.py         (新建) ← 测试脚本
```

---

## 🏗️ 实现细节

### 1️⃣ **DemoDataset 类设计** (dataloader/demo_dataset.py)

```python
class DemoDataset(Dataset):
    """生成虚拟的 RobotCar 格式数据"""

    def __init__(self, num_samples=10, batch_size=2):
        self.num_samples = num_samples
        self.batch_size = batch_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 返回 (data, xyz) 格式
        # data = (batch_id, vox_tenl, pt_fea_tenl, ?, vox_tenr,
        #         pt_fea_tenr, mono_left, mono_right, mono_rear, radar, labels)

        return {
            'data': [
                0,                           # data[0]: batch_id
                [随机voxel坐标],             # data[1]: vox_tenl (3D array)
                [随机点云特征],              # data[2]: pt_fea_tenl (Nx5 array)
                随机数据,                    # data[3]: ? (占位符)
                [随机voxel坐标],             # data[4]: vox_tenr
                [随机点云特征],              # data[5]: pt_fea_tenr
                随机摄像头图像 (3,512,512),  # data[6]: mono_left
                随机摄像头图像 (3,512,512),  # data[7]: mono_right
                随机摄像头图像 (3,512,512),  # data[8]: mono_rear
                随机雷达图像 (1,512,512),   # data[9]: radar
                随机标签 (6,) [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]  # data[10]
            ],
            'xyz': 随机点云
        }
```

### 2️⃣ **data_builder.py 改动**

```python
def build(dataset_config, train_dataloader_config, val_dataloader_config,
          grid_size=[480, 360, 32], use_demo=False):

    if use_demo:
        # ✅ Demo 模式：使用虚拟数据
        train_dataset = DemoDataset(num_samples=20)
        val_dataset = DemoDataset(num_samples=5)
    else:
        # 原始逻辑：加载真实数据
        train_dataset = RobotCar(...)
        val_dataset = RobotCar(...)

    # 后续逻辑保持不变
    ...
```

### 3️⃣ **demo_train.py 脚本**

```python
# 简化的训练循环，只验证流程
main_features:
- 使用 demo 数据集（20 samples）
- 只跑 2 个 epoch
- 每 batch 打印详细日志：
  - 输入数据维度
  - 模型输出维度
  - Loss 值
  - 内存使用情况
- 验证反向传播是否正常
- 保存检查点
```

### 4️⃣ **demo_test.py 脚本**

```python
# 验证推理过程
main_features:
- 加载预训练权重（如可用）
- 单个 batch 推理
- 打印各分支的输出：
  - LiDAR 特征维度
  - Camera 特征维度
  - Radar 特征维度
  - 融合后的 6DoF 输出 (rotation, translation)
- 验证没有 NaN/Inf
```

---

## 📋 具体实现步骤（按顺序）

### **Phase 1: 数据生成（15 min）**
```
✓ 创建 dataloader/demo_dataset.py
  ├─ DemoDataset 类
  ├─ 虚拟数据生成逻辑
  └─ collate_fn 适配
```

### **Phase 2: 数据加载器改造（10 min）**
```
✓ 修改 builder/data_builder.py
  ├─ 添加 use_demo 参数
  ├─ Demo 模式逻辑
  └─ 保持原有接口兼容
```

### **Phase 3: 训练脚本（20 min）**
```
✓ 创建 scripts/demo_train.py
  ├─ 复制 TrainModel.py 核心逻辑
  ├─ 简化为 2 epoch
  ├─ 添加详细日志
  └─ 错误处理
```

### **Phase 4: 测试脚本（15 min）**
```
✓ 创建 scripts/demo_test.py
  ├─ 推理流程
  ├─ 维度验证
  └─ 输出检查
```

### **Phase 5: 运行 & 调试（30 min）**
```
✓ 运行 demo_train.py
  ├─ 检查是否有导入错误
  ├─ 检查数据维度是否匹配
  ├─ 检查损失是否收敛
  └─ 检查内存是否正常释放

✓ 运行 demo_test.py
  ├─ 验证推理输出
  └─ 修复任何问题
```

---

## 🎯 验证清单

运行后应该验证：

| 检查项 | 预期结果 | 状态 |
|------|--------|------|
| **导入** | 所有模块正确导入 | - |
| **数据加载** | DemoDataset 正确生成 11 元组 | - |
| **模型初始化** | FusionModel 加载成功 | - |
| **LiDAR 分支** | 输入→1024维输出 | - |
| **Camera 分支** | RGB(3,512,512)→1024维 | - |
| **Radar 分支** | Gray(1,512,512)→1024维 | - |
| **融合层** | 1024维→6维输出(rot+trans) | - |
| **前向传播** | 无 NaN/Inf | - |
| **反向传播** | 梯度计算无错误 | - |
| **Loss 下降** | 损失在 5 个 batch 后下降 | - |
| **模型保存** | 检查点保存成功 | - |

---

## 🔧 数据格式细节

### **11元组输出格式说明**

```python
data = (
    batch_id,              # data[0] - 标识符
    vox_tenl,              # data[1] - 左LiDAR 体素坐标 List[np.array]
    pt_fea_tenl,           # data[2] - 左LiDAR 点云特征 List[np.array(Nx5)]
    something3,            # data[3] - ??? (未使用或占位符)
    vox_tenr,              # data[4] - 右LiDAR 体素坐标 List[np.array]
    pt_fea_tenr,           # data[5] - 右LiDAR 点云特征 List[np.array(Nx5)]
    mono_left,             # data[6] - 左摄像头 (3, 512, 512)
    mono_right,            # data[7] - 右摄像头 (3, 512, 512)
    mono_rear,             # data[8] - 后摄像头 (3, 512, 512)
    radar_image,           # data[9] - 雷达图像 (1, 512, 512)
    labels                 # data[10] - 标签 (6,) [wx,wy,wz, tx,ty,tz]
)
```

### **各字段维度示例**
```
vox_tenl: 1个元素的List，其中包含 np.array(shape=(3, 30, 30, 32)) 坐标
pt_fea_tenl: 1个元素的List，其中包含 np.array(shape=(M, 5)) 特征
mono_left: np.array(shape=(3, 512, 512))
radar_image: np.array(shape=(1, 512, 512))
labels: np.array(shape=(6,)) 6DoF 标签
```

---

## 📝 预期输出日志

### demo_train.py 运行后
```
=== DEMO TRAINING START ===
Epoch 1/2
  Batch 1/10:
    ✓ LiDAR feature shape: torch.Size([batch, 1024])
    ✓ Camera feature shape: torch.Size([batch, 1024])
    ✓ Radar feature shape: torch.Size([batch, 1024])
    ✓ Fusion output shape: torch.Size([batch, 3]) + torch.Size([batch, 3])
    ✓ Loss: 12.456 | Rot Loss: 5.123 | Trans Loss: 7.333
    ✓ GPU Memory: 2.3GB
  Batch 2/10:
    ✓ Loss: 11.234 (↓ 1.222)
  ...
Epoch 1 Average Loss: 9.876
Epoch 2 Average Loss: 8.765 (↓ 1.111)
✓ Model saved to checkpoints/demo_best.pt
=== DEMO TRAINING COMPLETE ===
```

### demo_test.py 运行后
```
=== DEMO INFERENCE TEST ===
✓ Model loaded successfully
✓ Input prepared:
  - LiDAR: 2 scans
  - Camera: 3 images (RGB 512×512)
  - Radar: 1 image (512×512)
✓ Forward pass successful
  - LiDAR output: (2, 1024)
  - Camera output: (2, 1024)
  - Radar output: (2, 1024)
  - Fusion output (rotation): (2, 3)
  - Fusion output (translation): (2, 3)
✓ Output values in valid range (no NaN/Inf)
✓ Inference time: 0.234s per sample
=== DEMO TEST COMPLETE ===
```

---

## ⚠️ 可能的问题 & 解决方案

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| ImportError | 缺少依赖库 | 检查 requirements.txt，安装缺失的包 |
| CUDA out of memory | 批次太大 | 减小 demo dataset size 或 batch size |
| Model load fail | 权重路径不对 | 使用相对路径或检查路径设置 |
| Shape mismatch | collate_fn 处理有误 | 调试 collate 函数的输出维度 |
| NaN in loss | 初始化问题 | 检查网络权重初始化 |
| GPU not used | CUDA 未配置 | 检查 torch.cuda.is_available() |

---

## 🎓 预期收获

完成本 Demo 后，你将获得：

1. ✅ **清晰的数据流**：知道每个环节的确切数据维度
2. ✅ **可运行的训练脚本**：能在没有真实数据的情况下测试
3. ✅ **完整的错误日志**：知道哪些地方可能会出问题
4. ✅ **模块化测试**：能独立测试每个分支
5. ✅ **性能基线**：知道 loss 下降的速度
6. ✅ **代码改进方向**：清楚需要改什么才能适配真实数据

---

## 🚀 之后的步骤（超出本计划）

一旦 Demo 运行成功，可以：
1. 用真实数据替换 DemoDataset
2. 进行长周期训练
3. 添加验证指标
4. 优化超参数
5. 测试不同的数据源组合

---

## 💡 建议

**推荐方案：实施方案 A（最小化 Demo）**

理由：
- ⏱️ 最快实现（90 分钟）
- 🎯 最直接验证流程
- 🔧 易于调试和改进
- 📊 为后续适配提供基础

**不建议方案：**
- ❌ 方案 B (合成真实数据)：太复杂
- ❌ 方案 C (在线下载数据)：时间长且可能失败

---

## 📞 评审检查

请评审以下方面：
- [ ] 整体方向是否可行？
- [ ] 优先级是否合理？
- [ ] 是否有遗漏的步骤？
- [ ] 预期输出是否清晰？
- [ ] 是否需要调整方案？