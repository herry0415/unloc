# -*- coding:utf-8 -*-
"""
Demo Dataset - Generates synthetic data for testing training pipeline
without requiring real RobotCar dataset
"""

import numpy as np
from torch.utils import data

# Grid size matching model config (output_shape in semantickitti.yaml)
DEFAULT_GRID_SIZE = [480, 360, 32]
# Feature dimension matching model config (fea_dim in semantickitti.yaml)
DEFAULT_FEA_DIM = 8


class DemoDataset(data.Dataset):
    """
    Generates random data in RobotCar format for pipeline testing.

    Output format (11-tuple):
    data[0] - batch_id (placeholder)
    data[1] - vox_tenl (left LiDAR voxel coords) - List[np.array(N, 3)]
    data[2] - pt_fea_tenl (left LiDAR point features) - List[np.array(N, fea_dim)]
    data[3] - placeholder (unused)
    data[4] - vox_tenr (right LiDAR voxel coords) - List[np.array(N, 3)]
    data[5] - pt_fea_tenr (right LiDAR point features) - List[np.array(N, fea_dim)]
    data[6] - mono_left (left camera image) - np.array(3, 512, 512)
    data[7] - mono_right (right camera image) - np.array(3, 512, 512)
    data[8] - mono_rear (rear camera image) - np.array(3, 512, 512)
    data[9] - radar_image (radar image) - np.array(1, 512, 512)
    data[10] - labels (6DoF: 3 rotation + 3 translation) - np.array(6,)
    """

    def __init__(self, num_samples=10, batch_size=2, seed=42,
                 grid_size=None, fea_dim=DEFAULT_FEA_DIM):
        """
        Args:
            num_samples: Number of samples to generate
            batch_size: For reference (not used here, used in DataLoader)
            seed: Random seed for reproducibility
            grid_size: Voxel grid dimensions [x, y, z] (default: [480, 360, 32])
            fea_dim: Point feature dimension (default: 8, matching config)
        """
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.grid_size = grid_size or DEFAULT_GRID_SIZE
        self.fea_dim = fea_dim
        np.random.seed(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate a single sample of synthetic data.

        Returns:
            tuple: (data_tuple, xyz_placeholder)
                data_tuple contains 11 elements as described above
                xyz_placeholder is a dummy point cloud array
        """

        # ========== LiDAR Data (Left) ==========
        # Use fewer points (200-500) to keep memory reasonable for demo
        num_points_left = np.random.randint(200, 500)

        # Voxel coordinates must be within grid_size bounds:
        #   dim0: [0, grid_size[0])  e.g. [0, 480)
        #   dim1: [0, grid_size[1])  e.g. [0, 360)
        #   dim2: [0, grid_size[2])  e.g. [0, 32)
        voxel_coords_left = np.column_stack([
            np.random.randint(0, self.grid_size[0], size=num_points_left),
            np.random.randint(0, self.grid_size[1], size=num_points_left),
            np.random.randint(0, self.grid_size[2], size=num_points_left),
        ]).astype(np.int32)

        # Point features: fea_dim dimensions
        # In cylinder3D, features are typically:
        # (x, y, z, r, theta, z_cyl, intensity, elongation) = 8 dims
        pt_features_left = np.random.randn(num_points_left, self.fea_dim).astype(np.float32)

        # Wrap in list (as collate_fn expects List of arrays for batching)
        vox_tenl = [voxel_coords_left]
        pt_fea_tenl = [pt_features_left]

        # ========== LiDAR Data (Right) ==========
        num_points_right = np.random.randint(200, 500)
        voxel_coords_right = np.column_stack([
            np.random.randint(0, self.grid_size[0], size=num_points_right),
            np.random.randint(0, self.grid_size[1], size=num_points_right),
            np.random.randint(0, self.grid_size[2], size=num_points_right),
        ]).astype(np.int32)
        pt_features_right = np.random.randn(num_points_right, self.fea_dim).astype(np.float32)

        vox_tenr = [voxel_coords_right]
        pt_fea_tenr = [pt_features_right]

        # ========== Camera Images (RGB) ==========
        # Shape: (3, 512, 512) - 3 channels, 512x512 resolution
        # Use float32 in [0, 1] range (matching torchvision transforms.ToTensor() output)
        mono_left = np.random.rand(3, 512, 512).astype(np.float32)
        mono_right = np.random.rand(3, 512, 512).astype(np.float32)
        mono_rear = np.random.rand(3, 512, 512).astype(np.float32)

        # ========== Radar Image (Grayscale) ==========
        # Shape: (1, 512, 512) - 1 channel, 512x512 resolution
        radar_image = np.random.rand(1, 512, 512).astype(np.float32)

        # ========== Labels (6DoF Pose) ==========
        # 6DoF: [rotation_x, rotation_y, rotation_z, translation_x, translation_y, translation_z]
        # Rotation: small angles in radians (-0.1 to 0.1)
        # Translation: position in meters (-10 to 10)
        rotation = np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)
        translation = np.random.uniform(-10, 10, size=3).astype(np.float32)
        labels = np.concatenate([rotation, translation]).astype(np.float32)

        # ========== Assemble Data Tuple ==========
        # 11 elements as expected by TrainModel.py
        data_tuple = (
            0,                    # data[0]: batch_id (placeholder)
            vox_tenl,            # data[1]: left LiDAR voxel coords
            pt_fea_tenl,         # data[2]: left LiDAR point features
            np.array([0]),       # data[3]: placeholder
            vox_tenr,            # data[4]: right LiDAR voxel coords
            pt_fea_tenr,         # data[5]: right LiDAR point features
            mono_left,           # data[6]: left camera image
            mono_right,          # data[7]: right camera image
            mono_rear,           # data[8]: rear camera image
            radar_image,         # data[9]: radar image
            labels               # data[10]: 6DoF labels
        )

        # Dummy point cloud (used in some older code paths)
        xyz_dummy = np.random.randn(num_points_left, 3).astype(np.float32)

        return data_tuple, xyz_dummy


def collate_fn_demo(batch):
    """
    Custom collate function for batching demo data.

    Handles the special 11-tuple structure and converts lists to tensors.

    Args:
        batch: List of samples from DemoDataset

    Returns:
        Batched data in the same format as expected by TrainModel.py
    """
    # Unpack batch
    batch_data = [item[0] for item in batch]
    batch_xyz = [item[1] for item in batch]

    batch_size = len(batch)

    # Process each element of the 11-tuple
    # data[0]: batch_id
    batch_ids = np.array([item[0] for item in batch_data])

    # data[1]: vox_tenl - List of voxel coords, concatenate and stack
    vox_tenl_list = []
    for item in batch_data:
        vox_tenl_list.extend(item[1])  # item[1] is already a list

    # data[2]: pt_fea_tenl - List of point features
    pt_fea_tenl_list = []
    for item in batch_data:
        pt_fea_tenl_list.extend(item[2])

    # data[3]: placeholder
    placeholder = np.array([item[3] for item in batch_data])

    # data[4]: vox_tenr
    vox_tenr_list = []
    for item in batch_data:
        vox_tenr_list.extend(item[4])

    # data[5]: pt_fea_tenr
    pt_fea_tenr_list = []
    for item in batch_data:
        pt_fea_tenr_list.extend(item[5])

    # data[6-8]: Camera images - Stack into batch
    mono_left = np.stack([item[6] for item in batch_data], axis=0)
    mono_right = np.stack([item[7] for item in batch_data], axis=0)
    mono_rear = np.stack([item[8] for item in batch_data], axis=0)

    # data[9]: Radar images - Stack into batch
    radar_image = np.stack([item[9] for item in batch_data], axis=0)

    # data[10]: Labels - Stack into batch
    labels = np.stack([item[10] for item in batch_data], axis=0)

    # Assemble output in the same order as original data
    output = (
        batch_ids,
        vox_tenl_list,
        pt_fea_tenl_list,
        placeholder,
        vox_tenr_list,
        pt_fea_tenr_list,
        mono_left,
        mono_right,
        mono_rear,
        radar_image,
        labels
    )

    return output


if __name__ == '__main__':
    # Test the demo dataset
    print("Testing DemoDataset...")
    dataset = DemoDataset(num_samples=5)

    print(f"Dataset length: {len(dataset)}")
    print(f"Feature dimension: {dataset.fea_dim}")
    print(f"Grid size: {dataset.grid_size}")

    for i in range(2):
        data_tuple, xyz = dataset[i]
        print(f"\nSample {i}:")
        print(f"  data[1] (vox_tenl): shape={data_tuple[1][0].shape}, "
              f"range=[{data_tuple[1][0].min()}, {data_tuple[1][0].max()}]")
        print(f"  data[2] (pt_fea_tenl): shape={data_tuple[2][0].shape}")
        print(f"  data[6] (mono_left): shape={data_tuple[6].shape}, "
              f"dtype={data_tuple[6].dtype}")
        print(f"  data[9] (radar): shape={data_tuple[9].shape}")
        print(f"  data[10] (labels): shape={data_tuple[10].shape}, "
              f"values={data_tuple[10]}")

    # Test collate function
    print("\n--- Testing collate_fn_demo ---")
    import torch
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, collate_fn=collate_fn_demo)
    for batch in loader:
        print(f"Batch vox_tenl: {len(batch[1])} items, "
              f"first shape={batch[1][0].shape}")
        print(f"Batch pt_fea_tenl: {len(batch[2])} items, "
              f"first shape={batch[2][0].shape}")
        print(f"Batch mono_left: shape={batch[6].shape}")
        print(f"Batch radar: shape={batch[9].shape}")
        print(f"Batch labels: shape={batch[10].shape}")
        break
    print("\nDemoDataset test PASSED")
