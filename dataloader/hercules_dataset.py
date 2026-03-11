# -*- coding: utf-8 -*-
"""
HeRCULES Cylinder Dataset Wrapper

Wraps HerculesFusion with cylindrical voxelization for 3D scene understanding.
Converts point clouds to cylindrical coordinates and voxelizes them.

Returns 11-tuple compatible with training pipeline:
(voxel_pos_l, grid_ind_l, fea_l, voxel_pos_r, grid_ind_r, fea_r,
 mono_left, mono_right, mono_rear, radar_image_2d, pose)

Where:
- voxel_pos_l/r: Voxel positions in 3D grid (480, 360, 32)
- grid_ind_l/r: Grid indices of voxels
- fea_l/r: Features per voxel
- mono_left/right/rear: Camera images
- radar_image_2d: 2D radar image (all zeros)
- pose: 6DoF pose label
"""

import os
import sys
import numpy as np
from torch.utils import data
import torch

# ============================================================
# Helper Functions (from dataset_semantickitti.py)
# ============================================================

def cart2polar(input_xyz):
    """Convert Cartesian to cylindrical coordinates."""
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    """Convert cylindrical to Cartesian coordinates."""
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


def collate_fn_BEV(data):
    """
    Collate function for batch processing.
    Handles variable-size point clouds via sparse representation.
    """
    # Extract voxel positions (stack to batch)
    data2stackl = np.stack([d[0] for d in data]).astype(np.float32)
    grid_indl = [d[1] for d in data]
    xyzl = [d[2] for d in data]

    data2stackr = np.stack([d[3] for d in data]).astype(np.float32)
    grid_indr = [d[4] for d in data]
    xyzr = [d[5] for d in data]

    # Extract camera images (stack to batch)
    monoleft = np.stack([d[6] for d in data])
    monoright = np.stack([d[7] for d in data])
    monorear = np.stack([d[8] for d in data])
    radarimage = np.stack([d[9] for d in data])

    # Extract poses
    point_label = np.stack([d[10] for d in data])

    # Convert voxel positions to torch tensors
    return (
        torch.from_numpy(data2stackl),
        grid_indl,
        xyzl,
        torch.from_numpy(data2stackr),
        grid_indr,
        xyzr,
        monoleft,
        monoright,
        monorear,
        radarimage,
        point_label
    )


# ============================================================
# Main Cylinder Dataset Class
# ============================================================

class hercules_cylinder_dataset(data.Dataset):
    """
    HeRCULES Cylindrical Voxelization Dataset

    Wraps HerculesFusion with cylindrical coordinate voxelization.
    Each point cloud (LiDAR and Radar) is independently voxelized.
    """

    def __init__(self, point_cloud_dataset, grid_size=[480, 360, 32],
                 rotate_aug=False, flip_aug=False, ignore_label=255,
                 return_test=False, fixed_volume_space=False,
                 max_volume_space=[50, np.pi, 2],
                 min_volume_space=[0, -np.pi, -4]):
        """
        Args:
            point_cloud_dataset: HerculesFusion instance
            grid_size: [rho, phi, z] - voxel grid dimensions
            rotate_aug: Random rotation augmentation
            flip_aug: Random flip augmentation
            ignore_label: Label for invalid voxels
            return_test: Whether to return test information
            fixed_volume_space: If True, use fixed bounds; else use data-adaptive
            max_volume_space: [rho_max, phi_max, z_max] in cylindrical coords
            min_volume_space: [rho_min, phi_min, z_min] in cylindrical coords
        """
        self.point_cloud_dataset = point_cloud_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = np.asarray(max_volume_space)
        self.min_volume_space = np.asarray(min_volume_space)

    def __len__(self):
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        """
        Voxelize point clouds and return 11-tuple.
        """
        # Get raw data from HerculesFusion
        (mono_left, mono_right, mono_rear, radar_image_2d,
         lidar_xyz, radar_xyz, pose) = self.point_cloud_dataset[index]

        # ============================================================
        # Voxelize LiDAR
        # ============================================================
        lidar_voxel_pos, lidar_grid_ind, lidar_fea = self.pointclouddata(lidar_xyz)

        # ============================================================
        # Voxelize Radar
        # ============================================================
        radar_voxel_pos, radar_grid_ind, radar_fea = self.pointclouddata(radar_xyz)

        # ============================================================
        # Assemble 11-tuple
        # ============================================================
        data_tuple = (
            lidar_voxel_pos,    # [0] - voxel positions (left LiDAR)
            lidar_grid_ind,     # [1] - grid indices (left LiDAR)
            lidar_fea,          # [2] - features (left LiDAR)
            radar_voxel_pos,    # [3] - voxel positions (right/Radar)
            radar_grid_ind,     # [4] - grid indices (right/Radar)
            radar_fea,          # [5] - features (right/Radar)
            mono_left,          # [6] - left camera
            mono_right,         # [7] - right camera (zeros)
            mono_rear,          # [8] - rear camera (zeros)
            radar_image_2d,     # [9] - 2D radar image (zeros)
            pose                # [10] - 6DoF pose label
        )

        return data_tuple

    def pointclouddata(self, xyz):
        """
        Voxelize point cloud in cylindrical coordinates.

        Args:
            xyz: (N, 3) point cloud in Cartesian coordinates

        Returns:
            voxel_position: (H, W, D) voxel center positions
            grid_ind: (N, 3) grid indices for each point
            return_fea: (N, 8) features for each point
                - Components: [dx, dy, dz, rho, phi, z, x, y]
                - where (dx, dy, dz) = point - voxel_center
                - (rho, phi, z) = cylindrical coords
                - (x, y) = Cartesian x, y
        """
        # Convert to cylindrical coordinates
        xyz_pol = cart2polar(xyz)

        # Determine volume bounds
        if self.fixed_volume_space:
            # Use pre-defined bounds
            max_bound = self.max_volume_space
            min_bound = self.min_volume_space
        else:
            # Use data-adaptive bounds (percentile-based)
            max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
            min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
            max_bound = np.max(xyz_pol[:, 1:], axis=0)
            min_bound = np.min(xyz_pol[:, 1:], axis=0)
            max_bound = np.concatenate(([max_bound_r], max_bound))
            min_bound = np.concatenate(([min_bound_r], min_bound))

        # Compute grid spacing
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any():
            print("[Warning] Zero interval in voxelization!")

        # Clip and quantize to grid
        xyz_pol_clipped = np.clip(xyz_pol, min_bound, max_bound)
        grid_ind = (np.floor(
            (xyz_pol_clipped - min_bound) / intervals
        )).astype(np.int32)

        # ============================================================
        # Generate voxel positions (grid centers in cylindrical coords)
        # ============================================================
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = (
            np.indices(self.grid_size) * intervals.reshape(dim_array) +
            min_bound.reshape(dim_array)
        )

        # Convert voxel positions back to Cartesian
        voxel_position = polar2cat(voxel_position)

        # ============================================================
        # Compute features
        # ============================================================
        voxel_centers = (
            (grid_ind.astype(np.float32) + 0.5) * intervals +
            min_bound
        )

        # Offset from voxel center in cylindrical coords
        return_xyz = xyz_pol - voxel_centers

        # Concatenate features: [offset_in_cyl, cyl_coords, cartesian_xy]
        # This gives: [drho, dphi, dz, rho, phi, z, x, y]
        return_fea = np.concatenate((
            return_xyz,           # [drho, dphi, dz]
            xyz_pol,              # [rho, phi, z]
            xyz[:, :2]            # [x, y]
        ), axis=1)

        data_tuple = (voxel_position, grid_ind, return_fea)
        return data_tuple


if __name__ == '__main__':
    # Quick test
    from data.hercules_fusion import HerculesFusion

    pc_dataset = HerculesFusion(
        data_root='/data/drj/HeRCULES/',
        sequence_name='Library',
        split='train'
    )

    cyl_dataset = hercules_cylinder_dataset(
        pc_dataset,
        grid_size=[480, 360, 32],
        fixed_volume_space=False
    )

    print(f'Cylinder dataset size: {len(cyl_dataset)}')

    # Load one sample
    sample = cyl_dataset[0]
    print(f'Sample 11-tuple:')
    for i, item in enumerate(sample):
        if isinstance(item, np.ndarray):
            print(f'  [{i}] shape={item.shape}, dtype={item.dtype}')
        else:
            print(f'  [{i}] {type(item).__name__}')
