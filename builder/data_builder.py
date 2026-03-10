# -*- coding:utf-8 -*-
# author: Xinge
# @file: data_builder.py

import torch
from torchvision import transforms

from data.dataloaders import RobotCar
from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
from dataloader.pc_dataset import get_pc_model_class
from data.interpolate_poses import interpolate_vo_poses, interpolate_ins_poses
from dataloader.demo_dataset import DemoDataset, collate_fn_demo


def build(dataset_config,
          train_dataloader_config,
          val_dataloader_config,
          grid_size=[480, 360, 32],
          use_demo=False):
    """
    Build data loaders for training and validation.

    Args:
        dataset_config: Dataset configuration dict
        train_dataloader_config: Training dataloader config
        val_dataloader_config: Validation dataloader config
        grid_size: Grid size for voxelization (default: [480, 360, 32])
        use_demo: If True, use synthetic DemoDataset for pipeline testing (default: False)

    Returns:
        tuple: (train_dataset_loader, val_dataset_loader)
    """

    # ========== DEMO MODE ==========
    if use_demo:
        print("=" * 60)
        print("🎪 DEMO MODE ENABLED - Using synthetic data")
        print("=" * 60)

        train_batch_size = train_dataloader_config["batch_size"]
        val_batch_size = val_dataloader_config["batch_size"]

        # Create synthetic datasets
        train_dataset = DemoDataset(num_samples=20, batch_size=train_batch_size, seed=42)
        val_dataset = DemoDataset(num_samples=5, batch_size=val_batch_size, seed=123)

        # Create data loaders with demo collate function
        train_dataset_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=train_batch_size,
            collate_fn=collate_fn_demo,
            shuffle=train_dataloader_config.get("shuffle", True),
            num_workers=0  # Demo uses 0 workers
        )

        val_dataset_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            collate_fn=collate_fn_demo,
            shuffle=val_dataloader_config.get("shuffle", False),
            num_workers=0
        )

        print(f"✓ Train dataset: {len(train_dataset)} samples")
        print(f"✓ Val dataset: {len(val_dataset)} samples")
        print(f"✓ Train batch size: {train_batch_size}")
        print(f"✓ Val batch size: {val_batch_size}")
        print("=" * 60)

        return train_dataset_loader, val_dataset_loader

    # ========== REAL DATA MODE ==========
    data_path = '/media/ibrahim/bc7f6dde-b04e-4186-a53a-9499c7b9c1d7/LOCALIZATION/localizationProject5/LocalizationProject5/Dataset1/'

    train_imageset = train_dataloader_config["imageset"]
    val_imageset = val_dataloader_config["imageset"]
    train_ref = train_dataloader_config["return_ref"]
    val_ref = val_dataloader_config["return_ref"]
    val_path = val_dataloader_config["data_path"]
    train_batch_size = train_dataloader_config["batch_size"]

    nusc = None
    if "nusc" in dataset_config['pc_dataset_type']:
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)

    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
    ])

    path = "/media/ibrahim/21b7344f-7f8b-47fb-8de7-5199b164b720/RoboDataset"

    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())
    train_pt_dataset = RobotCar(train=True, data_path=path, undistort=True, transform=preprocess, target_transform=target_transform)

    val_pt_dataset = RobotCar(train=False, data_path=path, undistort=True, transform=preprocess, target_transform=target_transform)

    train_dataset = get_model_class(dataset_config['dataset_type'])(
        train_pt_dataset,
        grid_size=grid_size,
        flip_aug=True,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
        rotate_aug=True,
        scale_aug=True,
        transform_aug=True
    )

    val_dataset = get_model_class(dataset_config['dataset_type'])(
        val_pt_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
    )

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_batch_size,
                                                       collate_fn=collate_fn_BEV,
                                                       shuffle=train_dataloader_config["shuffle"],
                                                       num_workers=train_dataloader_config["num_workers"])
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn_BEV,
                                                     shuffle=val_dataloader_config["shuffle"],
                                                     num_workers=val_dataloader_config["num_workers"])

    return train_dataset_loader, val_dataset_loader



# # -*- coding:utf-8 -*-
# # author: Xinge
# # @file: data_builder.py
#
# import torch
# from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
# from dataloader.pc_dataset import get_pc_model_class
#
#
# def build(dataset_config,
#           train_dataloader_config,
#           val_dataloader_config,
#           grid_size=[480, 360, 32]):
#     data_path = train_dataloader_config["data_path"]
#
#     train_imageset = train_dataloader_config["imageset"]
#     val_imageset = val_dataloader_config["imageset"]
#     train_ref = train_dataloader_config["return_ref"]
#     val_ref = val_dataloader_config["return_ref"]
#     val_path =val_dataloader_config["data_path"]
#
#     label_mapping = dataset_config["label_mapping"]
#
#     SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])
#
#     nusc=None
#     if "nusc" in dataset_config['pc_dataset_type']:
#         from nuscenes import NuScenes
#         nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)
#
#     train_pt_dataset = SemKITTI(imageset=train_imageset)
#     val_pt_dataset = SemKITTI(val_path, imageset=val_imageset,
#                               return_ref=val_ref, label_mapping=label_mapping)
#
#     train_dataset = get_model_class(dataset_config['dataset_type'])(
#         train_pt_dataset,
#         grid_size=grid_size,
#         flip_aug=True,
#         fixed_volume_space=dataset_config['fixed_volume_space'],
#         max_volume_space=dataset_config['max_volume_space'],
#         min_volume_space=dataset_config['min_volume_space'],
#         ignore_label=dataset_config["ignore_label"],
#         rotate_aug=True,
#         scale_aug=True,
#         transform_aug=True
#     )
#
#     val_dataset = get_model_class(dataset_config['dataset_type'])(
#         val_pt_dataset,
#         grid_size=grid_size,
#         fixed_volume_space=dataset_config['fixed_volume_space'],
#         max_volume_space=dataset_config['max_volume_space'],
#         min_volume_space=dataset_config['min_volume_space'],
#         ignore_label=dataset_config["ignore_label"],
#     )
#
#     train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                                        batch_size=train_dataloader_config["batch_size"],
#                                                        collate_fn=collate_fn_BEV,
#                                                        shuffle=train_dataloader_config["shuffle"],
#                                                        num_workers=train_dataloader_config["num_workers"])
#     val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                                      batch_size=val_dataloader_config["batch_size"],
#                                                      collate_fn=collate_fn_BEV,
#                                                      shuffle=val_dataloader_config["shuffle"],
#                                                      num_workers=val_dataloader_config["num_workers"])
#
#     return train_dataset_loader, val_dataset_loader
