
import os
import sys

import numpy as np
from torch.utils import data
import yaml
import pickle
from sklearn.cluster import DBSCAN
import open3d as o3d
from plyfile import PlyData, PlyElement
REGISTERED_PC_DATASET_CLASSES = {}
import torch





Xm = 0.1078
Ym = 0.8817
Zm = 0.6816

Xs = 12.2875
Ys = 12.0352
Zs = 2.4561
Xmean = np.array([Xm, Ym, Zm])
Xstandard = np.array([Xs, Ys, Zs])
def register_dataset(cls, name=None):
    global REGISTERED_PC_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
    REGISTERED_PC_DATASET_CLASSES[name] = cls
    return cls


def get_pc_model_class(name):
    global REGISTERED_PC_DATASET_CLASSES
    assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
    return REGISTERED_PC_DATASET_CLASSES[name]

@register_dataset
class SemKITTI_demo(data.Dataset):
    def __init__(self, data_path, imageset='demo',
                 return_ref=True, label_mapping="semantic-kitti.yaml", demo_label_path=None):
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.return_ref = return_ref

        self.im_idx = []
        self.im_idx += absoluteFilePaths(data_path)
        self.label_idx = []
        if self.imageset == 'val':
            print(demo_label_path)
            self.label_idx += absoluteFilePaths(demo_label_path)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'demo':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        elif self.imageset == 'val':
            annotated_data = np.fromfile(self.label_idx[index], dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple


splits = {
    "train": [1, 2, 0, 3, 4, 5, 6, 7, 9, 10],
    "val": [8],
    "test": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
}




@register_dataset
class SemKITTI_sk(data.Dataset):
    def __init__( self, data_path, imageset='train',
                 return_ref=False, label_mapping="semantic-kitti.yaml", nusc=None ):

        self.uniform = False
        self.return_ref = return_ref

        if imageset == 'train':
            self.path = data_path + "Training"
        else:
            self.path = data_path + "Test"
        list = []
        GT = []
        self.count = 0
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith("GT.txt"):
                    looppath = os.path.join(root, file)
                    GTfile = np.loadtxt(looppath, skiprows=1)
                    SHAPE = np.shape(GTfile)
                    for j in range(SHAPE[0]):
                        frame = root + "/" + "frame." + str(int(GTfile[j][0])) + ".ply"
                        list.append(frame)
                        GT.append(GTfile[j][1:7])
        self.im_idx = list
        self.GroundTruth = GT
        print('The size of %s data is %s' % (imageset, len(self.im_idx)))
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        pcd = o3d.io.read_point_cloud(self.im_idx[index])
        points = np.array(pcd.points)
        points = ((points - Xmean )/ Xstandard)
        gt = self.GroundTruth[index]
        data_tuple = (points, gt)

        return data_tuple


@register_dataset
class SemKITTI_nusc(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="nuscenes.yaml", nusc=None):
        self.return_ref = return_ref

        with open(imageset, 'rb') as f:
            data = pickle.load(f)

        with open(label_mapping, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']
        self.nusc_infos = data['infos']
        self.data_path = data_path
        self.nusc = nusc

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.nusc_infos)

    def __getitem__(self, index):
        info = self.nusc_infos[index]
        lidar_path = info['lidar_path'][16:]
        lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
        lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
                                                self.nusc.get('lidarseg', lidar_sd_token)['filename'])

        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])

        data_tuple = (points[:, :3], points_label.astype(np.uint8))
        if self.return_ref:
            data_tuple += (points[:, 3],)
        return data_tuple


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)


def SemKITTI2train_single(label):
    remove_ind = label == 0
    label -= 1
    label[remove_ind] = 255
    return label

from os.path import join
@register_dataset
class SemKITTI_sk_multiscan(data.Dataset):
    def __init__(self, data_path, imageset='train',return_ref=False, label_mapping="semantic-kitti-multiscan.yaml"):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.data_path = data_path
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        multiscan = 2 # additional two frames are fused with target-frame. Hence, 3 point clouds in total
        self.multiscan = multiscan
        self.im_idx = []

        self.calibrations = []
        self.times = []
        self.poses = []

        self.load_calib_poses()

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def load_calib_poses(self):
        """
        load calib poses and times.
        """

        ###########
        # Load data
        ###########

        self.calibrations = []
        self.times = []
        self.poses = []

        for seq in range(0, 22):
            seq_folder = join(self.data_path, str(seq).zfill(2))

            # Read Calib
            self.calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))

            # Read times
            self.times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))

            # Read poses
            poses_f64 = self.parse_poses(join(seq_folder, 'poses.txt'), self.calibrations[-1])
            self.poses.append([pose.astype(np.float32) for pose in poses_f64])

    def parse_calibration(self, filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

    def parse_poses(self, filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def fuse_multi_scan(self, points, pose0, pose):

        # pose = poses[0][idx]

        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        # new_points = hpoints.dot(pose.T)
        new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)

        new_points = new_points[:, :3]
        new_coords = new_points - pose0[:3, 3]
        # new_coords = new_coords.dot(pose0[:3, :3])
        new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
        new_coords = np.hstack((new_coords, points[:, 3:]))

        return new_coords

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        origin_len = len(raw_data)
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.int32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            # annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        number_idx = int(self.im_idx[index][-10:-4])
        dir_idx = int(self.im_idx[index][-22:-20])

        pose0 = self.poses[dir_idx][number_idx]

        if number_idx - self.multiscan >= 0:

            for fuse_idx in range(self.multiscan):
                plus_idx = fuse_idx + 1

                pose = self.poses[dir_idx][number_idx - plus_idx]

                newpath2 = self.im_idx[index][:-10] + str(number_idx - plus_idx).zfill(6) + self.im_idx[index][-4:]
                raw_data2 = np.fromfile(newpath2, dtype=np.float32).reshape((-1, 4))

                if self.imageset == 'test':
                    annotated_data2 = np.expand_dims(np.zeros_like(raw_data2[:, 0], dtype=int), axis=1)
                else:
                    annotated_data2 = np.fromfile(newpath2.replace('velodyne', 'labels')[:-3] + 'label',
                                                  dtype=np.int32).reshape((-1, 1))
                    annotated_data2 = annotated_data2 & 0xFFFF  # delete high 16 digits binary

                raw_data2 = self.fuse_multi_scan(raw_data2, pose0, pose)

                if len(raw_data2) != 0:
                    raw_data = np.concatenate((raw_data, raw_data2), 0)
                    annotated_data = np.concatenate((annotated_data, annotated_data2), 0)

        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))

        if self.return_ref:
            data_tuple += (raw_data[:, 3], origin_len) # origin_len is used to indicate the length of target-scan


        return data_tuple


# load Semantic KITTI class info

def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name


def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]

    return nuScenes_label_name




if __name__ == '__main__':
    import argparse
    from pathlib import Path

    import torch
    import torch.distributed as dist
    import argparse

    import utils

    import torch
    from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
    from dataloader.pc_dataset import get_pc_model_class

    from config.config import load_config_data

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='/home/ibrahim/Desktop/VoxelbasedLocalization/config/semantickitti.yaml')
    args = parser.parse_args()

    config_path = args.config_path


    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']

    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']
    data_path = train_dataloader_config["data_path"]

    train_imageset = train_dataloader_config["imageset"]
    val_imageset = val_dataloader_config["imageset"]
    train_ref = train_dataloader_config["return_ref"]
    val_ref = val_dataloader_config["return_ref"]

    label_mapping = dataset_config["label_mapping"]
    data_path = '/media/ibrahim/bc7f6dde-b04e-4186-a53a-9499c7b9c1d7/LOCALIZATION/localizationProject5/LocalizationProject5/Dataset1/'


    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])

    train_pt_dataset = SemKITTI(data_path, imageset=train_imageset,
                                return_ref=train_ref, label_mapping=label_mapping, nusc=None)


    # train_sampler = utils.dist_utils.TrainingSampler(train_pt_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_pt_dataset,
        batch_size=1,
        num_workers=8,
        drop_last=True,
        shuffle=False,
        pin_memory=True,


    )
    # val_loader = torch.utils.data.DataLoader(
    #     dataset=SemanticKitti(
    #         args.semantic_kitti_dir / "dataset/sequences", "val",
    #     ),
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=4,
    #     drop_last=False,
    # )

    print(len(train_loader))

    count = 0
    mean1 = torch.zeros(6, dtype=torch.float64)
    std1 = torch.zeros(6, dtype=torch.float64)
    GT = []
    xm = 0
    ym = 0
    zm = 0
    xsd = 0
    ysd = 0
    zsd = 0

    for step, items in enumerate(train_loader):
        pointcloud = items[0]
        print(pointcloud.shape)
        X = pointcloud[:,  :, 0]
        Y = pointcloud[:,  :, 1]
        Z = pointcloud[:,  :, 2]
        print(X.shape,"X shape")

        m1 = torch.mean(torch.squeeze(X))
        m2 = torch.mean(torch.squeeze(Y))
        m3 = torch.mean(torch.squeeze(Z))

        xm = xm + m1
        ym = ym + m2
        zm = zm + m3

        s1 = torch.std(torch.squeeze(X))
        s2 = torch.std(torch.squeeze(Y))
        s3 = torch.std(torch.squeeze(Z))

        xsd = xsd + s1
        ysd = ysd + s2
        zsd = zsd + s3
        count = count + 1
        print(count)


    print("training data means summation")
    print(xm)
    print(ym)
    print(zm)

    print("training data total mean")

    print(xm / count)
    print(ym / count)
    print(zm / count)

    print("Training data standard deviation summation")

    print(xsd)
    print(ysd)
    print(zsd)
    print("Training data standard deviation Total")

    print(xsd / count)
    print(ysd / count)
    print(zsd / count)



#
#
#
#
#
# # -*- coding:utf-8 -*-
# # author: Xinge
# # @file: pc_dataset.py
#
# import os
# import sys
#
# import numpy as np
# from torch.utils import data
# import yaml
# import pickle
# from sklearn.cluster import DBSCAN
# import open3d as o3d
# from plyfile import PlyData, PlyElement
# REGISTERED_PC_DATASET_CLASSES = {}
#
#
#
#
#
# def register_dataset(cls, name=None):
#     global REGISTERED_PC_DATASET_CLASSES
#     if name is None:
#         name = cls.__name__
#     assert name not in REGISTERED_PC_DATASET_CLASSES, f"exist class: {REGISTERED_PC_DATASET_CLASSES}"
#     REGISTERED_PC_DATASET_CLASSES[name] = cls
#     return cls
#
#
# def get_pc_model_class(name):
#     global REGISTERED_PC_DATASET_CLASSES
#     assert name in REGISTERED_PC_DATASET_CLASSES, f"available class: {REGISTERED_PC_DATASET_CLASSES}"
#     return REGISTERED_PC_DATASET_CLASSES[name]
#
# @register_dataset
# class SemKITTI_demo(data.Dataset):
#     def __init__(self, data_path, imageset='demo',
#                  return_ref=True, label_mapping="semantic-kitti.yaml", demo_label_path=None):
#         with open(label_mapping, 'r') as stream:
#             semkittiyaml = yaml.safe_load(stream)
#         self.learning_map = semkittiyaml['learning_map']
#         self.imageset = imageset
#         self.return_ref = return_ref
#
#         self.im_idx = []
#         self.im_idx += absoluteFilePaths(data_path)
#         self.label_idx = []
#         if self.imageset == 'val':
#             print(demo_label_path)
#             self.label_idx += absoluteFilePaths(demo_label_path)
#
#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.im_idx)
#
#     def __getitem__(self, index):
#         raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
#         if self.imageset == 'demo':
#             annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
#         elif self.imageset == 'val':
#             annotated_data = np.fromfile(self.label_idx[index], dtype=np.uint32).reshape((-1, 1))
#             annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
#             annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
#
#         data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
#         if self.return_ref:
#             data_tuple += (raw_data[:, 3],)
#         return data_tuple
#
#
# splits = {
#     "train": [1, 2, 0, 3, 4, 5, 6, 7, 9, 10],
#     "val": [8],
#     "test": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
# }
#
#
#
#
# @register_dataset
# class SemKITTI_sk(data.Dataset):
#     def __init__(self, data_path, imageset='train',
#                  return_ref=False, label_mapping="/media/ibrahim/v/OurApproachPCUrban/config/label_mapping/semantic-kitti.yaml"):
#         self.return_ref = return_ref
#         with open(label_mapping, 'r') as stream:
#             semkittiyaml = yaml.safe_load(stream)
#         self.learning_map = semkittiyaml['learning_map']
#         self.imageset = imageset
#         if imageset == 'train':
#             split = semkittiyaml['split']['train']
#         elif imageset == 'val':
#             split = semkittiyaml['split']['valid']
#         elif imageset == 'test':
#             split = semkittiyaml['split']['test']
#         else:
#             raise Exception('Split must be train/val/test')
#
#         self.seqs = splits[self.imageset]
#         print(data_path)
#
#
#         self.sweeps = np.loadtxt(data_path, dtype=np.str)
#         list = []
#
#         for j in range(len(self.sweeps)):
#             labelpath = self.sweeps[j].replace('.csv', '.ply')
#             list.append(labelpath)
#
#
#
#         self.im_idx = list
#
#
#
#
#
#
#         # for seq in self.seqs:
#         #     seq_str = f"{seq:0>2}"
#         #     seq_path = "/media/ibrahim/Research/SlotAttention/dataset/sequences/" + seq_str +  "/velodyne"
#         #     for sweep in seq_path.iterdir():
#         #         self.sweeps.append((seq_str, sweep.stem))
#
#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.im_idx)
#
#     def Clustering(self, points, labels, distance, arg):
#         clustering = DBSCAN(eps=distance, min_samples=1).fit(points)
#         clusters = clustering.labels_
#         for j in np.unique(clusters):
#             index = np.nonzero(clusters == j)
#             index = index[0]
#             npts = index.shape
#             # count = np.random.randint(0, 1000, 1)
#             if npts[0] >= 10:
#                 pts = points[index, :]
#                 lbs = labels[index]
#                 arr = (pts, lbs)
#                 arg.append(arr)
#
#     def PTcolor(self,labels, pointcloud, index):
#         map_inv = {
#             0: 0,  # "unlabeled", and others ignored
#             1: 10,  # "car"
#             2: 11,  # "bicycle"
#             3: 15,  # "motorcycle"
#             4: 18,  # "truck"
#             5: 20,  # "other-vehicle"
#             6: 30,  # "person"
#             7: 31,  # "bicyclist"
#             8: 32,  # "motorcyclist"
#             9: 40,  # "road"
#             10: 44,  # "parking"
#             11: 48,  # "sidewalk"
#             12: 49,  # "other-ground"
#             13: 50,  # "building"
#             14: 51,  # "fence"
#             15: 70,  # "vegetation"
#             16: 71,  # "trunk"
#             17: 72,  # "terrain"
#             18: 80,  # "pole"
#             19: 81  # "traffic-sign"
#         }
#
#         color_map = {0: [245, 150, 100], 1: [30, 30, 255], 2: [30, 30, 255], 3: [180, 30, 80], 4: [255, 0, 0],
#                      5: [30, 30, 255], 6: [30, 30, 255],
#                      7: [30, 30, 255], 8: [255, 0, 255], 9: [255, 150, 255], 10: [75, 0, 75], 11: [75, 0, 175],
#                      12: [0, 200, 255], 13: [50, 120, 255],
#                      14: [0, 175, 0], 15: [0, 60, 135], 16: [80, 240, 150], 17: [30, 30, 255], 18: [30, 30, 255],
#                      19: [90, 30, 150]}
#
#         Color = []
#         print(labels.shape)
#         labels = np.squeeze(np.array(labels))
#
#         for i in range(np.shape(labels)[0]):
#             color = color_map[labels[i]]
#             Color.append(color)
#
#         pcd2 = o3d.geometry.PointCloud()
#         print(np.array(Color))
#         Color = np.array(Color)
#
#         pcd2.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3])
#         pcd2.colors = o3d.utility.Vector3dVector(Color.astype(np.float) / 255.0)
#         # count = np.random.randint(1, 100, 1)
#
#         file = "/home/ibrahim/Desktop/test/" + "ColorOrg" + str(index) + "one" + ".ply"
#         o3d.io.write_point_cloud(file, pcd2)
#
#     def PTcolor2(self,labels, pointcloud, index):
#         map_inv = {
#             0: 0,  # "unlabeled", and others ignored
#             1: 10,  # "car"
#             2: 11,  # "bicycle"
#             3: 15,  # "motorcycle"
#             4: 18,  # "truck"
#             5: 20,  # "other-vehicle"
#             6: 30 , # "person"
#             7: 31 , # "bicyclist"
#             8: 32,  # "motorcyclist"
#             9: 40,  # "road"
#             10: 44 , # "parking"
#             11: 48,  # "sidewalk"
#             12: 49 , # "other-ground"
#             13: 50 , # "building"
#             14: 51,  # "fence"
#             15: 70,  # "vegetation"
#             16: 71,  # "trunk"
#             17: 72,  # "terrain"
#             18: 80,  # "pole"
#             19: 81  # "traffic-sign"
#             }
#
#
#         color_map = {20: [245, 150, 100], 1: [30, 30, 255], 2: [30, 30, 255], 3: [180, 30, 80], 4: [255, 0, 0],
#                      5: [30, 30, 255], 6: [30, 30, 255],
#                      7: [30, 30, 255], 8: [255, 0, 255], 9: [255, 150, 255], 10: [75, 0, 75], 11: [75, 0, 175],
#                      12: [0, 200, 255], 13: [50, 120, 255],
#                      14: [0, 175, 0], 15: [0, 60, 135], 16: [80, 240, 150], 17: [30, 30, 255], 18: [30, 30, 255],
#                      19: [90, 30, 150],0:[255,255,255]}
#
#         Color = []
#         # print(labels.shape)
#         labels = np.squeeze(np.array(labels))
#         for i in range(np.shape(labels)[0]):
#             color = color_map[labels[i]]
#             Color.append(color)
#
#         pcd2 = o3d.geometry.PointCloud()
#         # print(np.array(Color))
#         Color = np.array(Color)
#
#         pcd2.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3])
#         pcd2.colors = o3d.utility.Vector3dVector(Color.astype(np.float) / 255.0)
#         # count = np.random.randint(1, 100, 1)
#
#         file = "/home/ibrahim/Desktop/test/" + "ColorAug" + str(index) + "one" + ".ply"
#         o3d.io.write_point_cloud(file, pcd2)
#
#     def fun(self, data, label):
#         indrange = np.random.randint(1, 4400, 8)
#         Bcycle = []
#         Person = []
#         Mcycle = []
#         Bicyclist = []
#         Motorcyclist = []
#         Pole = []
#         Trafficsignal = []
#         # print(np.unique(label))
#
#         for i in range(8):
#             # print(self.im_idx[indrange[i]])
#             # points = np.fromfile(self.im_idx[indrange[i]], dtype=np.float32).reshape((-1, 4))
#             #
#             # points = points.reshape((-1, 4))
#
#             # labels = np.fromfile(self.im_idx[indrange[i]].replace('velodyne', 'labels')[:-3] + 'label',
#             #                              dtype=np.uint32).reshape((-1, 1))
#             # labels = labels & 0xFFFF  # delete high 16 digits binary
#             # labels = np.vectorize(self.learning_map.__getitem__)(labels)
#
#             # print(index)
#             # print(self.im_idx[index])
#             rawdata = PlyData.read(self.im_idx[indrange[i]])
#             values = rawdata.elements[0]
#             xyz = [values['x'], values['y'], values['z']]
#             points = np.array(xyz)
#             points_xyz = np.transpose((points)).astype(np.float32)
#
#
#
#             annotated_data = np.loadtxt(self.im_idx[indrange[i]].replace('.ply', 'Label.txt'),
#                                         dtype=np.uint32)
#             labels = annotated_data[:, 0].reshape((-1, 1))
#
#             labels = labels & 0xFFFF  # delete high 16 digits binary
#             labels = np.vectorize(self.learning_map.__getitem__)(labels)
#
#
#
#             person = np.nonzero(labels == 3)
#             mcycle = np.nonzero(labels == 5)
#             bcycle = np.nonzero(labels == 4)
#             bicyclist = np.nonzero(labels == 6)   # Rubbish bin
#             motorcyclist = np.nonzero(labels == 18)  # Road sign boards
#             pole = np.nonzero(labels == 15)
#             trafficsignal = np.nonzero(labels == 17)
#
#             if len(bcycle[0]) > 10:
#                 bcylepts = points_xyz[bcycle[0], :]
#                 bcyclabels = labels[bcycle[0]]
#                 self.Clustering(bcylepts, bcyclabels, 0.7, Bcycle)
#
#             if len(person[0]) > 10:
#                 personpts = points_xyz[person[0], :]
#                 personlabels = labels[person[0]]
#                 self.Clustering(personpts, personlabels, 0.7, Person)
#
#             if len(mcycle[0]) > 10:
#                 mcyclepts = points_xyz[mcycle[0], :]
#                 mcyclelabels = labels[mcycle[0]]
#                 self.Clustering(mcyclepts, mcyclelabels, 0.7, Mcycle)
#
#             if len(bicyclist[0]) > 10:
#                 bicyclistpts = points_xyz[bicyclist[0], :]
#                 bicyclistlabels = labels[bicyclist[0]]
#                 self.Clustering(bicyclistpts, bicyclistlabels, 0.7, Bicyclist)
#
#             if len(motorcyclist[0]) > 10:
#                 motorcyclistpts = points_xyz[motorcyclist[0], :]
#                 motorcyclistlabels = labels[motorcyclist[0]]
#                 self.Clustering(motorcyclistpts, motorcyclistlabels, 0.7, Motorcyclist)
#
#             if len(pole[0]) > 10:
#                 polepts = points_xyz[pole[0], :]
#                 polelabels = labels[pole[0]]
#                 self.Clustering(polepts, polelabels, 0.3, Pole)
#
#             if len(trafficsignal[0]) > 10:
#                 trafficsignalpts = points_xyz[trafficsignal[0], :]
#                 trafficsignallabels = labels[trafficsignal[0]]
#                 self.Clustering(trafficsignalpts, trafficsignallabels, 0.3, Trafficsignal)
#
#         RoadPts = np.nonzero(label == 11)
#         RoadPts = RoadPts[0]
#
#         ParkingPts = np.nonzero(label == 12)   # Road Divider
#         ParkingPts = ParkingPts[0]
#
#         SidewalkPts = np.nonzero(label == 19)
#         SidewalkPts = SidewalkPts[0]
#
#         GtkPts = np.nonzero(label == 0)
#         GtkPts = GtkPts[0]
#
#
#         if len(Person) > 0:
#
#             for j in range(6):
#                 for k in range(len(Person)):
#                     if len(RoadPts) > 0 and k%2 == 0:
#                         Selectedindex = np.random.choice(RoadPts)
#                         RoadPts = np.delete(RoadPts, np.argwhere(RoadPts == Selectedindex))
#
#                     elif len(SidewalkPts) > 0:
#                         Selectedindex = np.random.choice(SidewalkPts)
#                         SidewalkPts = np.delete(SidewalkPts, np.argwhere(SidewalkPts == Selectedindex))
#
#
#                     elif len(ParkingPts) > 0:
#                         Selectedindex = np.random.choice(ParkingPts)
#                         ParkingPts = np.delete(ParkingPts, np.argwhere(ParkingPts == Selectedindex))
#
#                     else:
#                         Selectedindex = np.random.choice(GtkPts)
#                         GtkPts = np.delete(GtkPts, np.argwhere(GtkPts == Selectedindex))
#
#
#                     point = data[Selectedindex, :]
#                     object = Person[k]
#                     BPts = object[0]
#                     McyLbs = object[1]
#                     X = BPts[:, 0]
#                     Y = BPts[:, 1]
#                     XY = np.c_[X, Y]
#                     XYZ = np.c_[XY, BPts[:, 2]]
#                     pcd22 = o3d.geometry.PointCloud()
#                     pcd22.points = o3d.utility.Vector3dVector(XYZ[:, 0:3])
#                     pcd22.translate((point[0], point[1], point[2]), relative=False)
#                     XYZ2 = np.array(pcd22.points)
#                     zmin = np.min(XYZ2[:, 2])
#                     disp = point[2] - zmin
#                     Z = XYZ2[:, 2] + disp
#                     XYZ = np.c_[XYZ2[:, 0:2], Z]
#                     data = np.concatenate((data, XYZ), axis=0)
#                     label = np.concatenate((label, McyLbs), axis=0)
#
#         if len(Bcycle) > 0:
#
#             for j in range(6):
#                 for k in range(len(Bcycle)):
#
#                     if len(RoadPts) > 0 and k %2 == 0:
#                         Selectedindex = np.random.choice(RoadPts)
#                         RoadPts = np.delete(RoadPts, np.argwhere(RoadPts == Selectedindex))
#
#                     elif len(SidewalkPts) > 0:
#                         Selectedindex = np.random.choice(SidewalkPts)
#                         SidewalkPts = np.delete(SidewalkPts, np.argwhere(SidewalkPts == Selectedindex))
#
#                     else:
#                         Selectedindex = np.random.choice(GtkPts)
#                         GtkPts = np.delete(GtkPts, np.argwhere(GtkPts == Selectedindex))
#
#                     point = data[Selectedindex, :]
#                     object = Bcycle[k]
#                     BPts = object[0]
#                     McyLbs = object[1]
#                     X = BPts[:, 0]
#                     Y = BPts[:, 1]
#                     XY = np.c_[X, Y]
#                     XYZ = np.c_[XY, BPts[:, 2]]
#                     pcd22 = o3d.geometry.PointCloud()
#                     pcd22.points = o3d.utility.Vector3dVector(XYZ[:, 0:3])
#                     pcd22.translate((point[0], point[1], point[2]), relative=False)
#                     XYZ2 = np.array(pcd22.points)
#                     zmin = np.min(XYZ2[:, 2])
#                     disp = point[2] - zmin
#                     Z = XYZ2[:, 2] + disp
#                     XYZ = np.c_[XYZ2[:, 0:2], Z]
#                     data = np.concatenate((data, XYZ), axis=0)
#                     label = np.concatenate((label, McyLbs), axis=0)
#
#         if len(Mcycle) > 0:
#
#             for j in range(7):
#                 for k in range(len(Mcycle)):
#
#                     if len(RoadPts) > 0:
#                         Selectedindex = np.random.choice(RoadPts)
#                         RoadPts = np.delete(RoadPts, np.argwhere(RoadPts == Selectedindex))
#
#                     elif len(SidewalkPts) > 0:
#                         Selectedindex = np.random.choice(SidewalkPts)
#                         SidewalkPts = np.delete(SidewalkPts, np.argwhere(SidewalkPts == Selectedindex))
#
#                     else:
#                         Selectedindex = np.random.choice(GtkPts)
#                         GtkPts = np.delete(GtkPts, np.argwhere(GtkPts == Selectedindex))
#                     point = data[Selectedindex, :]
#                     object = Mcycle[k]
#                     BPts = object[0]
#                     McyLbs = object[1]
#                     X = BPts[:, 0]
#                     Y = BPts[:, 1]
#                     XY = np.c_[X, Y]
#                     XYZ = np.c_[XY, BPts[:, 2]]
#                     pcd22 = o3d.geometry.PointCloud()
#                     pcd22.points = o3d.utility.Vector3dVector(XYZ[:, 0:3])
#                     pcd22.translate((point[0], point[1], point[2]), relative=False)
#                     XYZ2 = np.array(pcd22.points)
#                     zmin = np.min(XYZ2[:, 2])
#                     disp = point[2] - zmin
#                     Z = XYZ2[:, 2] + disp
#                     XYZ = np.c_[XYZ2[:, 0:2], Z]
#                     data = np.concatenate((data, XYZ), axis=0)
#                     label = np.concatenate((label, McyLbs), axis=0)
#
#         if len(Bicyclist) > 0:
#
#             for j in range(2):
#                 for k in range(len(Bicyclist)):
#                     if len(SidewalkPts)> 0:
#                         Selectedindex = np.random.choice(SidewalkPts)
#                         SidewalkPts = np.delete(SidewalkPts, np.argwhere(RoadPts == Selectedindex))
#
#                     elif len(ParkingPts) > 0:
#                         Selectedindex = np.random.choice(ParkingPts)
#                         ParkingPts = np.delete(ParkingPts, np.argwhere(ParkingPts == Selectedindex))
#
#                     else:
#                         Selectedindex = np.random.choice(GtkPts)
#                         GtkPts = np.delete(GtkPts, np.argwhere(GtkPts == Selectedindex))
#
#                     point = data[Selectedindex, :]
#                     object = Bicyclist[k]
#                     BPts = object[0]
#                     McyLbs = object[1]
#                     X = BPts[:, 0]
#                     Y = BPts[:, 1]
#                     XY = np.c_[X, Y]
#                     XYZ = np.c_[XY, BPts[:, 2]]
#                     pcd22 = o3d.geometry.PointCloud()
#                     pcd22.points = o3d.utility.Vector3dVector(XYZ[:, 0:3])
#                     pcd22.translate((point[0], point[1], point[2]), relative=False)
#                     XYZ2 = np.array(pcd22.points)
#                     zmin = np.min(XYZ2[:, 2])
#                     disp = point[2] - zmin
#                     Z = XYZ2[:, 2] + disp
#                     XYZ = np.c_[XYZ2[:, 0:2], Z]
#                     data = np.concatenate((data, XYZ), axis=0)
#                     label = np.concatenate((label, McyLbs), axis=0)
#
#         if len(Motorcyclist) > 0:
#
#             for j in range(4):
#                 for k in range(len(Motorcyclist)):
#                     if len(SidewalkPts) > 0:
#                         Selectedindex = np.random.choice(SidewalkPts)
#                         SidewalkPts = np.delete(SidewalkPts, np.argwhere(RoadPts == Selectedindex))
#
#                     elif len(ParkingPts) > 0:
#                         Selectedindex = np.random.choice(ParkingPts)
#                         ParkingPts = np.delete(ParkingPts, np.argwhere(ParkingPts == Selectedindex))
#
#                     else:
#                         Selectedindex = np.random.choice(GtkPts)
#                         GtkPts = np.delete(GtkPts, np.argwhere(GtkPts == Selectedindex))
#                     point = data[Selectedindex, :]
#                     object = Motorcyclist[k]
#                     BPts = object[0]
#                     McyLbs = object[1]
#                     X = BPts[:, 0]
#                     Y = BPts[:, 1]
#                     XY = np.c_[X, Y]
#                     XYZ = np.c_[XY, BPts[:, 2]]
#                     pcd22 = o3d.geometry.PointCloud()
#                     pcd22.points = o3d.utility.Vector3dVector(XYZ[:, 0:3])
#                     pcd22.translate((point[0], point[1], point[2]), relative=False)
#                     XYZ2 = np.array(pcd22.points)
#                     zmin = np.min(XYZ2[:, 2])
#                     disp = point[2] - zmin
#                     Z = XYZ2[:, 2] + disp
#                     XYZ = np.c_[XYZ2[:, 0:2], Z]
#                     data = np.concatenate((data, XYZ), axis=0)
#                     label = np.concatenate((label, McyLbs), axis=0)
#
#         if len(Pole) > 0:
#
#             for k in range(len(Pole)):
#
#                 if len(ParkingPts) > 0:
#                     Selectedindex = np.random.choice(ParkingPts)
#                     ParkingPts = np.delete(ParkingPts, np.argwhere(ParkingPts == Selectedindex))
#
#                 elif len(SidewalkPts) > 0:
#                     Selectedindex = np.random.choice(SidewalkPts)
#                     SidewalkPts = np.delete(SidewalkPts, np.argwhere(RoadPts == Selectedindex))
#                 else:
#                     Selectedindex = np.random.choice(GtkPts)
#                     GtkPts = np.delete(GtkPts, np.argwhere(GtkPts == Selectedindex))
#
#
#                 point = data[Selectedindex, :]
#                 object = Pole[k]
#                 BPts = object[0]
#                 McyLbs = object[1]
#
#                 X = BPts[:, 0]
#                 Y = BPts[:, 1]
#                 XY = np.c_[X, Y]
#
#                 XYZ = np.c_[XY, BPts[:, 2]]
#                 pcd22 = o3d.geometry.PointCloud()
#                 pcd22.points = o3d.utility.Vector3dVector(XYZ[:, 0:3])
#                 pcd22.translate((point[0], point[1], point[2]), relative=False)
#                 XYZ2 = np.array(pcd22.points)
#                 zmin = np.min(XYZ2[:, 2])
#                 disp = point[2] - zmin
#
#                 Z = XYZ2[:, 2] + disp
#                 XYZ = np.c_[XYZ2[:, 0:2], Z]
#
#                 data = np.concatenate((data, XYZ), axis=0)
#                 label = np.concatenate((label, McyLbs), axis=0)
#
#         if len(Trafficsignal) > 0:
#
#             for k in range(len(Trafficsignal)):
#
#                 if len(ParkingPts) > 0:
#                     Selectedindex = np.random.choice(ParkingPts)
#                     ParkingPts = np.delete(ParkingPts, np.argwhere(ParkingPts == Selectedindex))
#
#                 elif len(SidewalkPts) > 0:
#                     Selectedindex = np.random.choice(SidewalkPts)
#                     SidewalkPts = np.delete(SidewalkPts, np.argwhere(RoadPts == Selectedindex))
#                 else:
#                     Selectedindex = np.random.choice(GtkPts)
#                     GtkPts = np.delete(GtkPts, np.argwhere(GtkPts == Selectedindex))
#                 point = data[Selectedindex, :]
#                 object = Trafficsignal[k]
#                 BPts = object[0]
#                 McyLbs = object[1]
#                 X = BPts[:, 0]
#                 Y = BPts[:, 1]
#                 XY = np.c_[X, Y]
#                 XYZ = np.c_[XY, BPts[:, 2]]
#                 pcd22 = o3d.geometry.PointCloud()
#                 pcd22.points = o3d.utility.Vector3dVector(XYZ[:, 0:3])
#                 pcd22.translate((point[0], point[1], point[2]), relative=False)
#                 XYZ2 = np.array(pcd22.points)
#                 zmin = np.min(XYZ2[:, 2])
#                 disp = point[2] - zmin
#                 Z = XYZ2[:, 2] + disp
#                 XYZ = np.c_[XYZ2[:, 0:2], Z]
#                 data = np.concatenate((data, XYZ), axis=0)
#                 label = np.concatenate((label, McyLbs), axis=0)
#
#
#
#         return data, label
#
#     def __getitem__(self, index):
#         data = PlyData.read(self.im_idx[index])
#
#         values = data.elements[0]
#         xyz = [values['x'], values['y'], values['z']]
#         xyz = np.array(xyz)
#         raw_data = np.transpose((xyz)).astype(np.float32)
#
#         labelpath = self.im_idx[index].replace('.ply', 'Label.txt')
#         # print(labelpath)
#
#         annotated_data = np.loadtxt(self.im_idx[index].replace('.ply', 'Label.txt'),
#                                     dtype=np.uint32)
#         annotated_data = annotated_data[:, 0].reshape((-1, 1))
#
#         annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
#         annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
#
#         # self.PTcolor2(annotated_data, raw_data, index+1)
#
#         # data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
#
#         # self.PTcolor(annotated_data,raw_data,index)
#         # print(raw_data.shape,"Before")
#
#
#         if (index % 3 ) != 0:
#             if self.imageset == "train":
#                 raw_data,annotated_data = self.fun(raw_data,annotated_data.astype(np.uint8))
#
#         # self.PTcolor2(annotated_data, raw_data,index)
#
#         # print(raw_data.shape,"After")
#
#
#         data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
#         # if self.return_ref:
#         #     data_tuple += (raw_data[:, 3],)
#         return data_tuple
#
#
# @register_dataset
# class SemKITTI_nusc(data.Dataset):
#     def __init__(self, data_path, imageset='train',
#                  return_ref=False, label_mapping="nuscenes.yaml", nusc=None):
#         self.return_ref = return_ref
#
#         with open(imageset, 'rb') as f:
#             data = pickle.load(f)
#
#         with open(label_mapping, 'r') as stream:
#             nuscenesyaml = yaml.safe_load(stream)
#         self.learning_map = nuscenesyaml['learning_map']
#
#         self.nusc_infos = data['infos']
#         self.data_path = data_path
#         self.nusc = nusc
#
#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.nusc_infos)
#
#     def __getitem__(self, index):
#         info = self.nusc_infos[index]
#         lidar_path = info['lidar_path'][16:]
#         lidar_sd_token = self.nusc.get('sample', info['token'])['data']['LIDAR_TOP']
#         lidarseg_labels_filename = os.path.join(self.nusc.dataroot,
#                                                 self.nusc.get('lidarseg', lidar_sd_token)['filename'])
#
#         points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
#         points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
#         points = np.fromfile(os.path.join(self.data_path, lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
#
#         data_tuple = (points[:, :3], points_label.astype(np.uint8))
#         if self.return_ref:
#             data_tuple += (points[:, 3],)
#         return data_tuple
#
#
# def absoluteFilePaths(directory):
#     for dirpath, _, filenames in os.walk(directory):
#         filenames.sort()
#         for f in filenames:
#             yield os.path.abspath(os.path.join(dirpath, f))
#
#
# def SemKITTI2train(label):
#     if isinstance(label, list):
#         return [SemKITTI2train_single(a) for a in label]
#     else:
#         return SemKITTI2train_single(label)
#
#
# def SemKITTI2train_single(label):
#     remove_ind = label == 0
#     label -= 1
#     label[remove_ind] = 255
#     return label
#
# from os.path import join
# @register_dataset
# class SemKITTI_sk_multiscan(data.Dataset):
#     def __init__(self, data_path, imageset='train',return_ref=False, label_mapping="semantic-kitti-multiscan.yaml"):
#         self.return_ref = return_ref
#         with open(label_mapping, 'r') as stream:
#             semkittiyaml = yaml.safe_load(stream)
#         self.learning_map = semkittiyaml['learning_map']
#         self.imageset = imageset
#         self.data_path = data_path
#         if imageset == 'train':
#             split = semkittiyaml['split']['train']
#         elif imageset == 'val':
#             split = semkittiyaml['split']['valid']
#         elif imageset == 'test':
#             split = semkittiyaml['split']['test']
#         else:
#             raise Exception('Split must be train/val/test')
#
#         multiscan = 2 # additional two frames are fused with target-frame. Hence, 3 point clouds in total
#         self.multiscan = multiscan
#         self.im_idx = []
#
#         self.calibrations = []
#         self.times = []
#         self.poses = []
#
#         self.load_calib_poses()
#
#         for i_folder in split:
#             self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))
#
#
#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.im_idx)
#
#     def load_calib_poses(self):
#         """
#         load calib poses and times.
#         """
#
#         ###########
#         # Load data
#         ###########
#
#         self.calibrations = []
#         self.times = []
#         self.poses = []
#
#         for seq in range(0, 22):
#             seq_folder = join(self.data_path, str(seq).zfill(2))
#
#             # Read Calib
#             self.calibrations.append(self.parse_calibration(join(seq_folder, "calib.txt")))
#
#             # Read times
#             self.times.append(np.loadtxt(join(seq_folder, 'times.txt'), dtype=np.float32))
#
#             # Read poses
#             poses_f64 = self.parse_poses(join(seq_folder, 'poses.txt'), self.calibrations[-1])
#             self.poses.append([pose.astype(np.float32) for pose in poses_f64])
#
#     def parse_calibration(self, filename):
#         """ read calibration file with given filename
#
#             Returns
#             -------
#             dict
#                 Calibration matrices as 4x4 numpy arrays.
#         """
#         calib = {}
#
#         calib_file = open(filename)
#         for line in calib_file:
#             key, content = line.strip().split(":")
#             values = [float(v) for v in content.strip().split()]
#
#             pose = np.zeros((4, 4))
#             pose[0, 0:4] = values[0:4]
#             pose[1, 0:4] = values[4:8]
#             pose[2, 0:4] = values[8:12]
#             pose[3, 3] = 1.0
#
#             calib[key] = pose
#
#         calib_file.close()
#
#         return calib
#
#     def parse_poses(self, filename, calibration):
#         """ read poses file with per-scan poses from given filename
#
#             Returns
#             -------
#             list
#                 list of poses as 4x4 numpy arrays.
#         """
#         file = open(filename)
#
#         poses = []
#
#         Tr = calibration["Tr"]
#         Tr_inv = np.linalg.inv(Tr)
#
#         for line in file:
#             values = [float(v) for v in line.strip().split()]
#
#             pose = np.zeros((4, 4))
#             pose[0, 0:4] = values[0:4]
#             pose[1, 0:4] = values[4:8]
#             pose[2, 0:4] = values[8:12]
#             pose[3, 3] = 1.0
#
#             poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))
#
#         return poses
#
#     def fuse_multi_scan(self, points, pose0, pose):
#
#         # pose = poses[0][idx]
#
#         hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
#         # new_points = hpoints.dot(pose.T)
#         new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
#
#         new_points = new_points[:, :3]
#         new_coords = new_points - pose0[:3, 3]
#         # new_coords = new_coords.dot(pose0[:3, :3])
#         new_coords = np.sum(np.expand_dims(new_coords, 2) * pose0[:3, :3], axis=1)
#         new_coords = np.hstack((new_coords, points[:, 3:]))
#
#         return new_coords
#
#     def __getitem__(self, index):
#         raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
#         origin_len = len(raw_data)
#         if self.imageset == 'test':
#             annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
#         else:
#             annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
#                                          dtype=np.int32).reshape((-1, 1))
#             annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
#             # annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
#
#         number_idx = int(self.im_idx[index][-10:-4])
#         dir_idx = int(self.im_idx[index][-22:-20])
#
#         pose0 = self.poses[dir_idx][number_idx]
#
#         if number_idx - self.multiscan >= 0:
#
#             for fuse_idx in range(self.multiscan):
#                 plus_idx = fuse_idx + 1
#
#                 pose = self.poses[dir_idx][number_idx - plus_idx]
#
#                 newpath2 = self.im_idx[index][:-10] + str(number_idx - plus_idx).zfill(6) + self.im_idx[index][-4:]
#                 raw_data2 = np.fromfile(newpath2, dtype=np.float32).reshape((-1, 4))
#
#                 if self.imageset == 'test':
#                     annotated_data2 = np.expand_dims(np.zeros_like(raw_data2[:, 0], dtype=int), axis=1)
#                 else:
#                     annotated_data2 = np.fromfile(newpath2.replace('velodyne', 'labels')[:-3] + 'label',
#                                                   dtype=np.int32).reshape((-1, 1))
#                     annotated_data2 = annotated_data2 & 0xFFFF  # delete high 16 digits binary
#
#                 raw_data2 = self.fuse_multi_scan(raw_data2, pose0, pose)
#
#                 if len(raw_data2) != 0:
#                     raw_data = np.concatenate((raw_data, raw_data2), 0)
#                     annotated_data = np.concatenate((annotated_data, annotated_data2), 0)
#
#         annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
#
#         data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
#
#         if self.return_ref:
#             data_tuple += (raw_data[:, 3], origin_len) # origin_len is used to indicate the length of target-scan
#
#
#         return data_tuple
#
#
# # load Semantic KITTI class info
#
# def get_SemKITTI_label_name(label_mapping):
#     with open(label_mapping, 'r') as stream:
#         semkittiyaml = yaml.safe_load(stream)
#     SemKITTI_label_name = dict()
#     for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
#         SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]
#
#     return SemKITTI_label_name
#
#
# def get_nuScenes_label_name(label_mapping):
#     with open(label_mapping, 'r') as stream:
#         nuScenesyaml = yaml.safe_load(stream)
#     nuScenes_label_name = dict()
#     for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
#         val_ = nuScenesyaml['learning_map'][i]
#         nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]
#
#     return nuScenes_label_name
#
#
#
#
# if __name__ == '__main__':
#     import argparse
#     from pathlib import Path
#
#     import torch
#     import torch.distributed as dist
#     import argparse
#
#     import utils
#
#     import torch
#     from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
#     from dataloader.pc_dataset import get_pc_model_class
#
#     from config.config import load_config_data
#
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('-y', '--config_path', default='/home/ibrahim/Desktop/VoxelbasedLocalization/config/semantickitti.yaml')
#     args = parser.parse_args()
#
#     config_path = args.config_path
#
#
#     configs = load_config_data(config_path)
#
#     dataset_config = configs['dataset_params']
#
#     train_dataloader_config = configs['train_data_loader']
#     val_dataloader_config = configs['val_data_loader']
#
#     val_batch_size = val_dataloader_config['batch_size']
#     train_batch_size = train_dataloader_config['batch_size']
#     data_path = train_dataloader_config["data_path"]
#
#     train_imageset = train_dataloader_config["imageset"]
#     val_imageset = val_dataloader_config["imageset"]
#     train_ref = train_dataloader_config["return_ref"]
#     val_ref = val_dataloader_config["return_ref"]
#
#     label_mapping = dataset_config["label_mapping"]
#
#
#     SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])
#
#     train_pt_dataset = SemKITTI(data_path, imageset="train",
#                                 return_ref=train_ref, label_mapping=label_mapping)
#
#
#     # train_sampler = utils.dist_utils.TrainingSampler(train_pt_dataset)
#
#     train_loader = torch.utils.data.DataLoader(
#         train_pt_dataset,
#         batch_size=1,
#         num_workers=8,
#         drop_last=True,
#         shuffle=False,
#         pin_memory=True,
#
#
#     )
#     # val_loader = torch.utils.data.DataLoader(
#     #     dataset=SemanticKitti(
#     #         args.semantic_kitti_dir / "dataset/sequences", "val",
#     #     ),
#     #     batch_size=1,
#     #     shuffle=False,
#     #     num_workers=4,
#     #     drop_last=False,
#     # )
#
#     print(len(train_loader))
#
#
#
#     for step, items in enumerate(train_loader):
#         labels = items[0]
#         xyz = items[1]
#
#         print(labels.shape)
#         print(xyz.shape)
#
