import h5py
import torch
import struct, sys
import open3d as o3d
import os
import numpy as np
import pickle
import os.path as osp
import random
import math
from copy import deepcopy
from data.augmentor import Augmentor #todo  数据增强
import transforms3d.quaternions as txq
from torch.utils import data
#todo utils 2 util
from util.pose_util import process_poses, poses_to_matrices, filter_overflow_ts, qlog, ds_pc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Hercules(data.Dataset):
    def __init__(self, 
                 data_path,
                 split='train',
                 real=False,
                 valid=False,
                 vo_lib='stereo',
                 num_grid_x=1,
                 num_grid_y=3,
                 block_num=1, 
                 augment=False,
                 ):
        #todo 传入参数不一致
        self.dataset_root = data_path
        sequence_name='Library'
        self.sequence_name = sequence_name #['Library', 'Mountain', 'Sports']

        self.augment = augment #todo 数据增强
        self.data_dir = os.path.join(self.dataset_root, sequence_name)
        
        train = True if split == 'train' else False
        seqs = self._get_sequences(sequence_name, train) #todo train -> split
    
        ps = {}
        ts = {}
        pcs_all = [] #todo 多了一个这个
        vo_stats = {}
        self.pcs = []

        for seq in seqs:
            seq_dir = os.path.join(self.data_dir, seq)

            # h5_path = os.path.join(seq_dir, 'radar_poses.h5')
            # pose_file_path = os.path.join(seq_dir, 'PR_GT/newContinental_gt.txt')
            pose_file_path = os.path.join(seq_dir, 'PR_GT/Aeva_gt.txt')
            ts_raw = np.loadtxt(pose_file_path, dtype=np.int64, usecols=0) # float读取数字丢精度
            ts[seq] = ts_raw
            
            pose_file = np.loadtxt(pose_file_path) #保证pose值不变
            p = poses_to_matrices(pose_file) # (n,4,4) #毫米波雷达坐标系
            ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))     #  (n, 12)
            
            # write to h5 file
            # print('write interpolate pose to ' + h5_path)
            # h5_file = h5py.File(h5_path, 'w')
            # h5_file.create_dataset('valid_timestamps', data=np.asarray(ts[seq], dtype=np.int64))
            # h5_file.create_dataset('poses', data=ps[seq])
            
            # else:
            #     print("load " + seq + ' pose from ' + h5_path)
            #     h5_file = h5py.File(h5_path, 'r')
            #     ts[seq] = h5_file['valid_timestamps'][...]
            #     ps[seq] = h5_file['poses'][...]
            #     print(f'pose len {len(ts[seq])}')

            if real:
                vo_stats_filename = osp.join(seq_dir, '{:s}_vo_stats.pkl'.format(vo_lib))
                with open(vo_stats_filename, 'r') as f:
                    vo_stats[seq] = pickle.load(f)
            else:
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
            
            #todo 【带分割标签的】先存到 pcs_all 里面
            # pcs_all.extend([osp.join(seq_dir, 'LiDAR/SPVNAS_np8Aeva_plane_segmented', '{:d}.bin'.format(t)) for t in ts[seq]])
            # vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
            pcs_all.extend(os.path.join(seq_dir, 'LiDAR/np8Aeva', str(t) + '.bin') for t in ts[seq])

        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = os.path.join(self.data_dir, self.sequence_name + '_lidar'+'_pose_stats.txt')
        
        if split=='train': #Todo 从split赋值
            self.mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # (3,)
            self.std_t = np.std(poses[:, [3, 7, 11]], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((self.mean_t, self.std_t)), fmt='%8.7f')
            print(f'saving pose stats in {pose_stats_filename}')
        else:
            self.mean_t, self.std_t = np.loadtxt(pose_stats_filename)
        
        # convert the pose to translation + log quaternion, align, normalize
            
        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))

        #todo 多了一下几个参数
        self.poses_max = np.empty((0, 2))
        self.poses_min = np.empty((0, 2))
        poses_all = np.empty((0, 6))
        rots_all = np.empty((0, 3, 3))
        self.augmentor = Augmentor()
        #todo 新增pose_max_min_filename
        pose_max_min_filename = os.path.join(self.data_dir, self.sequence_name + '_lidar'+ '_pose_max_min.txt')
        
        for seq in seqs:
            pss, rotation, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=self.mean_t, std_t=self.std_t,
                                                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                            align_s=vo_stats[seq]['s'])
            #todo 有变化
            poses_all = np.vstack((poses_all, pss))
            self.poses_max = np.vstack((self.poses_max, pss_max))
            self.poses_min = np.vstack((self.poses_min, pss_min))
            rots_all = np.vstack((rots_all, rotation))
            
        # self.voxel_size = voxel_size #todo 不用voxel_size
        # todo RALoc的处理措施  mean_t 2 self.mean_t
        if split == 'train':
            self.poses_max = np.max(self.poses_max, axis=0) + self.mean_t[:2]
            self.poses_min = np.min(self.poses_min, axis=0) + self.mean_t[:2]
            block_size = list((np.array(list(self.poses_max)) - np.array(list(self.poses_min))) / block_num)
            np.savetxt(pose_max_min_filename, np.vstack((self.poses_max, self.poses_min)), fmt='%8.7f')
        else:
            self.poses_max, self.poses_min = np.loadtxt(pose_max_min_filename)
            block_size = list((np.array(list(self.poses_max)) - np.array(list(self.poses_min))) / block_num)
        poses_all_real = poses_all[:, :2] + self.mean_t[:2] - self.poses_min
        # divide the area into subregions
        if block_num != 1:
            for i in range(len(poses_all)):
                if (int((poses_all_real[i, 0]) / block_size[0]) == 0 and int(poses_all_real[i, 1] / block_size[1]) == 0) or (int((poses_all_real[i, 0]) / block_size[0]) == 0 and int(poses_all_real[i, 1] / block_size[1]) == 1):
                # if int((poses_all_real[i, 0]) / block_size[0]) == num_grid_x and int(poses_all_real[i, 1] / block_size[1]) == num_grid_y:
                    self.poses = np.vstack((self.poses, poses_all[i]))
                    self.pcs.append(pcs_all[i])
                    self.rots = np.vstack((self.rots, rots_all[i].reshape(1, 3, 3)))
        else:
            self.poses = poses_all
            self.pcs = pcs_all
            self.rots = rots_all

        self.split = split
        self.augment = augment

        if self.augment:
            print("=============use data augment=============")

        #todo 以上都是新增的部分
        if split == 'train':
            print("train data num:" + str(len(self.poses)))
        else:
            print("valid data num:" + str(len(self.poses)))
            
    def _get_sequences(self, sequence_name, train):
        mapping = {
            'Library': (['Library_01_Day','Library_02_Night'], ['Library_03_Day']),
            'Mountain': (['Mountain_01_Day','Mountain_02_Night'], ['Mountain_03_Day']),
            'Sports': (['Complex_01_Day','Complex_02_Night'], ['Complex_03_Day'])
        }
        return mapping[sequence_name][0] if train else mapping[sequence_name][1]
    
    def __getitem__(self, index):
        scan_path = self.pcs[index]
        # # pts, extra,_ = bin_to_pcd(scan_path, sensor_type='Aeva')
        # # scan = np.hstack([pts, extra])[:,:3] #xyz baseline
        # # todo 原来SGLoc的处理
        # scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 8)[:,:3]
        # scan = np.ascontiguousarray(scan)

        #todo RALoc的处理        
        ptcld =  np.fromfile(scan_path, dtype=np.float32).reshape(-1, 8)  # (N, 4) #todo 
        xyz = ptcld[:, :3]  # (N, 3)
        # dsxyz = xyz[xyz[:,2]>=-2]        
        pose = self.poses[index]  # (6,)
        rot = self.rots[index]

        # ground truth
        gt = (rot @ xyz.transpose()).transpose() + pose[:3].reshape(1, 3) #todo 转移坐标系
    
        # labels = np.concatenate((xyz, gt, mask), axis=1)
        labels = np.concatenate((xyz, gt), axis=1) #todo 
        # xyz,r,t = self.augmentor.doAugmentation(xyz) # todo 不用数据增强
        data_dict = {}
        data_dict['xyz'] = xyz
        data_dict['ds_xyz'] = ds_pc(xyz,4096)
        data_dict['labels'] = labels
        data_dict['pose'] = pose
        data_dict['rot'] = rot

        return data_dict
            
        # # todo 原来SGLoc的处理
        # pose = self.poses[index]  # (6,)
        # rot = self.rots[index]
        # # ground truth
        # scan_gt = (rot @ scan.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)
        # scan_gt_s8 = np.concatenate((scan, scan_gt), axis=1) #! Nx6 (x1 y1 z1 x2 y2 z2)
        
        # coords, feats= ME.utils.sparse_quantize(
        #     coordinates=scan,
        #     features=scan,
        #     quantization_size=self.voxel_size)

        # coords_s8, feats_s8 = ME.utils.sparse_quantize(
        #     coordinates=scan,
        #     features=scan_gt_s8,
        #     quantization_size=self.voxel_size*8)

        # return (coords, feats, coords_s8, feats_s8, rot, pose)

    def __len__(self):
        return len(self.poses)