import h5py
import torch
import struct, sys
import open3d as o3d
import os
import numpy as np
import pickle
from sklearn.neighbors import KDTree
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


def bin_to_pcd(bin_fn, sensor_type):
    pts, extra = [], []

    with open(bin_fn, 'rb') as f:
        if sensor_type == "Continental":
            rec = 29
            while (chunk := f.read(rec)):
                if len(chunk) < rec: break
                x, y, z, v, r = struct.unpack('<fffff',  chunk[:20])
                RCS           = struct.unpack('<B',      chunk[20:21])[0]
                az, el        = struct.unpack('<ff',     chunk[21:29])
                pts.append([x,y,z])
                extra.append([v, r, RCS, az, el])

        elif sensor_type == "ContinentalObject":
            rec = 20
            while (chunk := f.read(rec)):
                if len(chunk) < rec: break
                x, y, z, vx, vy = struct.unpack('<fffff', chunk)
                pts.append([x,y,z])
                extra.append([vx, vy])

        elif sensor_type == "Aeva":
            rec = 29
            while (chunk := f.read(rec)):
                if len(chunk) < rec: break
                x, y, z, refl, vel = struct.unpack('<fffff', chunk[:20])
                t_off   = struct.unpack('<I', chunk[20:24])[0]
                line_id = struct.unpack('<B', chunk[24:25])[0]
                inten   = struct.unpack('<f', chunk[25:29])[0]
                pts.append([x,y,z])
                extra.append([refl, vel, t_off, line_id, inten])

        else:
            raise ValueError("sensor_type must be Continental / ContinentalObject / Aeva")

    return np.asarray(pts, dtype=np.float32), np.asarray(extra), sensor_type


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

        self.dataset_root = data_path
        sequence_name='Library' #todo 
        self.sequence_name = sequence_name #['Library', 'Mountain', 'Sports']

        self.augment = augment #todo 数据增强
        self.data_dir = os.path.join(self.dataset_root, sequence_name)

        train = True if split == 'train' else False
         # 根据 sequence_name 和 train/val 设置序列
        seqs = self._get_sequences(sequence_name, train) #todo train -> split         # 根据 sequence_name 和 train/val 设置序列
    

        self.calib_lidar_to_radar = [
            0.9996720412467692, 0.02257917399331767, -0.0120826673916502, -1.4450608431262415,
            -0.022426134790843006, 0.9996684039844378, 0.01265506039297812, -0.008365884285909064,
            0.01236440163779473, -0.012379942527591536, 0.9998469175804628, 1.1050482497723382
            ]
        E = np.array(self.calib_lidar_to_radar).reshape(3,4)
        T_lr = np.eye(4)
        T_lr[:3,:4] = E  # LiDAR -> Radar

        # 求逆得到 Radar -> LiDAR
        self.T_rl = np.linalg.inv(T_lr)
        
        self.R_rl = self.T_rl[:3, :3]
        self.t_rl = self.T_rl[:3, 3]


        ps = {}
        ts = {}
        vo_stats = {}
        self.pcs = []

        # #! 初始加载文件
        # for seq in seqs:
        #     seq_dir = os.path.join(self.data_dir, seq)

        #     # h5_path = os.path.join(seq_dir, 'radar_poses.h5')
         
        #     # if not os.path.isfile(h5_path):
        #     #     print('interpolate ' + seq)
        #     pose_file_path = os.path.join(seq_dir, 'PR_GT/newContinental_gt.txt')
        #     ts_raw = np.loadtxt(pose_file_path, dtype=np.int64, usecols=0) # float读取数字丢精度
        #     ts[seq] = ts_raw
                
        #     pose_file = np.loadtxt(pose_file_path) #保证pose值不变
        #     p = poses_to_matrices(pose_file) # (n,4,4) #毫米波雷达坐标系
        #     ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))     #  (n, 12)
                
        #     #     # write to h5 file
        #     #     print('write interpolate pose to ' + h5_path)
        #     #     h5_file = h5py.File(h5_path, 'w')
        #     #     h5_file.create_dataset('valid_timestamps', data=np.asarray(ts[seq], dtype=np.int64))
        #     #     h5_file.create_dataset('poses', data=ps[seq])
            
        #     # else:
        #     #     print("load " + seq + ' pose from ' + h5_path)
        #     #     h5_file = h5py.File(h5_path, 'r')
        #     #     ts[seq] = h5_file['valid_timestamps'][...]
        #     #     ps[seq] = h5_file['poses'][...]
        #     #     print(f'pose len {len(ts[seq])}')
           
        #     # self.pcs.extend(os.path.join(seq_dir, 'LiDAR/np8Aeva', str(t) + '.bin') for t in ts[seq])
        #     self.pcs.extend(os.path.join(seq_dir, 'Radar/multi_frame_w7', str(t)+'_multi_w7' + '.bin') for t in ts[seq])
        #     vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}
        
        #! 选择与lidar一一对应的帧进行训练
        self.lidar_files = []
        self.radar_files = []
        for seq in seqs:
            seq_dir = os.path.join(self.data_dir, seq)         
           
            lidar_pose_file_path = os.path.join(seq_dir, 'PR_GT/Aeva_gt.txt')
        
            ts_raw = np.loadtxt(lidar_pose_file_path, dtype=np.int64, usecols=0)
            ts[seq] = ts_raw
            lidar_ts = ts_raw
            
            lidar_pose_file = np.loadtxt(lidar_pose_file_path)
            p = poses_to_matrices(lidar_pose_file) # (n,4,4) #激光雷达坐标系
            ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))     #  (n, 12)
            
            
            radar_file_path = os.path.join(seq_dir, 'PR_GT/newContinental_gt.txt')
            radar_ts= np.loadtxt(radar_file_path, dtype=np.int64, usecols=0)
            radar_pose_file = np.loadtxt(radar_file_path)
           

            lidar_xy = self.get_xy(lidar_pose_file)
            radar_xy = self.get_xy(radar_pose_file)
            
            #build kdtree to find match
            lidar_files = [os.path.join(seq_dir, 'LiDAR/np8Aeva', str(t) + '.bin') for t in lidar_ts]
            radar_files = [os.path.join(seq_dir, 'Radar/multi_frame_w7', str(t)+'_multi_w7' + '.bin') for t in radar_ts]
            
            tree_radar = KDTree(radar_xy)
            dists, idxs = tree_radar.query(lidar_xy, k=1)  # for each lidar -> nearest radar
            idxs = idxs.ravel()
            dists = dists.ravel()
                # filter by distance threshold
            for li, ridx in enumerate(idxs):
                dist = float(dists[li])
                # print(dist)

                lf = lidar_files[li]
                rf = radar_files[ridx]
                # check existence of files
                if os.path.exists(lf) and os.path.exists(rf):
                    self.lidar_files.append(lf)
                    self.radar_files.append(rf)
                else:
                    # 缺文件则略过并打印警告
                    if not os.path.exists(lf): print(f"Missing LIDAR file: {lf}")
                    if not os.path.exists(rf): print(f"Missing RADAR file: {rf}")

            if real:
                vo_stats_filename = osp.join(seq_dir, '{:s}_vo_stats.pkl'.format(vo_lib))
                with open(vo_stats_filename, 'r') as f:
                    vo_stats[seq] = pickle.load(f)
            else:
                vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

        pcs_all = self.radar_files  #todo 多了一个这个 -> self.radar_files 

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
        # self.augmentor = Augmentor()
        #todo 新增pose_max_min_filename
        pose_max_min_filename = os.path.join(self.data_dir, self.sequence_name + '_radar'+ '_pose_max_min.txt')


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
        print(f'self.radar_files length: {len(self.radar_files)}')
        print(f'self.poses length: {len(self.poses)}')
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
        # #! 原始的坐标系
        # scan_path = self.pcs[index]
        # # scan_path = self.radar_files[index]
        # # pts, extra,_ = bin_to_pcd(scan_path, sensor_type='Aeva')
        # # scan = np.hstack([pts, extra])[:,:3] #xyz baseline
        # scan = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 8)[:,:3]
        # scan = np.ascontiguousarray(scan)

        #! 把radar转换到lidar坐标系
        radar_scan_path = self.radar_files[index]
        radar_ptcld = np.fromfile(radar_scan_path, dtype=np.float32).reshape(-1,8)[:,:3]
        radar_new = (self.R_rl @ radar_ptcld.T).T + self.t_rl.reshape(1, 3)
        radar_scan = np.ascontiguousarray(radar_new)
        # scan = radar_scan
        xyz = radar_scan
        
        pose = self.poses[index]  # (6,)
        rot = self.rots[index]

        # ground truth
        gt = (rot @ xyz.transpose()).transpose() + pose[:3].reshape(1, 3) #todo 转移坐标系

        # scan_gt = (rot @ scan.transpose(1, 0)).transpose(1, 0) + pose[:3].reshape(1, 3)
        # scan_gt_s8 = np.concatenate((scan, scan_gt), axis=1)
        
        # labels = np.concatenate((xyz, gt, mask), axis=1)
        labels = np.concatenate((xyz, gt), axis=1) #todo 
        # xyz,r,t = self.augmentor.doAugmentation(xyz)
        data_dict = {}
        data_dict['xyz'] = xyz
        data_dict['ds_xyz'] = ds_pc(xyz,4096)
        data_dict['labels'] = labels
        data_dict['pose'] = pose
        data_dict['rot'] = rot

        return data_dict


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

    def get_xy(self, pose_file):
        pose = poses_to_matrices(pose_file) # (n,4,4)
        xy = pose[:,:2, 3]
        return xy
        