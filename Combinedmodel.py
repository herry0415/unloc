# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py
from thop import profile, clever_format
from torch import nn
from torchstat import stat
from torch.profiler import profile, record_function, ProfilerActivity
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import tensorboard
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint
from thop import profile

import warnings

class Regressionlayer(nn.Module):
    def __init__(self):
        super(Regressionlayer, self).__init__()

        self.regress_fc_pose = nn.Linear(1024, 1024)
        self.regress_fc_pose1r = nn.Linear(1024, 512)
        self.regress_fc_pose2r = nn.Linear(512, 256)
        self.regress_fc_pose3r = nn.Linear(256, 256)
        self.regress_fc_wpqr = nn.Linear(256, 3)

        self.regress_fc_pose1t = nn.Linear(1024, 512)
        self.regress_fc_pose2t = nn.Linear(512, 256)
        self.regress_fc_pose3t = nn.Linear(256, 256)
        self.regress_fc_xyz = nn.Linear(256, 3)


        # self.hidden1 = torch.nn.Linear(144, 64)
        # self.hidden2 = torch.nn.Linear(64, 32)
        # self.output = torch.nn.Linear(32, 6)
        nn.init.xavier_uniform_(self.regress_fc_pose.weight)
        nn.init.zeros_(self.regress_fc_pose.bias)

        nn.init.xavier_uniform_(self.regress_fc_pose1r.weight)
        nn.init.zeros_(self.regress_fc_pose1r.bias)

        nn.init.xavier_uniform_(self.regress_fc_pose2r.weight)
        nn.init.zeros_(self.regress_fc_pose2r.bias)

        nn.init.xavier_uniform_(self.regress_fc_pose3r.weight)
        nn.init.zeros_(self.regress_fc_pose3r.bias)

        nn.init.xavier_uniform_(self.regress_fc_pose1t.weight)
        nn.init.zeros_(self.regress_fc_pose1t.bias)

        nn.init.xavier_uniform_(self.regress_fc_pose2t.weight)
        nn.init.zeros_(self.regress_fc_pose2t.bias)

        nn.init.xavier_uniform_(self.regress_fc_pose3t.weight)
        nn.init.zeros_(self.regress_fc_pose3t.bias)

        nn.init.xavier_uniform_(self.regress_fc_xyz.weight)
        nn.init.zeros_(self.regress_fc_xyz.bias)

        nn.init.xavier_uniform_(self.regress_fc_wpqr.weight)
        nn.init.zeros_(self.regress_fc_wpqr.bias)
        self.drop1 = torch.nn.Dropout(0.2)
        # self.drop2 = torch.nn.Dropout(0.1)
        # self.drop3 = torch.nn.Dropout(0.1)
        # self.drop4 = torch.nn.Dropout(0.1)
        # self.drop5 = torch.nn.Dropout(0.1)
        # self.drop6 = torch.nn.Dropout(0.1)
        # self.drop7 = torch.nn.Dropout(0.1)
        self.Relu1 = nn.LeakyReLU(0.2)
        self.Relu2 = nn.LeakyReLU(0.2)
        self.Relu3 = nn.LeakyReLU(0.2)
        self.Relu4 = nn.LeakyReLU(0.2)
        self.Relu5 = nn.LeakyReLU(0.2)
        self.Relu6 = nn.LeakyReLU(0.2)
        self.Relu7 = nn.LeakyReLU(0.2)

        # self.drop2 = torch.nn.Dropout(0.1)

    def forward(self, x):
        x3 = self.drop1(self.Relu1(self.regress_fc_pose(x)))
        xr = self.Relu2(self.regress_fc_pose1r(x3))
        xr = self.Relu3(self.regress_fc_pose2r(xr))
        xr = self.Relu4(self.regress_fc_pose3r(xr))

        xt = self.Relu5(self.regress_fc_pose1t(x3))
        xt = self.Relu6(self.regress_fc_pose2t(xt))
        xt = self.Relu7(self.regress_fc_pose3t(xt))

        xyz = self.regress_fc_xyz(xt)
        wpqr = self.regress_fc_wpqr(xr)
        wpqr = F.normalize(wpqr, p=2, dim=1)

        return xyz, wpqr




class localizationmodel(nn.Module):
    def __init__(self):
        super(localizationmodel, self).__init__()

        parser = argparse.ArgumentParser(description='')
        parser.add_argument('-y', '--config_path',
                            default='/home/ibrahim/Desktop/VoxelbasedLocalizationProject33/config/semantickitti.yaml')
        args = parser.parse_args()
        print(' '.join(sys.argv))
        print(args)
        config_path = args.config_path
        configs = load_config_data(config_path)
        dataset_config = configs['dataset_params']
        model_config = configs['model_params']
        train_hypers = configs['train_params']
        grid_size = model_config['output_shape']
        model_load_path = train_hypers['model_load_path']
        print(model_load_path)
        self.my_model = model_builder.build(model_config)
        if os.path.exists(model_load_path):
            print(model_load_path)
            self.my_model = load_checkpoint(model_load_path, self.my_model)

        # torch.nn.Sequential(*(list(self.my_model.children())[:2]))

        self.Reglay= Regressionlayer()

        # self.my_model.fc = None
        print(self.my_model)
        # self.my_model.fc = Regressionlayer()


    def forward(self, voxel_features, coors, batch_size):

        # xyz,wpqr = self.my_model(voxel_features, coors, batch_size)
        _, out = self.my_model(voxel_features, coors, batch_size)
        xyz, rot =self.Reglay(out)

        return xyz, rot,out
if __name__ == '__main__':
    # Training settings
    localizationmodel()