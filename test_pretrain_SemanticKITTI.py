#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import errno
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint
import open3d as o3d

def PTcolor(labels, pointcloud, index):

    # color_map = {0: [245, 150, 100], 1: [245, 230, 100], 2: [150, 60, 30], 3: [180, 30, 80], 4: [255, 0, 0],
    #              5: [30, 30, 255], 6: [200, 40, 255],
    #              7: [90, 30, 150], 8: [255, 0, 255], 9: [255, 150, 255], 10: [75, 0, 75], 11: [75, 0, 175],
    #              12: [0, 200, 255], 13: [50, 120, 255],
    #              14: [0, 175, 0], 15: [0, 60, 135], 16: [80, 240, 150], 17: [150, 240, 255], 18: [0, 0, 255],
    #              255: [90, 30, 150]}

    # color_map = {0: [245, 150, 100], 1: [245, 230, 100], 2: [150, 60, 30], 3: [180, 30, 80], 4: [255, 0, 0],
    #              5: [30, 30, 255], 6: [200, 40, 255],
    #              7: [90, 30, 150], 8: [255, 0, 255], 9: [255, 150, 255], 10: [75, 0, 75], 11: [75, 0, 175],
    #              12: [0, 200, 255], 13: [50, 120, 255],
    #              14: [0, 175, 0], 15: [0, 60, 135], 16: [80, 240, 150], 17: [150, 240, 255], 18: [0, 0, 255],
    #              255: [90, 30, 150]}

    color_map = { 1: [245, 150, 100], 2: [255, 60, 30], 3: [180, 30, 80], 4: [255, 0, 0],
                 5: [30, 30, 255], 6: [200, 40, 255],
                 7: [90, 30, 150], 8: [255, 0, 255], 9: [235, 64, 52], 10: [75, 0, 75], 11: [75, 0, 175],
                 12: [0, 200, 255], 13: [50, 120, 255],
                 14: [0, 175, 0], 15: [0, 60, 135], 16: [80, 240, 150], 17: [150, 240, 255], 18: [0, 0, 255],
                 19: [90, 30, 150]}

    Color = []
    print(labels.shape)
    labels = np.squeeze(np.array(labels))
    for i in range(np.shape(labels)[0]):
        color = color_map[labels[i]]
        Color.append(color)

    pcd2 = o3d.geometry.PointCloud()
    # print(np.array(Color))
    Color = np.array(Color)

    pcd2.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3])
    pcd2.colors = o3d.utility.Vector3dVector(Color.astype(np.float) / 255.0)
    # count = np.random.randint(1, 100, 1)

    file = "/home/ibrahim/Desktop/test/" + "ColorOrg" + str(index) + "one" + ".ply"
    o3d.io.write_point_cloud(file, pcd2)

def PTcolor2(labels, pointcloud, index):
    map_inv = {
        0: 10,  # "car"
        1: 11,  # "bicycle"
        2: 15,  # "motorcycle"
        3: 18,  # "truck"
        4: 20,  # "other-vehicle"
        5: 30,  # "person"
        6: 31,  # "bicyclist"
        7: 32,  # "motorcyclist"
        8: 40,  # "road"
        9: 44,  # "parking"
        10: 48,  # "sidewalk"
        11: 49,  # "other-ground"
        12: 50,  # "building"
        13: 51,  # "fence"
        14: 70,  # "vegetation"
        15: 71,  # "trunk"
        16: 72,  # "terrain"
        17: 80,  # "pole"
        18: 81,  # "traffic-sign
        255: 0,
    }
    # color_map = {0: [245, 150, 100], 1: [245, 230, 100], 2: [150, 60, 30], 3: [180, 30, 80], 4: [255, 0, 0],
    #              5: [30, 30, 255], 6: [200, 40, 255],
    #              7: [90, 30, 150], 8: [255, 0, 255], 9: [255, 150, 255], 10: [75, 0, 75], 11: [75, 0, 175],
    #              12: [0, 200, 255], 13: [50, 120, 255],
    #              14: [0, 175, 0], 15: [0, 60, 135], 16: [80, 240, 150], 17: [150, 240, 255], 18: [0, 0, 255],
    #              255: [90, 30, 150]}

    # color_map = {0: [245, 150, 100], 1: [245, 230, 100], 2: [150, 60, 30], 3: [180, 30, 80], 4: [255, 0, 0],
    #              5: [30, 30, 255], 6: [200, 40, 255],
    #              7: [90, 30, 150], 8: [255, 0, 255], 9: [255, 150, 255], 10: [75, 0, 75], 11: [75, 0, 175],
    #              12: [0, 200, 255], 13: [50, 120, 255],
    #              14: [0, 175, 0], 15: [0, 60, 135], 16: [80, 240, 150], 17: [150, 240, 255], 18: [0, 0, 255],
    #              255: [90, 30, 150]}

    color_map = {0: [245, 150, 100], 1: [30, 30, 255], 2: [30, 30, 255], 3: [180, 30, 80], 4: [255, 0, 0],
                 5: [30, 30, 255], 6: [30, 30, 255],
                 7: [30, 30, 255], 8: [255, 0, 255], 9: [255, 150, 255], 10: [75, 0, 75], 11: [75, 0, 175],
                 12: [0, 200, 255], 13: [50, 120, 255],
                 14: [0, 175, 0], 15: [0, 60, 135], 16: [80, 240, 150], 17: [30, 30, 255], 18: [30, 30, 255],
                 255: [90, 30, 150]}


    Color = []
    print(labels.shape)
    labels = np.array(labels)
    for i in range(np.shape(labels)[0]):
        color = color_map[labels[i]]
        Color.append(color)

    pcd2 = o3d.geometry.PointCloud()
    print(np.array(Color))
    Color = np.array(Color)

    pcd2.points = o3d.utility.Vector3dVector(pointcloud[:, 0:3])
    pcd2.colors = o3d.utility.Vector3dVector(Color.astype(np.float) / 255.0)
    # count = np.random.randint(1, 100, 1)

    file = "/home/ibrahim/Desktop/test/" + "ColorAug" + str(index) + "one" + ".ply"
    o3d.io.write_point_cloud(file, pcd2)
#ignore weird np warning
import warnings
warnings.filterwarnings("ignore")

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count=np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label)+1)
    hist=hist[unique_label,:]
    hist=hist[:,unique_label]
    return hist

def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)

def SemKITTI2train_single(label):
    return label - 1 # uint8 trick

def train2SemKITTI(input_label):
    # delete 0 label (uses uint8 trick : 0 - 1 = 255 )
    return input_label + 1

def main(args):
    config_path = args.config_path
    pytorch_device = torch.device('cuda:0')

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']


    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]




    # prepare miou fun
    # unique_label=np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    # unique_label_str=[SemKITTI_label_name[x] for x in unique_label+1]

    # prepare model

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        print(model_load_path)
        my_model = load_checkpoint(model_load_path, my_model)



    # my_model.eval()

    my_model.to(pytorch_device)
    print(my_model)


    # prepare dataset
    from dataloader.pc_dataset import  get_pc_model_class
    from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV, collate_fn_BEV_test
    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])
    cylinder = get_model_class(dataset_config['dataset_type'])


    test_pt_dataset = SemKITTI(train_dataloader_config['data_path'], imageset='test', return_ref=True,  label_mapping = dataset_config["label_mapping"])
    test_dataset = cylinder(test_pt_dataset, grid_size=grid_size, ignore_label=0, fixed_volume_space=True,
                                     return_test=True)

    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=val_batch_size,
                                                      collate_fn=collate_fn_BEV_test,
                                                      shuffle=False,
                                                      num_workers=4)

    # for i_iter_test, (_, _, test_grid, _, test_pt_fea, test_index) in enumerate(test_dataset_loader):
    #
    #     print(test_index)



    print('*'*80)
    print('Generate predictions for test split')
    print('*'*80)
    pbar = tqdm(total=len(test_dataset_loader))
    output_path = "/media/ibrahim/21b7344f-7f8b-47fb-8de7-5199b164b720/KittiTestOutput"
    with torch.no_grad():
        for i_iter_test,(_,_,test_grid,_,test_pt_fea,test_index) in enumerate(test_dataset_loader):
            # print(test_grid[0].shape)
            # print(test_pt_fea[0].shape)
            # print(test_index)
            # predict

            # val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
            #                       val_pt_fea]
            # val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            # val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)
            points = np.array(test_grid[0])

            test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in test_pt_fea]
            # train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
            test_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in test_grid]


            #
            # test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in test_pt_fea]
            # test_grid_ten = [torch.from_numpy(i[:,:3]).to(pytorch_device) for i in test_grid]
            # print(test_grid_ten[0].shape,"Input shape")




            predict_labels = my_model(test_pt_fea_ten,test_grid_ten,val_batch_size)
            # print(predict_labels.shape,"Initial prediction")
            predict_labels = torch.argmax(predict_labels,1).type(torch.uint8)
            predict_labels = predict_labels.cpu().detach().numpy()
            # print(predict_labels.shape,"Middle Predicted Results")
            # write to label file
            for count,i_test_grid in enumerate(test_grid):
                test_pred_label = predict_labels[count,test_grid[count][:,0],test_grid[count][:,1],test_grid[count][:,2]]
                test_pred_label = train2SemKITTI(test_pred_label)
                test_pred_label = np.expand_dims(test_pred_label,axis=1)

                save_dir = test_pt_dataset.im_idx[test_index[count]]
                _,dir2 = save_dir.split('/sequences/',1)
                new_save_dir = output_path + '/sequences/' +dir2.replace('velodyne','predictions')[:-3]+'label'
                if not os.path.exists(os.path.dirname(new_save_dir)):
                    try:
                        os.makedirs(os.path.dirname(new_save_dir))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                test_pred_label = test_pred_label.astype(np.uint32)
                test_pred_label -=1
                test_pred_label.tofile(new_save_dir)
            # print(test_pred_label.shape,"Final Label")
            print(np.unique(test_pred_label))
            pbar.update(1)

            # PTcolor(test_pred_label,points,test_index)
    del test_grid,test_pt_fea,test_index
    pbar.close()
    print('Predicted test labels are saved in %s. Need to be shifted to original label format before submitting to the Competition website.' % output_path)
    print('Remapping script can be found in semantic-kitti-api.')

if __name__ == '__main__':
    # Testing settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
