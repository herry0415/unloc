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
from tools.utils import quaternion_angular_error, qexp, load_state_dict

from torch.utils.tensorboard import SummaryWriter
from utils.metric_util import per_class_iu, fast_hist_crop
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from FusionModel import Fusionmodel
from utils.load_save_util import load_checkpoint
from thop import profile
writepath = "/home/ibrahim/Desktop/run7"
import warnings
warnings.filterwarnings("ignore")
class AtLocCriterion(nn.Module):
    def __init__(self, t_loss_fn=nn.MSELoss(), q_loss_fn=nn.MSELoss(), sax=0.0, saq=0.0, learn_beta=False):
        super(AtLocCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, trans, rot, rotgt, transgt):
        loss = torch.exp(-self.sax) * self.t_loss_fn(trans, transgt) + self.sax + \
               torch.exp(-self.saq) * self.q_loss_fn(rot, rotgt) + self.saq
        # loss =  self.t_loss_fn(pred[:, 3:6], targ[:, 3:6]) + self.sax + \
        #        150 * self.q_loss_fn(pred[:, 0:3], targ[:, 0:3]) + self.saq
        return loss

class AtLocCriterion2(nn.Module):
    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, learn_beta=False):
        super(AtLocCriterion2, self).__init__()
        # nn.L1Loss()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, trans, rot, rotgt, transgt):
        loss = torch.exp(-self.sax) * self.t_loss_fn(trans, transgt) + self.sax + \
               torch.exp(-self.saq) * self.q_loss_fn(rot, rotgt) + self.saq
        # loss =  self.t_loss_fn(pred[:, 3:6], targ[:, 3:6]) + self.sax + \
        #        150 * self.q_loss_fn(pred[:, 0:3], targ[:, 0:3]) + self.saq
        return loss

t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
q_criterion = quaternion_angular_error

t_loss = nn.L1Loss().cuda()
r_loss = nn.L1Loss().cuda()
def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']
    grid_size = model_config['output_shape']
    model_load_path = train_hypers['model_load_path']
    print(model_load_path)
    model_save_path = train_hypers['model_save_path']

    writer = SummaryWriter(log_dir=writepath)
    my_model = Fusionmodel()

    # print(my_model)
    #
    # my_model = model_builder.build(model_config)
    if os.path.exists('/home/ibrahim/Desktop/FusionLocalizationProject33/LOCALIZATIONMODELS/BestLIRModel_save.pt'):
        print(model_load_path)
        my_model = load_checkpoint('/home/ibrahim/Desktop/FusionLocalizationProject33/LOCALIZATIONMODELS/BestLIRModel_save.pt', my_model)

    my_model.to(pytorch_device)
    # val_loss = AtLocCriterion().to(pytorch_device)


    R1, R2, R3, R4, R5, R6, T1, T2, T3, T4, T5, T6 = 0, 0, 0,0,0,0,0,0,0,0,0,0
    R12, R345, R136, R236, R123456, R246, R12345, R126, R3456 = 0,0,0,0,0,0,0,0,0
    T12, T345, T136, T236, T123456, T246, T12345, T126, T3456 = 0,0,0,0,0,0,0,0,0



    # optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    posmean = np.array([5735484.8334842, 620018.8443548, 108.8114717])
    posstd = np.array([402.7248388, 260.6031783, 2.2196876])
    posmean = torch.from_numpy(posmean)
    posstd = torch.from_numpy(posstd)

    my_model.eval()
    with torch.no_grad():
        totalVal_loss = 0
        for i_iter_val, data in enumerate(val_dataset_loader):
            val_grid_tenl = [torch.from_numpy(i).to(pytorch_device) for i in data[1]]
            val_pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in data[2]]
            val_pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in data[5]]
            val_grid_tenr = [torch.from_numpy(i).to(pytorch_device) for i in data[4]]
            monoleft = torch.from_numpy(data[6])
            monoright = torch.from_numpy(data[7])
            monorear = torch.from_numpy(data[8])
            radarimage = torch.from_numpy(data[9]).reshape(val_batch_size, 1, 512, 512)
            labels = torch.from_numpy(data[10])
            Transgt = (((labels[:, 3:6] * posstd)+ posmean)).to(pytorch_device)
            Rotgt = labels[:, 0:3].to(pytorch_device)
            trans1, rot1 = my_model([val_pt_fea_tenl, val_grid_tenl, val_batch_size])
            trans2, rot2 = my_model([val_pt_fea_tenr, val_grid_tenr, val_batch_size])
            trans1 = ((trans1 * posstd.to(pytorch_device))+ posmean.to(pytorch_device))
            trans2 = ((trans2 * posstd.to(pytorch_device)) + posmean.to(pytorch_device))
            # print(rot1.shape,"rot1 shape")
            # print(Rotgt.shape,'rot1 gt')

            # rot1 = torch.squeeze(rot1)
            # trans1 = torch.squeeze(trans1)
            # rot2 = torch.squeeze(rot2)
            # trans2 =torch.squeeze(trans2)


            r1 = r_loss(rot1,(Rotgt))
            t1 = t_loss(trans1, Transgt)
            r2 = r_loss(rot2, Rotgt)
            t2 = t_loss(trans2, Transgt)
            #
            # loss1 = val_loss(trans1, rot1, Rotgt.to(pytorch_device), Transgt.to(pytorch_device))
            # loss2 = val_loss(trans2, rot2, Rotgt.to(pytorch_device), Transgt.to(pytorch_device))
            # forward + backward + optimize for image data
            trans3, rot3 = my_model([monoleft.to(pytorch_device)])
            trans4, rot4 = my_model([monoright.to(pytorch_device)])
            trans5, rot5 = my_model([monorear.to(pytorch_device)])
            trans3 = ((trans3 * posstd.to(pytorch_device)) + posmean.to(pytorch_device))
            trans4 = ((trans4 * posstd.to(pytorch_device)) + posmean.to(pytorch_device))
            trans5 = ((trans5 * posstd.to(pytorch_device)) + posmean.to(pytorch_device))

            # rot3 = torch.squeeze(rot3)
            # trans3 = torch.squeeze(trans3)
            # rot4 = torch.squeeze(rot4)
            # trans4 = torch.squeeze(trans4)
            #
            # rot5 = torch.squeeze(rot5)
            # trans5 = torch.squeeze(trans5)

            r3 = r_loss(rot3, Rotgt)
            t3 = t_loss(trans3, Transgt)
            r4 = r_loss(rot4, Rotgt)
            t4 = t_loss(trans4, Transgt)
            r5 = r_loss(rot5, Rotgt)
            t5 = t_loss(trans5, Transgt)

            # loss3 = val_loss(trans3, rot3, Rotgt.to(pytorch_device), Transgt.to(pytorch_device))
            # loss4 = val_loss(trans4, rot4, Rotgt.to(pytorch_device), Transgt.to(pytorch_device))
            # loss5 = val_loss(trans5, rot5, Rotgt.to(pytorch_device), Transgt.to(pytorch_device))

            # forward + backward + optimize for radar data
            trans6, rot6 = my_model([radarimage.to(pytorch_device)])
            trans6 = ((trans6 * posstd.to(pytorch_device)) + posmean.to(pytorch_device))
            # rot6 = torch.squeeze(rot6)
            # trans6 = torch.squeeze(trans6)
            #
            # loss6 = val_loss(trans6, rot6, Rotgt.to(pytorch_device), Transgt.to(pytorch_device))
            # loss = ((loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/6)

            r6 = r_loss(rot6, Rotgt)
            t6 = t_loss(trans6, Transgt)

            R1 = R1 +r1
            R2 = R2 + r2
            R3 = R3 + r3
            R4 = R4 + r4
            R5 = R5 + r5
            R6 = R6 + r6
            T1 = T1 + t1
            T2 = T2 + t2
            T3 = T3 + t3
            T4 = T4 + t4
            T5 = T5 + t5
            T6 = T6 + t6

            # r12 = ((r1 + r2) / 2)
            if r1 >=r2:
                r12 = r1
            else:
                r12 = r2

            R12 = (R12+r12)

            # t12 = ((t1 + t2) / 2)
            if t1 >=t2:
                t12 = t2
            else:
                t12 = t2

            T12 = (T12 + t12)

            writer.add_scalar('Rotation Loss for Left and Right LiDAR', r12, i_iter_val)
            writer.add_scalar('Translational Loss for Left and Right LiDAR', t12, i_iter_val)

            # r345 = ((r3 + r4 + r5) / 3)
            r345 = torch.min(torch.FloatTensor([r3, r4, r5]))
            R345 = (R345 +r345 )

            t345 = torch.min(torch.FloatTensor([t3, t4, t5]))
            T345 = (T345 + t345)

            writer.add_scalar('Rotation Loss for Left, Right and Rear Cameras', r345, i_iter_val)
            writer.add_scalar('Translational Loss for Left, Right and Rear Cameras', t345, i_iter_val)

            # r136 = ((r1 + r3 + r6) / 3)
            r136 = torch.min(torch.FloatTensor([r1, r3, r6]))
            R136 = (R136 + r136 )

            # t136 = ((t1 + t3 + t6) / 3)
            t136 = torch.min(torch.FloatTensor([t1, t3, t6]))
            T136 = (T136 + t136)

            writer.add_scalar('Rotation Loss for Left LiDAR, Left Camera and Radar', r136, i_iter_val)
            writer.add_scalar('Translational Loss for Left LiDAR, Left Camera and Radar', t136, i_iter_val)


            r236 = torch.min(torch.FloatTensor([r2, r3, r6]))
            R236 = (R236 + r236)

            # t236 = ((t2 + t3 + t6) / 3)
            t236 = torch.min(torch.FloatTensor([t2, t3, t6]))
            T236 = (T236 + t236)

            writer.add_scalar('Rotation Loss for Right LiDAR, Left Camera and Radar', r236, i_iter_val)
            writer.add_scalar('Translational Loss for Right LiDAR, Left Camera and Radar', t236, i_iter_val)

            # r123456 = ((r1 + r2 +r3 +r4 +r5 +r6)/6)
            r123456 = torch.min(torch.FloatTensor([r1, r2, r3,r4,r5, r6]))
            R123456 = R123456 + r123456

            t123456 = torch.min(torch.FloatTensor([t1, t2, t3, t4, t5, t6]))
            T123456 = T123456 + t123456

            writer.add_scalar('Rotation Loss for all sensors', r123456, i_iter_val)
            writer.add_scalar('Translational Loss for all sensors', t123456, i_iter_val)

            # r246 = ((r2 + r4 + r6) / 3)
            r246 = torch.min(torch.FloatTensor([r2 ,r4, r6]))
            R246 = ((R246 + r246 ))


            # t246 = ((t2 + t4 + t6) / 3)
            t246 = torch.min(torch.FloatTensor([t2, t4, t6]))
            T246 = (T246 + t246)


            writer.add_scalar('Rotation Loss for Right LiDAR, Right Camera and Radar', r246, i_iter_val)
            writer.add_scalar('Translational Loss for Right LiDAR, Right Camera and Radar', t246, i_iter_val)


            # r12345 = ((r1 + r2 + r3 + r4 +r5) / 5)
            r12345 = torch.min(torch.FloatTensor([r1, r2, r3, r4, r5]))
            R12345 = R12345 + r12345

            # t12345 = ((t2 + t1 +t3 + t4 + t5) / 5)
            t12345 = torch.min(torch.FloatTensor([t1, t2, t3, t4, t5]))
            T12345 = (T12345 + t12345)

            writer.add_scalar('Rotation Loss for Left and Right LiDAR & left,right and rear cameras', r12345, i_iter_val)
            writer.add_scalar('Translational Loss for Left and Right LiDAR & left,right and rear cameras', t12345, i_iter_val)

            # r3456 = ((r6 + r5 + r3 + r4) / 4)

            # t3456 = ((t4 + t5 + t3 + t6) / 4)
            t3456 = torch.min(torch.FloatTensor([t3, t4, t5, t6]))
            r3456 = torch.min(torch.FloatTensor([r3, r4, r5, r6]))
            R3456 = R3456 + r3456
            T3456 = (T3456 + t3456)

            writer.add_scalar('Rotation Loss for Left, right and rear cameras & Radar', r3456, i_iter_val)
            writer.add_scalar('Translational Loss for Left, right and rear cameras & Radar', t3456, i_iter_val)

            # r126 = ((r1 + r2 + r6 ) / 3)

            # t126 = ((t1 + t2 + t6) / 3)
            t126 = torch.min(torch.FloatTensor([t1, t2, t6]))

            r126 = torch.min(torch.FloatTensor([r1, r2,  r6]))

            R126 = R126 + r126

            T126 = (T126 + t126)

            writer.add_scalar('Rotation Loss for Left and Right LiDAR & Radar', r126, i_iter_val)
            writer.add_scalar('Translational Loss for Left and Right LiDAR & Radar', t126, i_iter_val)

            writer.add_scalar('Rotation Loss for Left LiDAR', r1, i_iter_val)
            writer.add_scalar('Rotation Loss for Right LiDAR', r2, i_iter_val)
            writer.add_scalar('Rotation Loss for Left Camera', r3, i_iter_val)
            writer.add_scalar('Rotation Loss for Right Camera', r4, i_iter_val)
            writer.add_scalar('Rotation Loss for Rear Camera', r5, i_iter_val)
            writer.add_scalar('Rotation Loss for Radar', r6, i_iter_val)
            writer.add_scalar('Translational Loss for Left LiDAR', t1, i_iter_val)
            writer.add_scalar('Translational Loss for Right LiDAR', t2, i_iter_val)
            writer.add_scalar('Translational Loss for Left Camera', t3, i_iter_val)
            writer.add_scalar('Translational Loss for Right Camera', t4, i_iter_val)
            writer.add_scalar('Translational Loss for Rear Camera', t5, i_iter_val)
            writer.add_scalar('Translational Loss for Radar', t6, i_iter_val)


            # print(loss_val,"loss_val")
            # totalVal_loss += loss.item()
            # writer.add_scalar('Validation total Loss', loss, i_iter_val)

            # if i_iter_val >= 100:
            #     break
        # print((totalVal_loss / i_iter_val), " Average Net Loss for all modalities")


        print((R126 / i_iter_val), " Average Rotation Loss for Left and Right LiDAR & Radar")
        print((T126 / i_iter_val), " Average Translational Loss for Left and Right LiDAR & Radar")

        print((R12345 / i_iter_val), " Average Rotation Loss for Left and Right LiDAR,  & left,right and rear cameras")
        print((T12345 / i_iter_val), " Average Translational Loss for Left and Right LiDAR, & left,right and rear cameras")


        print((R246 / i_iter_val), "Average Rotation Loss for Right LiDAR, Right Camera and Radar")
        print((T246 / i_iter_val), "Average Translational Loss for Right LiDAR, Right Camera and Radar")


        print((R123456 / i_iter_val), "Average Rotation Loss for all sensor")
        print((T123456 / i_iter_val), "Average Translational Loss for all sensors")


        print((R236 / i_iter_val), "Average Rotation Loss for Right LiDAR, Left Camera and Radar")
        print((T236 / i_iter_val), "Average Translational Loss for Right LiDAR, Left Camera and Radar")

        print((R136 / i_iter_val), "Average Rotation Loss for Left LiDAR, Left Camera and Radar")
        print((T136 / i_iter_val), "Average Translational Loss for Left LiDAR, Left Camera and Radar")

        print((R12 / i_iter_val), " Average Rotation Loss for Left and Right LiDAR")
        print((T12 / i_iter_val), "Average Translational Loss for Left and Right LiDAR")

        print((R345 / i_iter_val), "Average Rotation Loss for Left, Right and Rear Cameras")
        print((T345 / i_iter_val), "Average Translational Loss for Left, Right and Rear Cameras")

        print((R6 / i_iter_val), "Average Rotation Loss for Radar")
        print((T6 / i_iter_val), "Average Translational Loss for Radar")

        print((R5 / i_iter_val), "Average Rotation Loss for Rear Camera")
        print((T5 / i_iter_val), "Average Translational Loss for Rear Camera")

        print((R4 / i_iter_val), "Average Rotation Loss for Right Camera")
        print((T4 / i_iter_val), "Average Translational Loss for Right Camera")

        print((R3 / i_iter_val), "Average Rotation Loss for Left Camera")
        print((T3 / i_iter_val), "Average Translational Loss for Left Camera")

        print(( R2 / i_iter_val), "Average Rotation Loss for Left LiDAR")
        print((T2 / i_iter_val), "Average Translational Loss for Right LiDAR")

        print((R1 / i_iter_val), "Average Rotation Loss for Left LiDAR")
        print((T1 / i_iter_val), "Average Translational Loss for Left LiDAR")


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='/home/ibrahim/Desktop/FusionLocalizationProject33/config/semantickitti.yaml')
    args = parser.parse_args()
    print(' '.join(sys.argv))
    print(args)
    main(args)

