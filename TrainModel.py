from torch import nn
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
from torch.utils.tensorboard import SummaryWriter
from utils.metric_util import per_class_iu, fast_hist_crop
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from FusionModel import Fusionmodel
from utils.load_save_util import load_checkpoint
from thop import profile
writepath = "/home/ibrahim/Desktop/FusionLocalizationProject33/runs/"
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

        # rot1 = rot.reshape(rot.shape[0] * rot.shape[1])
        # predr = rotgt
        # predr = predr.reshape(predr.shape[0] * predr.shape[1])
        loss = torch.exp(-self.sax) * self.t_loss_fn(trans, transgt) + self.sax + \
               torch.exp(-self.saq) * self.q_loss_fn(rot, rotgt) + self.saq
        # loss =  self.t_loss_fn(pred[:, 3:6], targ[:, 3:6]) + self.sax + \
        #        150 * self.q_loss_fn(pred[:, 0:3], targ[:, 0:3]) + self.saq
        return loss

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
    # if os.path.exists(model_load_path):
    #     print(model_load_path)
    #     my_model = load_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)

    training_loss = AtLocCriterion(saq=-3, learn_beta=True).to(pytorch_device)
    val_loss = AtLocCriterion().to(pytorch_device)

    # Optimizer
    param_list = [{'params': my_model.parameters()}]
    param_list.append({'params': [training_loss.sax, training_loss.saq]})
    print('learn_beta')
    optimizer = torch.optim.Adam(param_list, lr=0.0001, weight_decay=0.0005)

    # optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)

    # training
    epoch = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']
    while epoch < train_hypers['max_num_epochs']:
        loss_listlidar = []
        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(10)
        lidarepoch_loss = 0
        previouslossi = 100
        # lr_scheduler.step(epoch)
        my_model.train()
        for i_iter, data in enumerate(train_dataset_loader):
            train_vox_tenl = [torch.from_numpy(i).to(pytorch_device) for i in data[1]]
            train_pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in data[2]]
            train_pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in data[5]]
            train_vox_tenr = [torch.from_numpy(i).to(pytorch_device) for i in data[4]]
            monoleft = torch.from_numpy(data[6])
            monoright = torch.from_numpy(data[7])
            monorear = torch.from_numpy(data[8])
            radarimage = torch.from_numpy(data[9]).reshape(train_batch_size,1,512,512)
            labels = torch.from_numpy(data[10])
            Transgt = (labels[:, 3:6])
            rotgt = labels[:, 0:3]
            # print(Transgt, "Transgtv")
            if i_iter >= 7000:
                break
            # forward + backward + optimize for lidar data
            trans1, rot1 = my_model([train_pt_fea_tenl, train_vox_tenl, train_batch_size])
            trans2, rot2 = my_model([train_pt_fea_tenr, train_vox_tenr, train_batch_size])
            loss1 = training_loss(trans1, rot1, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
            loss2 = training_loss(trans2, rot2, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
            # forward + backward + optimize for image data
            trans1, rot1 = my_model([monoleft.to(pytorch_device)])
            trans2, rot2 = my_model([monoright.to(pytorch_device)])
            trans3, rot3 = my_model([monorear.to(pytorch_device)])
            loss3 = training_loss(trans1, rot1, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
            loss4 = training_loss(trans2, rot2, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
            loss5 = training_loss(trans3, rot3, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
            trans1, rot1 = my_model([radarimage.to(pytorch_device)])
            loss6 = training_loss(trans1, rot1, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
            loss = loss1 + loss2 + loss3 +loss4 +loss5 + loss6
            loss.backward()
            optimizer.step()
            loss = loss.cpu()
            loss = loss.detach().numpy()
            loss_listlidar.append(loss)
            writer.add_scalar('Training Loss', loss, i_iter)
            # forward + backward + optimize for radar data
            # writer.add_scalar('Training Loss', loss, i_iter)
            if global_iter % 1000 == 0:
                if len(loss_listlidar) > 0:
                    print('epoch %d iter %5d, Training loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_listlidar)))
                else:
                    print('loss error')
            optimizer.zero_grad()
            pbar.update(1)
            torch.cuda.empty_cache()
            global_iter += 1
            if global_iter % check_iter == 0:
                if len(loss_listlidar) > 0:
                    print('epoch %d iter %5d, Lidarloss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_listlidar)))
                else:
                    print('loss error')

        print('epoch %d iter %5d, Training loss: %.3f\n' %
              (epoch, i_iter, np.mean(loss_listlidar)))
        pbar.close()
        epoch += 1
        writer.add_scalar('Training Loss per epoch', np.mean(loss_listlidar), epoch)
        del monoleft, monoright, monorear, radarimage
        del loss1, loss2, loss3, loss4, loss5, loss6, trans1, rot1, trans2, rot2, trans3, rot3
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
                Transgt = labels[:, 3:6]
                Rotgt = labels[:, 0:3]
                trans1, rot1= my_model([val_pt_fea_tenl, val_grid_tenl, val_batch_size])
                trans2, rot2 = my_model([val_pt_fea_tenr, val_grid_tenr, val_batch_size])

                loss1 = val_loss(trans1,rot1,Rotgt.to(pytorch_device), Transgt.to(pytorch_device))
                loss2 = val_loss(trans2, rot2, Rotgt.to(pytorch_device), Transgt.to(pytorch_device))

                # forward + backward + optimize for image data
                trans1, rot1 = my_model([monoleft.to(pytorch_device)])
                trans2, rot2 = my_model([monoright.to(pytorch_device)])
                trans3, rot3 = my_model([monorear.to(pytorch_device)])
                loss3 = training_loss(trans1, rot1, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
                loss4 = training_loss(trans2, rot2, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
                loss5 = training_loss(trans3, rot3, rotgt.to(pytorch_device), Transgt.to(pytorch_device))

                # forward + backward + optimize for radar data
                trans1, rot1 = my_model([radarimage.to(pytorch_device)])
                loss6 = training_loss(trans1, rot1, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

                # print(loss_val,"loss_val")
                totalVal_loss += loss.item()
                writer.add_scalar('Validation total Loss', loss, i_iter_val)

                # if i_iter_val >= 100:
                #     break
            print((totalVal_loss / i_iter_val), " Average Validation Loss")

        my_model.train()
        del val_grid_tenl, val_pt_fea_tenl, val_pt_fea_tenr, val_grid_tenr,monoleft,monoright,monorear,radarimage
        del loss1, loss2, loss3, loss4, loss5, loss6, trans1, rot1, trans2, rot2, trans3, rot3
        # save model if performance is improved
        if previouslossi > (totalVal_loss / i_iter_val):
            previouslossi = (totalVal_loss / i_iter_val)
            torch.save(my_model.state_dict(), "/home/ibrahim/Desktop/FusionLocalizationProject33/LOCALIZATIONMODELS/BestLIRModel_save.pt")


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='/home/ibrahim/Desktop/FusionLocalizationProject33/config/semantickitti.yaml')
    args = parser.parse_args()
    print(' '.join(sys.argv))
    print(args)
    main(args)


#
# from torch import nn
# from torchstat import stat
# from torch.profiler import profile, record_function, ProfilerActivity
# import os
# import time
# import argparse
# import sys
# import numpy as np
# import torch
# import torch.optim as optim
# from tqdm import tqdm
# import tensorboard
# from torch.utils.tensorboard import SummaryWriter
# from utils.metric_util import per_class_iu, fast_hist_crop
# from builder import data_builder, model_builder, loss_builder
# from config.config import load_config_data
# from FusionModel import Fusionmodel
# from utils.load_save_util import load_checkpoint
# from thop import profile
# writepath = "/home/ibrahim/Desktop/FusionLocalizationProject33/runs/"
# import warnings
# warnings.filterwarnings("ignore")
# class AtLocCriterion(nn.Module):
#     def __init__(self, t_loss_fn=nn.MSELoss(), q_loss_fn=nn.MSELoss(), sax=0.0, saq=0.0, learn_beta=False):
#         super(AtLocCriterion, self).__init__()
#         self.t_loss_fn = t_loss_fn
#         self.q_loss_fn = q_loss_fn
#         self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
#         self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
#
#     def forward(self, trans, rot, rotgt, transgt):
#         loss = torch.exp(-self.sax) * self.t_loss_fn(trans, transgt) + self.sax + \
#                torch.exp(-self.saq) * self.q_loss_fn(rot, rotgt) + self.saq
#         # loss =  self.t_loss_fn(pred[:, 3:6], targ[:, 3:6]) + self.sax + \
#         #        150 * self.q_loss_fn(pred[:, 0:3], targ[:, 0:3]) + self.saq
#         return loss
#
# class AtLocCriterion2(nn.Module):
#     def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=0.0, saq=0.0, learn_beta=False):
#         super(AtLocCriterion2, self).__init__()
#         # nn.L1Loss()
#         self.t_loss_fn = t_loss_fn
#         self.q_loss_fn = q_loss_fn
#         self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
#         self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
#
#     def forward(self, trans, rot, rotgt, transgt):
#
#         # rot1 = rot.reshape(rot.shape[0] * rot.shape[1])
#         # predr = rotgt
#         # predr = predr.reshape(predr.shape[0] * predr.shape[1])
#         loss = torch.exp(-self.sax) * self.t_loss_fn(trans, transgt) + self.sax + \
#                torch.exp(-self.saq) * self.q_loss_fn(rot, rotgt) + self.saq
#         # loss =  self.t_loss_fn(pred[:, 3:6], targ[:, 3:6]) + self.sax + \
#         #        150 * self.q_loss_fn(pred[:, 0:3], targ[:, 0:3]) + self.saq
#         return loss
#
# def main(args):
#     pytorch_device = torch.device('cuda:0')
#
#     config_path = args.config_path
#
#     configs = load_config_data(config_path)
#
#     dataset_config = configs['dataset_params']
#     train_dataloader_config = configs['train_data_loader']
#     val_dataloader_config = configs['val_data_loader']
#
#     val_batch_size = val_dataloader_config['batch_size']
#     train_batch_size = train_dataloader_config['batch_size']
#
#     model_config = configs['model_params']
#     train_hypers = configs['train_params']
#     grid_size = model_config['output_shape']
#     model_load_path = train_hypers['model_load_path']
#     print(model_load_path)
#     model_save_path = train_hypers['model_save_path']
#
#     writer = SummaryWriter(log_dir=writepath)
#     my_model = Fusionmodel()
#
#     # print(my_model)
#     #
#     # my_model = model_builder.build(model_config)
#     # if os.path.exists(model_load_path):
#     #     print(model_load_path)
#     #     my_model = load_checkpoint(model_load_path, my_model)
#
#     my_model.to(pytorch_device)
#
#     training_loss = AtLocCriterion(saq=-3, learn_beta=True).to(pytorch_device)
#     val_loss = AtLocCriterion().to(pytorch_device)
#
#     # Optimizer
#     param_list = [{'params': my_model.parameters()}]
#     param_list.append({'params': [training_loss.sax, training_loss.saq]})
#     print('learn_beta')
#     optimizer = torch.optim.Adam(param_list, lr=0.0001, weight_decay=0.0005)
#
#     # optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])
#
#     train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
#                                                                   train_dataloader_config,
#                                                                   val_dataloader_config,
#                                                                   grid_size=grid_size)
#
#     # training
#     epoch = 0
#     my_model.train()
#     global_iter = 0
#     check_iter = train_hypers['eval_every_n_steps']
#     while epoch < train_hypers['max_num_epochs']:
#         loss_listlidar = []
#         loss_listimage = []
#         loss_listradar = []
#         pbar = tqdm(total=len(train_dataset_loader))
#         time.sleep(10)
#
#         lidarepoch_loss = 0
#         imageepoch_loss = 0
#         radarpoch_loss = 0
#         previouslossl = 50
#         previouslossr = 50
#
#         previouslossi = 50
#         # lr_scheduler.step(epoch)
#         my_model.train()
#         for i_iter, data in enumerate(train_dataset_loader):
#             train_vox_tenl = [torch.from_numpy(i).to(pytorch_device) for i in data[1]]
#             train_pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in data[2]]
#             train_pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in data[5]]
#             train_vox_tenr = [torch.from_numpy(i).to(pytorch_device) for i in data[4]]
#             monoleft = torch.from_numpy(data[6])
#             monoright = torch.from_numpy(data[7])
#             monorear = torch.from_numpy(data[8])
#             radarimage = torch.from_numpy(data[9]).reshape(train_batch_size,1,512,512)
#             labels = torch.from_numpy(data[10])
#             Transgt = (labels[:, 3:6])
#             rotgt = labels[:, 0:3]
#             # print(Transgt, "Transgtv")
#             if i_iter >= 100:
#                 break
#             # forward + backward + optimize for lidar data
#             trans1, rot1 = my_model([train_pt_fea_tenl, train_vox_tenl, train_batch_size])
#             trans2, rot2 = my_model([train_pt_fea_tenr, train_vox_tenr, train_batch_size])
#             loss1 = training_loss(trans1, rot1, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
#             loss2 = training_loss(trans2, rot2, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
#             loss = loss1 + loss2
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             loss = loss.cpu()
#             loss = loss.detach().numpy()
#             lidarepoch_loss += loss
#             # forward + backward + optimize for image data
#             trans1, rot1 = my_model([monoleft.to(pytorch_device)])
#             trans2, rot2 = my_model([monoright.to(pytorch_device)])
#             trans3, rot3 = my_model([monorear.to(pytorch_device)])
#             loss1 = training_loss(trans1, rot1, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
#             loss2 = training_loss(trans2, rot2, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
#             loss3 = training_loss(trans3, rot3, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
#             lossimage = loss1 + loss2 +loss3
#             lossimage.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             lossimage = lossimage.cpu()
#             lossimage = lossimage.detach().numpy()
#             imageepoch_loss += lossimage
#
#             # forward + backward + optimize for radar data
#             trans1, rot1 = my_model([radarimage.to(pytorch_device)])
#             lossradar = training_loss(trans1, rot1, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
#             lossradar.backward()
#             lossradar =lossradar.cpu()
#             lossradar = lossradar.detach().numpy()
#             optimizer.step()
#             optimizer.zero_grad()
#             radarpoch_loss+= lossradar
#             loss_listlidar.append(loss)
#             loss_listimage.append(lossimage)
#             loss_listradar.append(lossradar)
#             # writer.add_scalar('Training Loss', loss, i_iter)
#             if global_iter % 1000 == 0:
#                 if len(loss_listlidar) > 0:
#                     print('epoch %d iter %5d, Lidarloss: %.3f\n' %
#                           (epoch, i_iter, np.mean(loss_listlidar)))
#
#                     print('epoch %d iter %5d, imageloss: %.3f\n' %
#                           (epoch, i_iter, np.mean(loss_listimage)))
#
#                     print('epoch %d iter %5d, radarloss: %.3f\n' %
#                           (epoch, i_iter, np.mean(loss_listradar)))
#                 else:
#                     print('loss error')
#             optimizer.zero_grad()
#             pbar.update(1)
#             torch.cuda.empty_cache()
#             global_iter += 1
#             if global_iter % check_iter == 0:
#                 if len(loss_listlidar) > 0:
#                     print('epoch %d iter %5d, Lidarloss: %.3f\n' %
#                           (epoch, i_iter, np.mean(loss_listlidar)))
#
#                     print('epoch %d iter %5d, imageloss: %.3f\n' %
#                           (epoch, i_iter, np.mean(loss_listimage)))
#
#                     print('epoch %d iter %5d, radarloss: %.3f\n' %
#                           (epoch, i_iter, np.mean(loss_listradar)))
#                 else:
#                     print('loss error')
#
#         # print((epoch_loss/i_iter)," Training loss")
#         pbar.close()
#         epoch += 1
#         my_model.eval()
#         with torch.no_grad():
#             totalVal_lidarloss = 0
#             totalVal_imageloss = 0
#             totalVal_radarloss = 0
#             for i_iter_val, data in enumerate(val_dataset_loader):
#                 val_grid_tenl = [torch.from_numpy(i).to(pytorch_device) for i in data[1]]
#                 val_pt_fea_tenl = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in data[2]]
#                 val_pt_fea_tenr = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in data[5]]
#                 val_grid_tenr = [torch.from_numpy(i).to(pytorch_device) for i in data[4]]
#                 monoleft = torch.from_numpy(data[6])
#                 monoright = torch.from_numpy(data[7])
#                 monorear = torch.from_numpy(data[8])
#                 radarimage = torch.from_numpy(data[9]).reshape(val_batch_size, 1, 512, 512)
#                 labels = torch.from_numpy(data[10])
#                 Transgt = labels[:, 3:6]
#                 Rotgt = labels[:, 0:3]
#                 trans1, rot1= my_model([val_pt_fea_tenl, val_grid_tenl, val_batch_size])
#                 trans2, rot2 = my_model([val_pt_fea_tenr, val_grid_tenr, val_batch_size])
#
#                 loss_val1 = val_loss(trans1,rot1,Rotgt.to(pytorch_device), Transgt.to(pytorch_device))
#                 loss_val2 = val_loss(trans2, rot2, Rotgt.to(pytorch_device), Transgt.to(pytorch_device))
#                 losslidar = loss_val1 +loss_val2
#
#                 # forward + backward + optimize for image data
#                 trans1, rot1 = my_model([monoleft.to(pytorch_device)])
#                 trans2, rot2 = my_model([monoright.to(pytorch_device)])
#                 trans3, rot3 = my_model([monorear.to(pytorch_device)])
#                 loss1 = training_loss(trans1, rot1, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
#                 loss2 = training_loss(trans2, rot2, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
#                 loss3 = training_loss(trans3, rot3, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
#                 losscam = loss1 + loss2 + loss3
#                 # forward + backward + optimize for radar data
#                 trans1, rot1 = my_model([radarimage.to(pytorch_device)])
#                 lossradar = training_loss(trans1, rot1, rotgt.to(pytorch_device), Transgt.to(pytorch_device))
#                 # print(loss_val,"loss_val")
#                 totalVal_lidarloss += losslidar.item()
#                 writer.add_scalar('Validation Lidar Loss', losslidar, i_iter_val)
#                 totalVal_imageloss += losscam.item()
#                 writer.add_scalar('Validation image Loss', losscam, i_iter_val)
#
#                 totalVal_radarloss += lossradar.item()
#                 writer.add_scalar('Validation Radar Loss', lossradar, i_iter_val)
#                 if i_iter_val >= 100:
#                     break
#             print((totalVal_lidarloss / i_iter_val), " Average Validation  Lidar Loss")
#             print((totalVal_imageloss / i_iter_val), " Average Validation  Image Loss")
#             print((totalVal_radarloss / i_iter_val), " Average Validation  Radar Loss")
#         my_model.train()
#         del val_grid_tenl, val_pt_fea_tenl, val_pt_fea_tenr, val_grid_tenr,monoleft,monoright,monorear,radarimage
#         # save model if performance is improved
#         if previouslossi > (totalVal_imageloss / i_iter_val):
#             previouslossi = (totalVal_imageloss / i_iter_val)
#             torch.save(my_model.state_dict(), "/home/ibrahim/Desktop/FusionLocalizationProject33/LOCALIZATIONMODELS/BestImageModel_save.pt")
#
#         if previouslossl > (totalVal_lidarloss / i_iter_val):
#             previouslossl = (totalVal_lidarloss / i_iter_val)
#             torch.save(my_model.state_dict(), "/home/ibrahim/Desktop/FusionLocalizationProject33/LOCALIZATIONMODELS/BestLidarModel_save.pt")
#
#         if previouslossr > (totalVal_radarloss / i_iter_val):
#             previouslossr = (totalVal_radarloss / i_iter_val)
#             torch.save(my_model.state_dict(), "/home/ibrahim/Desktop/FusionLocalizationProject33/LOCALIZATIONMODELS/BestRidarModel_save.pt")
#
#         if (previouslossr > (totalVal_radarloss / i_iter_val)) and (previouslossl > (totalVal_lidarloss / i_iter_val)) and ( previouslossi > (totalVal_imageloss / i_iter_val)):
#             torch.save(my_model.state_dict(),"/home/ibrahim/Desktop/FusionLocalizationProject33/LOCALIZATIONMODELS/BestLRIModel_save.pt")
#
# if __name__ == '__main__':
#     # Training settings
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('-y', '--config_path', default='/home/ibrahim/Desktop/VoxelbasedLocalizationProject33/config/semantickitti.yaml')
#     args = parser.parse_args()
#     print(' '.join(sys.argv))
#     print(args)
#     main(args)
