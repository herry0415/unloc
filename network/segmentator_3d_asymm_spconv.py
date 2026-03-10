# -*- coding:utf-8 -*-
# author: Xinge
# @file: segmentator_3d_asymm_spconv.py

import numpy as np
import spconv
import torch
from torch import nn
import torch.nn as nn

from Slotattentionmodule import SlotAttentionAutoEncoder


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print(x.features.shape, "conv")
        shortcut = self.conv1(x)
        # print(shortcut.features.shape, "shortcut")

        shortcut.features = self.act1(shortcut.features)
        shortcut.features = self.bn0(shortcut.features)

        shortcut = self.conv1_2(shortcut)
        shortcut.features = self.act1_2(shortcut.features)
        shortcut.features = self.bn0_2(shortcut.features)

        resA = self.conv2(x)
        resA.features = self.act2(resA.features)
        resA.features = self.bn1(resA.features)

        resA = self.conv3(resA)
        resA.features = self.act3(resA.features)
        resA.features = self.bn2(resA.features)
        resA.features = resA.features + shortcut.features

        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.features = self.act1(shortcut.features)
        shortcut.features = self.bn0(shortcut.features)

        shortcut = self.conv1_2(shortcut)
        shortcut.features = self.act1_2(shortcut.features)
        shortcut.features = self.bn0_2(shortcut.features)

        resA = self.conv2(x)
        resA.features = self.act2(resA.features)
        resA.features = self.bn1(resA.features)

        resA = self.conv3(resA)
        resA.features = self.act3(resA.features)
        resA.features = self.bn2(resA.features)

        resA.features = resA.features + shortcut.features

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA.features = self.trans_act(upA.features)
        upA.features = self.trans_bn(upA.features)

        ## upsample
        upA = self.up_subm(upA)

        upA.features = upA.features + skip.features

        upE = self.conv1(upA)
        upE.features = self.act1(upE.features)
        upE.features = self.bn1(upE.features)

        upE = self.conv2(upE)
        upE.features = self.act2(upE.features)
        upE.features = self.bn2(upE.features)

        upE = self.conv3(upE)
        upE.features = self.act3(upE.features)
        upE.features = self.bn3(upE.features)

        return upE




class WeightedAttention(nn.Module):
    def __init__(self, dim, eps = 1e-8, softmax_dim = 1, weighted_mean_dim = 2):
        super().__init__()
        self.norm_input = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.eps = eps
        self.scale = dim ** -0.5
        self.softmax_dim = softmax_dim
        self.weighted_mean_dim = weighted_mean_dim

    def forward(self, inputs, context):

        inputs = self.norm_input(inputs)
        context = self.norm_context(context)

        q = self.to_q(inputs)
        k = self.to_k(context)
        v = self.to_v(context)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim = self.softmax_dim) + self.eps
        attn = attn / attn.sum(dim = self.weighted_mean_dim, keepdim=True)

        updates = torch.einsum('bjd,bij->bid', v, attn)
        return updates

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class GatedResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.fn = fn
    def forward(self, *args):
        inputs = args[0]
        b, _, d = inputs.shape

        updates = self.fn(*args)

        inputs = self.gru(
            updates.reshape(-1, d),
            inputs.reshape(-1, d)
        )
        return inputs.reshape(b, -1, d)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_dim = max(dim, hidden_dim)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)


class MLPs(nn.Module):

    def __init__(self, fea_dim=20,
                 out_pt_fea_dim=64):
        super(MLPs, self).__init__()

        self.PPmodel = nn.Sequential(
            # nn.BatchNorm1d(fea_dim),

            nn.Linear(fea_dim, out_pt_fea_dim),
            # nn.BatchNorm1d(64),
            nn.LayerNorm(out_pt_fea_dim),
            nn.ReLU(),

            # nn.Linear(64, out_pt_fea_dim),
            # # nn.BatchNorm1d(128),
            # nn.LayerNorm(out_pt_fea_dim),
            # nn.ReLU()

        )

    def forward(self, x):
        x = self.PPmodel(x)
        return x



class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_slots, encoder_dims, iters=3, hidden_dim=128, eps=1e-8):
        """Builds the Slot Attention module.
        Args:
            iters: Number of iterations.
            num_slots: Number of slots.
            encoder_dims: Dimensionality of slot feature vectors.
            hidden_dim: Hidden layer size of MLP.
            eps: Offset for attention coefficients before normalization.
        """
        super(SlotAttention, self).__init__()

        self.eps = eps
        self.iters = iters
        self.num_slots = num_slots
        self.scale = encoder_dims ** -0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.norm_input = nn.LayerNorm(encoder_dims)
        self.norm_slots = nn.LayerNorm(encoder_dims)
        self.norm_pre_ff = nn.LayerNorm(encoder_dims)

        # Parameters for Gaussian init (shared by all slots).
        # self.slots_mu = nn.Parameter(torch.randn(1, 1, encoder_dims))
        # self.slots_sigma = nn.Parameter(torch.randn(1, 1, encoder_dims))

        self.slots_embedding = nn.Embedding(num_slots, encoder_dims)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(encoder_dims, encoder_dims)
        self.project_k = nn.Linear(encoder_dims, encoder_dims)
        self.project_v = nn.Linear(encoder_dims, encoder_dims)

        # Slot update functions.
        self.gru = nn.GRUCell(encoder_dims, encoder_dims)

        hidden_dim = max(encoder_dims, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dims, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, encoder_dims)
        )

    def forward(self, inputs, num_slots=None):
        # inputs has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_input(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        # random slots initialization,
        # mu = self.slots_mu.expand(b, n_s, -1)
        # sigma = self.slots_sigma.expand(b, n_s, -1)
        # slots = torch.normal(mu, sigma)

        # learnable slots initialization
        slots = self.slots_embedding(torch.arange(0, n_s).expand(b, n_s).to(self.device))

        # Multiple rounds of attention.
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean.

            updates = torch.einsum('bjd,bij->bid', v, attn)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )
            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots

def truncated_normal_(tensor, mean=0, std=.02):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4, )).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor
class SoftPositionEmbed(nn.Module):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, hidden_size):
        """Builds the soft position embedding layer.
        Args:
          hidden_size: Size of input feature dimension.
          resolution: Tuple of integers specifying width and height of grid.
        """
        super(SoftPositionEmbed, self).__init__()
        self.proj = nn.Linear(4, hidden_size)
        # self.grid = build_grid(resolution)
        self.pos_embedd = nn.Parameter(truncated_normal_(torch.randn(1, 1, hidden_size)))

    def forward(self, inputs):
        # b,x,c
        return inputs + self.pos_embedd # + self.proj(self.grid)

class FeedForward1(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)






class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.features = self.bn0(shortcut.features)
        shortcut.features = self.act1(shortcut.features)

        shortcut2 = self.conv1_2(x)
        shortcut2.features = self.bn0_2(shortcut2.features)
        shortcut2.features = self.act1_2(shortcut2.features)

        shortcut3 = self.conv1_3(x)
        shortcut3.features = self.bn0_3(shortcut3.features)
        shortcut3.features = self.act1_3(shortcut3.features)
        shortcut.features = shortcut.features + shortcut2.features + shortcut3.features

        shortcut.features = shortcut.features * x.features

        return shortcut


class Asymm_3d_spconv(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")
        # self.slotatten = SlotAttentionAutoEncoder(20,8)

        self.logits = spconv.SubMConv3d(4* init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)

        # self.slot_attention1 = SlotAttention(
        #     iters=6,
        #     num_slots=20,
        #     encoder_dims=128,
        #     hidden_dim=128)
        #
        # self.posemd = SoftPositionEmbed(128)
        #
        # self.layer_norm = nn.LayerNorm(128)
        #
        # self.mlp = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 128))
        #
        # self.inputs_to_slots_attn = GatedResidual(128, WeightedAttention(128, eps=1e-8, softmax_dim=2,
        #                                                                      weighted_mean_dim=1))
        # self.inputs_ff = GatedResidual(128, FeedForward(128, 128))





    def forward(self, voxel_features, coors, batch_size):

        # x = x.contiguous()
        # print(len(coors))
        coors = coors.int()



        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        # print(ret.features.shape,"Before")

        ret = self.downCntx(ret)
        # print(ret.features.shape,"After")
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        # feats = torch.reshape(up1e.features, (1, up1e.features.shape[0], up1e.features.shape[1]))
        # slotfeat = self.slotatten(feats)
        # slotfeat = slotfeat.squeeze()


        up0e = self.ReconNet(up1e)

        up0e.features = torch.cat((up0e.features, up1e.features), 1)


        # t1 = up0e.
        # t2 = up1e.features

        # print(t1.shape,"up0e.features")
        # print(t2.shape,"up1e.features")


        # up0e.features = torch.cat((up0e.features, up1e.features,slotfeat), 1)
        # tens = up0e.dense()
        # print(tens.shape,"features")
        # feats = torch.reshape(up0e.features,(1,up0e.features.shape[0],up0e.features.shape[1]))
        # feats = self.posemd(up0e.features)
        # feats  = self.mlp(self.layer_norm(feats))
        #
        # # print(feats.shape)
        #
        # slots = self.slot_attention1(feats)
        #
        # x2 = self.inputs_to_slots_attn(feats , slots)
        # x2 = self.inputs_ff(x2)
        # x2= x2.squeeze()
        # up0e.features =up0e.features + x2


        # print(up0e.features.shape)
        # print(up0e.Coors.shape)

        logits = self.logits(up0e)
        # print(logits.features.shape)

        y = logits.dense()
        # print(y.shape)
        return y
