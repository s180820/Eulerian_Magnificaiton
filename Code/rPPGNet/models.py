import math
import torch.nn as nn
from torch.nn.modules.utils import _triple
import pdb
import torch
import torch.nn.functional as F
import random


class SpatioTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

		
        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/(kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))
        
        # self-definition
        #intermed_channels = int((in_channels+intermed_channels)/2)

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()   ##   nn.Tanh()   or   nn.ReLU(inplace=True)


        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))  
        x = self.temporal_conv(x)                      
        return x


class MixA_Module(nn.Module):
    """ Spatial-Skin attention module"""
    def __init__(self):
        super(MixA_Module,self).__init__()
        self.softmax  = nn.Softmax(dim=-1)
        self.AVGpool = nn.AdaptiveAvgPool1d(1)
        self.MAXpool = nn.AdaptiveMaxPool1d(1)
    def forward(self,x , skin):
        """
            inputs :
                x : input feature maps( B X C X T x W X H)
                skin : skin confidence maps( B X T x W X H)
            returns :
                out : attention value
                spatial attention: W x H
        """
        m_batchsize, C, T ,W, H = x.size()
        B_C_TWH = x.view(m_batchsize,C,-1)
        B_TWH_C = x.view(m_batchsize,C,-1).permute(0,2,1)
        B_TWH_C_AVG =  torch.sigmoid(self.AVGpool(B_TWH_C)).view(m_batchsize,T,W,H)
        B_TWH_C_MAX =  torch.sigmoid(self.MAXpool(B_TWH_C)).view(m_batchsize,T,W,H)
        B_TWH_C_Fusion = B_TWH_C_AVG + B_TWH_C_MAX + skin
        Attention_weight = self.softmax(B_TWH_C_Fusion.view(m_batchsize,T,-1))
        Attention_weight = Attention_weight.view(m_batchsize,T,W,H)
        # mask1 mul
        output = x.clone()
        for i in range(C):
            output[:,i,:,:,:] = output[:,i,:,:,:].clone()*Attention_weight
        
        return output, Attention_weight    


# for open-source
# skin segmentation + PhysNet + MixA3232 + MixA1616part4
class rPPGNet(nn.Module):
    def __init__(self, frames=64):  
        super(rPPGNet, self).__init__()
        
        self.ConvSpa1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvSpa3 = nn.Sequential(
            SpatioTemporalConv(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa4 = nn.Sequential(
            SpatioTemporalConv(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        
        self.ConvSpa5 = nn.Sequential(
            SpatioTemporalConv(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa6 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa7 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa8 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa9 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
 
        self.ConvSpa10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.ConvSpa11 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.ConvPart1 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.ConvPart2 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.ConvPart3 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        self.ConvPart4 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
           
        
        self.AvgpoolSpa = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.AvgpoolSkin_down = nn.AvgPool2d((2,2), stride=2)
        self.AvgpoolSpaTem = nn.AvgPool3d((2, 2, 2), stride=2)
        
        self.ConvSpa = nn.Conv3d(3, 16, [1,3,3],stride=1, padding=[0,1,1])
        
        
        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))    # attention to this value 
        

        # skin_branch
        self.skin_main = nn.Sequential(
            nn.Conv3d(32, 16, [1,3,3], stride=1, padding=[0,1,1]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, [1,3,3], stride=1, padding=[0,1,1]),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        
        self.skin_residual = nn.Sequential(
            nn.Conv3d(32, 8, [1,1,1], stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        
        self.skin_output = nn.Sequential(
            nn.Conv3d(8, 1, [1,3,3], stride=1, padding=[0,1,1]),
            nn.Sigmoid(),   ## binary 
        )
        
        self.MixA_Module = MixA_Module()
        
    def forward(self, x):	    	# x [3, 64, 128,128]
        x_visual = x
          
        x = self.ConvSpa1(x)		     # x [3, 64, 128,128]
        x = self.AvgpoolSpa(x)       # x [16, 64, 64,64]
        
        x = self.ConvSpa3(x)		    # x [32, 64, 64,64]
        x_visual6464 = self.ConvSpa4(x)	    	# x [32, 64, 64,64]
        x = self.AvgpoolSpa(x_visual6464)      # x [32, 64, 32,32]
        
        
        ## branch 1: skin segmentation
        x_skin_main = self.skin_main(x_visual6464)    # x [8, 64, 64,64]
        x_skin_residual = self.skin_residual(x_visual6464)   # x [8, 64, 64,64]
        x_skin = self.skin_output(x_skin_main+x_skin_residual)    # x [1, 64, 64,64]
        x_skin = x_skin[:,0,:,:,:]    # x [74, 64,64]
        
        
        ## branch 2: rPPG
        x = self.ConvSpa5(x)		    # x [64, 64, 32,32]
        x_visual3232 = self.ConvSpa6(x)	    	# x [64, 64, 32,32]
        x = self.AvgpoolSpa(x_visual3232)      # x [64, 64, 16,16]
        
        x = self.ConvSpa7(x)		    # x [64, 64, 16,16]
        x = self.ConvSpa8(x)	    	# x [64, 64, 16,16]
        x_visual1616 = self.ConvSpa9(x)	    	# x [64, 64, 16,16]
        
        
        ## SkinA1_loss
        x_skin3232 = self.AvgpoolSkin_down(x_skin)          # x [64, 32,32]
        x_visual3232_SA1, Attention3232 = self.MixA_Module(x_visual3232, x_skin3232)
        x_visual3232_SA1 = self.poolspa(x_visual3232_SA1)     # x [64, 64, 1,1]    
        ecg_SA1  = self.ConvSpa10(x_visual3232_SA1).squeeze(1).squeeze(-1).squeeze(-1)
        
        
        ## SkinA2_loss
        x_skin1616 = self.AvgpoolSkin_down(x_skin3232)       # x [64, 16,16]
        x_visual1616_SA2, Attention1616 = self.MixA_Module(x_visual1616, x_skin1616)
        ## Global
        global_F = self.poolspa(x_visual1616_SA2)     # x [64, 64, 1,1]    
        ecg_global = self.ConvSpa11(global_F).squeeze(1).squeeze(-1).squeeze(-1)
        
        ## Local
        Part1 = x_visual1616_SA2[:,:,:,:8,:8]
        Part1 = self.poolspa(Part1)     # x [64, 64, 1,1]    
        ecg_part1 = self.ConvSpa11(Part1).squeeze(1).squeeze(-1).squeeze(-1)
        
        Part2 = x_visual1616_SA2[:,:,:,8:16,:8]
        Part2 = self.poolspa(Part2)     # x [64, 64, 1,1]    
        ecg_part2 = self.ConvPart2(Part2).squeeze(1).squeeze(-1).squeeze(-1)
        
        Part3 = x_visual1616_SA2[:,:,:,:8,8:16]
        Part3 = self.poolspa(Part3)     # x [64, 64, 1,1]    
        ecg_part3 = self.ConvPart3(Part3).squeeze(1).squeeze(-1).squeeze(-1)
        
        Part4 = x_visual1616_SA2[:,:,:,8:16,8:16]
        Part4 = self.poolspa(Part4)     # x [64, 64, 1,1]    
        ecg_part4 = self.ConvPart4(Part4).squeeze(1).squeeze(-1).squeeze(-1)
        
        

        return x_skin, ecg_SA1, ecg_global, ecg_part1, ecg_part2, ecg_part3, ecg_part4, x_visual6464, x_visual3232
    
    @property 
    def name(self):
        return "rPPGNet"



class SNREstimatorNetMonteCarlo(nn.Module):
    def __init__(self):
        super(SNREstimatorNetMonteCarlo, self).__init__()
        self.is_active_layer = []

    def setup(self, active_layers, max_pool_kernel_size, conv_kernel_size, conv_filter_size):
        self.is_active_layer = active_layers

        max_pool_kernel_size = int(max_pool_kernel_size)
        conv_kernel_size = int(conv_kernel_size)

        conv_init_mean = 0
        conv_init_std = .1
        xavier_normal_gain = 1

        self.bn_input = nn.BatchNorm1d(1)
        nn.init.normal_(self.bn_input.weight, conv_init_mean, conv_init_std)

        output_count = int(conv_filter_size)
        input_count = 1
        self.conv_00 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
        nn.init.xavier_normal_(self.conv_00.weight, gain=xavier_normal_gain)
        self.bn_00 = nn.BatchNorm1d(output_count)
        nn.init.normal_(self.bn_00.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[1] == 1:
            input_count = output_count
            self.conv_01 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_01.weight, gain=xavier_normal_gain)
            self.bn_01 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_01.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[2] == 1:
            input_count = output_count
            self.conv_02 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_02.weight, gain=xavier_normal_gain)
            self.max_pool1d_02 = nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=1)
            self.bn_02 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_02.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[3] == 1:
            input_count = output_count
            self.conv_10 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_10.weight, gain=xavier_normal_gain)
            self.bn_10 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_10.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[4] == 1:
            input_count = output_count
            self.conv_11 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_11.weight, gain=xavier_normal_gain)
            self.bn_11 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_11.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[5] == 1:
            input_count = output_count
            self.conv_12 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_12.weight, gain=xavier_normal_gain)
            self.max_pool1d_12 = nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=1)
            self.bn_12 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_12.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[6] == 1:
            input_count = output_count
            self.conv_20 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_20.weight, gain=xavier_normal_gain)
            self.bn_20 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_20.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[7] == 1:
            input_count = output_count
            self.conv_21 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_21.weight, gain=xavier_normal_gain)
            self.bn_21 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_21.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[8] == 1:
            input_count = output_count
            self.conv_22 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_22.weight, gain=xavier_normal_gain)
            self.max_pool1d_22 = nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=1)
            self.bn_22 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_22.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[9] == 1:
            input_count = output_count
            self.conv_30 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_30.weight, gain=xavier_normal_gain)
            self.bn_30 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_30.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[10] == 1:
            input_count = output_count
            self.conv_31 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_31.weight, gain=xavier_normal_gain)
            self.bn_31 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_31.weight, conv_init_mean, conv_init_std)

        if self.is_active_layer[11] == 1:
            input_count = output_count
            self.conv_32 = nn.Conv1d(input_count, output_count, kernel_size=conv_kernel_size, stride=1, padding=0)
            nn.init.xavier_normal_(self.conv_32.weight, gain=xavier_normal_gain)
            self.max_pool1d_32 = nn.MaxPool1d(kernel_size=max_pool_kernel_size, stride=1)
            self.bn_32 = nn.BatchNorm1d(output_count)
            nn.init.normal_(self.bn_32.weight, conv_init_mean, conv_init_std)

        input_count = output_count
        self.conv_last = nn.Conv1d(input_count, 1, kernel_size=1, stride=1, padding=0)
        nn.init.xavier_normal_(self.conv_last.weight, gain=xavier_normal_gain)

        self.ada_avg_pool1d = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x):
        nonlin = F.elu

        x = x - torch.mean(x)
        
        x = self.bn_input(x)
        print(x.size())
        x = nonlin(self.bn_00(self.conv_00(F.dropout(x, p=0.1, training=self.training))))
        if self.is_active_layer[1] == 1:
            x = nonlin(self.bn_01(self.conv_01(F.dropout(x, p=0.1, training=self.training))))
        if self.is_active_layer[2] == 1:
            x = nonlin(self.bn_02(self.max_pool1d_02(self.conv_02(F.dropout(x, p=0.1, training=self.training)))))
        if self.is_active_layer[3] == 1:
            x = nonlin(self.bn_10(self.conv_10(F.dropout(x, p=0.15, training=self.training))))
        if self.is_active_layer[4] == 1:
            x = nonlin(self.bn_11(self.conv_11(F.dropout(x, p=0.15, training=self.training))))
        if self.is_active_layer[5] == 1:
            x = nonlin(self.bn_12(self.max_pool1d_12(self.conv_12(F.dropout(x, p=0.15, training=self.training)))))
        if self.is_active_layer[6] == 1:
            x = nonlin(self.bn_20(self.conv_20(F.dropout(x, p=0.2, training=self.training))))
        if self.is_active_layer[7] == 1:
            x = nonlin(self.bn_21(self.conv_21(F.dropout(x, p=0.2, training=self.training))))
        if self.is_active_layer[8] == 1:
            x = nonlin(self.bn_22(self.max_pool1d_22(self.conv_22(F.dropout(x, p=0.2, training=self.training)))))
        if self.is_active_layer[9] == 1:
            x = nonlin(self.bn_30(self.conv_30(F.dropout(x, p=0.3, training=self.training))))
        if self.is_active_layer[10] == 1:
            x = nonlin(self.bn_31(self.conv_31(F.dropout(x, p=0.3, training=self.training))))
        if self.is_active_layer[11] == 1:
            x = nonlin(self.bn_32(self.max_pool1d_32(self.conv_32(F.dropout(x, p=0.3, training=self.training)))))

        x = self.conv_last(F.dropout(x, p=0.5, training=self.training))
        
        x = self.ada_avg_pool1d(x)
        
        if sum(x.size()[1:]) > x.dim() - 1:
            print(x.size())
            raise ValueError('Check your network idiot!')

        return x
    
    @property 
    def name(self):
        return "SNREstimatorNetMonteCarlo"



class FaceHRNet09V4ELU(nn.Module):
    def __init__(self, rgb):
        super(FaceHRNet09V4ELU, self).__init__()

        self.rgb = rgb

        self.ada_avg_pool2d = nn.AdaptiveAvgPool2d(output_size=(192, 128))

        conv_init_mean = 0
        conv_init_std = .1
        xavier_normal_gain = 1

        self.bn_input = nn.BatchNorm2d(3 if rgb else 1)
        nn.init.normal_(self.bn_input.weight, conv_init_mean, conv_init_std)

        input_count = 1
        if self.rgb:
            input_count = 3

        output_count = 64 
        self.conv_00 = nn.Conv2d(input_count, output_count, kernel_size=(15, 10), stride=1, padding=0)
        nn.init.xavier_normal_(self.conv_00.weight, gain=xavier_normal_gain)
        self.max_pool2d_00 = nn.MaxPool2d(kernel_size=(15, 10), stride=(2, 2))
        self.bn_00 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_00.weight, conv_init_mean, conv_init_std)

        input_count = 64 
        self.conv_01 = nn.Conv2d(input_count, output_count, kernel_size=(15, 10), stride=1, padding=0)
        nn.init.xavier_normal_(self.conv_01.weight, gain=xavier_normal_gain)
        self.max_pool2d_01 = nn.MaxPool2d(kernel_size=(15, 10), stride=(1, 1))
        self.bn_01 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_01.weight, conv_init_mean, conv_init_std)

        output_count = 64 
        self.conv_10 = nn.Conv2d(input_count, output_count, kernel_size=(15, 10), stride=1, padding=0)
        nn.init.xavier_normal_(self.conv_10.weight, gain=xavier_normal_gain)
        self.max_pool2d_10 = nn.MaxPool2d(kernel_size=(15, 10), stride=(1, 1))
        self.bn_10 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_10.weight, conv_init_mean, conv_init_std)

        input_count = 64        

        output_count = 64
        self.conv_20 = nn.Conv2d(input_count, output_count, kernel_size=(12, 10), stride=1, padding=0)
        nn.init.xavier_normal_(self.conv_20.weight, gain=xavier_normal_gain)
        self.max_pool2d_20 = nn.MaxPool2d(kernel_size=(15, 10), stride=(1, 1))
        self.bn_20 = nn.BatchNorm2d(output_count)
        nn.init.normal_(self.bn_20.weight, conv_init_mean, conv_init_std)

        input_count = 64
        self.conv_last = nn.Conv2d(input_count, 1, kernel_size=1, stride=1, padding=0)
        nn.init.xavier_normal_(self.conv_last.weight, gain=xavier_normal_gain)

    def forward(self, x):
        nonlin = F.elu
        
        x = self.ada_avg_pool2d(x)

        x = self.bn_input(x)

        x = nonlin(self.bn_00(self.max_pool2d_00(self.conv_00(F.dropout2d(x, p=0.0, training=self.training)))))
        x = nonlin(self.bn_01(self.max_pool2d_01(self.conv_01(F.dropout(x, p=0.0, training=self.training)))))
        x = nonlin(self.bn_10(self.max_pool2d_10(self.conv_10(F.dropout(x, p=0.0, training=self.training)))))
        x = nonlin(self.bn_20(self.max_pool2d_20(self.conv_20(F.dropout2d(x, p=0.2, training=self.training)))))  
        #x = nonlin(self.max_pool2d_00(self.conv_00(F.dropout2d(x, p=0.0, training=self.training))))
        #x = nonlin(self.max_pool2d_01(self.conv_01(F.dropout(x, p=0.0, training=self.training))))
        #x = nonlin(self.max_pool2d_10(self.conv_10(F.dropout(x, p=0.0, training=self.training))))
        #x = nonlin(self.max_pool2d_20(self.conv_20(F.dropout2d(x, p=0.2, training=self.training))))       

        x = self.conv_last(F.dropout(x, p=0.5, training=self.training))
        if sum(x.size()[1:]) > x.dim() - 1:
            print(x.size())
            raise ValueError('Check your network idiot!')

        return x    
    
    @property 
    def name(self):
        return "FaceExtractor"

models = {
    "rPPGNet": rPPGNet,
    "SNREstimatorNetMonteCarlo" : SNREstimatorNetMonteCarlo,
    "FaceHRNet09V4ELU" : FaceHRNet09V4ELU,
}