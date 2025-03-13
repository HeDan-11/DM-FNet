import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# DEA-Net: Single Image Dehazing Based on Detail-Enhanced Convolution and Content-Guided Attention
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect' ,bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn
    

# DEA-Net: Single Image Dehazing Based on Detail-Enhanced Convolution and Content-Guided Attention    
class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction = 8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )
    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn
    
# DEA-Net: Single Image Dehazing Based on Detail-Enhanced Convolution and Content-Guided Attention    
class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 3, padding=1, padding_mode='reflect' ,groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2) # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2) # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2) # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2

# DEA-Net: Single Image Dehazing Based on Detail-Enhanced Convolution and Content-Guided Attention
class ThreeAttBlock(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ThreeAttBlock, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x):
        res = x
        cattn = self.ca(res)  # torch.Size([1, 768, 1, 1])
        sattn = self.sa(res)  # torch.Size([1, 1, 10, 10])
        pattn1 = sattn + cattn  # torch.Size([1, 768, 10, 10])
        pattn2 = self.pa(res, pattn1)
        res = res * pattn2
        return res
    

def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    '''
    Get the number of input layers to the change detection head.
    '''
    in_channels = 0
    for scale in feat_scales:
        if scale < 3:  # 256 x 256
            in_channels += inner_channel * channel_multiplier[0]  # 128
        elif scale < 6:  # 128 x 128
            in_channels += inner_channel * channel_multiplier[1]  # 256
        elif scale < 9:  # 64 x 64
            in_channels += inner_channel * channel_multiplier[2]  # 128*3
        elif scale < 12:  # 32 x 32
            in_channels += inner_channel * channel_multiplier[3]  # 128*4
        elif scale < 15:  # 16 x 16
            in_channels += inner_channel * channel_multiplier[4]  # 128*6
        else:
            print('Unbounded number for feat_scales. 0<=feat_scales<=14')
    return in_channels


class AttentionBlock2(nn.Module):
    def __init__(self, dim, dim_out, dims, layer_num):
        super().__init__()
        self.att = ThreeAttBlock(dim=dim, reduction=8)
        if layer_num >= 2:
            self.block1 = nn.Sequential(
            nn.Conv2d(dims[layer_num-1], dim, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1, bias=True),
            nn.ReLU(),
            )
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1, bias=True),
            nn.ReLU(),
            )

    def forward(self, x, y, MSFs, lvl):
        w = self.att(x + y)
        fea = w*x + (1-w)*y +  x + y

        if lvl == 1:
            fea = self.block(fea)
        elif lvl > 2:
            be_fea = F.interpolate(self.block1(MSFs[lvl-3]), scale_factor=2, mode="bilinear", align_corners=True)
            fea = self.block(fea+MSFs[lvl-2]+be_fea)
        else:
            # print(fea.shape, F.shape)
            fea = self.block(fea+MSFs[lvl-2])
        # fea = f_s
        return fea


class Block1(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(dim * len(time_steps)),
            # nn.ReLU(),
            nn.Conv2d(dim * len(time_steps), dim, 1),
            # if len(time_steps) > 1
            # else None,
            nn.ReLU(),
            # if len(time_steps) > 1
            # else None,
            # nn.Conv2d(dim_out, dim_out, 3, padding=1),
            # nn.ReLU(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        print()
        c_num = len(time_steps)
        if len(time_steps) == 0:
            c_num = 1
        self.block = nn.Sequential(
            nn.Conv2d(dim * c_num, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # fea_ca = self.c_att(self.bn(x))
        return self.block(x)



class HeadTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(HeadTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)

    def forward(self, x):
        return torch.tanh(self.conv(x))   # (-1, 1)


class HeadLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(HeadLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class Fusion_Head(nn.Module):
    '''
    Change detection head (version 2).
    '''
    def __init__(self, feat_scales, out_channels=1, inner_channel=None, channel_multiplier=None, img_size=256,
                 time_steps=None, hard_gate=False):
        super(Fusion_Head, self).__init__()
        # feat_scales = [2]
        # Define the parameters of the change detection head
        feat_scales.sort(reverse=True)
        self.feat_scales = feat_scales
        self.in_channels = get_in_channels(feat_scales, inner_channel, channel_multiplier)
        self.img_size = img_size
        self.time_steps = time_steps
        
        dims = []
        for i in range(0, len(self.feat_scales)):
            dim = get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)
            dims.append(dim)
        # Convolutional layers before parsing to difference head
        self.decoder = nn.ModuleList()
        for i in range(0, len(self.feat_scales)):  # [2, 5, 8, 11, 14]
            dim = dims[i]
            self.decoder.append(
                Block(dim=dim, dim_out=dim, time_steps=time_steps)
            )

            if i != len(self.feat_scales) - 1:
                dim_out = dims[i+1]
            else:
                dim_out = dims[i]
                
            # AttentionBlock2
            self.decoder.append(AttentionBlock2(dim=dim, dim_out=dim_out, dims=dims,layer_num = i))
        # Final head
        self.rgb_decode2 = HeadLeakyRelu2d(128, 64)
        self.rgb_decode1 = HeadTanh2d(64, out_channels)

    def forward(self, y1, y2, feats, hard_gate=False):
        # Decoder
        lvl = 0
        MSFs = []
        for layer in self.decoder: # 10-20-40-80-160
            if isinstance(layer, Block):
                f_s_1 = feats['MRI'][0][self.feat_scales[lvl]]  # feature stacked [2, 5, 8, 11, 14]
                f_s_2 = feats['Other'][0][self.feat_scales[lvl]]  # feature stacked [2, 5, 8, 11, 14]
                if len(self.time_steps) > 1:
                   for i in range(1, len(self.time_steps)):                 
                      f_s_1 = torch.cat((f_s_1, feats['MRI'][i][self.feat_scales[lvl]]), dim=1)
                      f_s_2 = torch.cat((f_s_2, feats['Other'][i][self.feat_scales[lvl]]), dim=1)
                
                   f_s_1 = layer(f_s_1) 
                   f_s_2 = layer(f_s_2) 
                lvl += 1
            else:
                f_s = layer(f_s_1, f_s_2, MSFs, lvl)
                if lvl != len(self.feat_scales):
                    MSFs.append(F.interpolate(f_s, scale_factor=2, mode="bilinear", align_corners=True))

                    
                # MSFs.append(x)
        # MSFs.append(f_s)

        # Fusion Head
        x = self.rgb_decode2(f_s)
        rgb_img = self.rgb_decode1(x)
        return rgb_img