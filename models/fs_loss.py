import torch
import torch.nn as nn
import torch.nn.functional as F
# Parts of these codes are from: https://github.com/Linfeng-Tang/SeAFusion
from torch.autograd import Variable
import numpy as np
import os
from math import exp
import config as c

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim_swinfusion(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim_raw(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)


    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    ret = ssim_map.mean()

    return 1 - ret

def _ssim(img1, img2, img3, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu3 = F.conv2d(img3, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu3_sq = mu3.pow(2)
    mu1_mu2 = mu1 * mu2
    mu1_mu3 = mu1 * mu3
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma3_sq = F.conv2d(img3 * img3, window, padding=window_size // 2, groups=channel) - mu3_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    sigma13 = F.conv2d(img1 * img3, window, padding=window_size // 2, groups=channel) - mu1_mu3

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    x2=torch.sqrt(sigma2_sq)
    x3=torch.sqrt(sigma3_sq)
    #ssim_map_12 = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    #ssim_map_13 = ((2 * mu1_mu3 + C1) * (2 * sigma13 + C2)) / ((mu1_sq + mu3_sq + C1) * (sigma1_sq + sigma3_sq + C2))
    ssim_map_12 = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map_13 = (2 * sigma13 + C2) / (sigma1_sq + sigma3_sq + C2)
    nen_2=mu2_sq
    nen_3=mu3_sq
    if size_average:
        ssim_map_std = torch.where(torch.abs((x2))>=torch.abs((x3)), ssim_map_12, ssim_map_13)
        #ssim_map_mu = torch.where(torch.abs((mu2))>=torch.abs((mu3)), ssim_map_12, ssim_map_13)
        #ssim_map_nen = (nen_2 >= nen_3) * ssim_map_12 + (nen_2 < nen_3) * ssim_map_13
        #ssim_map = torch.max(ssim_map_mu.mean(),ssim_map_nen.mean())
        #ssim_map = torch.cat([ssim_map_std, ssim_map_mu], dim=1)
        return ssim_map_std.mean()
    else:
        ssim_map = torch.where(x2>x3, torch.abs(ssim_map_12), torch.abs(ssim_map_13))
        return ssim_map.mean(1).mean(1).mean(1)

def SSIM(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map_12 = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map_12.mean()

def ssim_loss(fused_result, input_max):
    ssim_loss = SSIM(fused_result, input_max)
    return 1 - ssim_loss

def ssim(img1, img2, img3, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, img3,  window, window_size, channel, size_average)


def ssim_swin(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return ssim_swinfusion(img1, img2, window, window_size, channel, size_average)


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device)

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                tensor_list.append(add_0)

            batch_list.append(torch.stack(tensor_list, dim=1))

        return torch.cat(batch_list, dim=0)


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()
        self.mse_criterion = torch.nn.MSELoss()

    def forward(self, image_vis, image_ir, generate_img):
        
        B, C, W, H = image_vis.shape
        image_ir = image_ir.expand(B, C, W, H)
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(generate_img, x_in_max)
        # Gradient
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        B, C, K, W, H = y_grad.shape
        ir_grad = ir_grad.expand(B, C, K, W, H)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.maximum(y_grad, ir_grad)
        loss_grad = F.l1_loss(generate_img_grad, x_grad_joint)

        return loss_in, loss_grad



class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                tensor_list.append(add_0)

            batch_list.append(torch.stack(tensor_list, dim=1))

        return torch.cat(batch_list, dim=0)


def rgb2ycbcr_t(img_rgb):
    R = img_rgb[:,0, :, :].unsqueeze(1)
    G = img_rgb[:,1, :, :].unsqueeze(1)
    B = img_rgb[:,2, :, :].unsqueeze(1)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128/255
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128/255
    img_ycbcr = torch.cat([Y, Cb, Cr], axis=1)
    return img_ycbcr, Y, Cb, Cr


class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()
        self.mse_criterion = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, data, generate_img):
        image_vis, image_ir = data["vis"], data["ir"]
        image_y = image_vis

        B, C, W, H = image_vis.shape
        image_ir = image_ir.expand(B, C, W, H)
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(generate_img, x_in_max)
        loss_in = 1.5 * loss_in
        # Gradient loss
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        B, C, K, W, H = y_grad.shape
        ir_grad = ir_grad.expand(B, C, K, W, H)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.maximum(y_grad, ir_grad)
        loss_grad = F.l1_loss(generate_img_grad, x_grad_joint)
        
        # ssim + grad
        lssim = 0.5 * (1 - ssim(generate_img, image_ir, image_vis))
        loss_grad =  lssim + loss_grad


        return loss_in, loss_grad
