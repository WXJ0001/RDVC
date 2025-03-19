import math
import numpy
import PIL.Image
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import math

import torch
from torch import nn
import torch.nn.functional as F

backwarp_tenGrid = {}

Backward_tensorGrid = [{} for i in range(8)]
Backward_tensorGrid_cpu = {}

def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(
            1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(
            1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenInput,
                                           grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1),
                                           mode='bilinear', padding_mode='border', align_corners=False)


# 3:720150, 4:960200, 5:1200250  6:1440300
class SpyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.netBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )
            def forward(self, tenInput):
                return self.netBasic(tenInput)

        self.L = 4
        self.netBasic = torch.nn.ModuleList([Basic() for _ in range(self.L)])
        self.ckpt = './checkpoint/spynet-sintel-final'
        tgt_model_dict = self.netBasic.state_dict()
        src_pretrained_dict = torch.load(self.ckpt)
        # for k, v in src_pretrained_dict.items():
        #     if k[12:].replace('module', 'net') in tgt_model_dict:
        #         print(f'Load Parameters {k}, {v.shape}')
        _pretrained_dict = {k[12:].replace('module', 'net'): v for k, v in src_pretrained_dict.items()
                            if k[12:].replace('module', 'net') in tgt_model_dict}
        tgt_model_dict.update(_pretrained_dict)
        self.netBasic.load_state_dict(tgt_model_dict)

    def forward(self, tenOne, tenTwo):
        tenOne = [tenOne]
        tenTwo = [tenTwo]
        for intLevel in range(self.L - 1):
            tenOne.insert(0, torch.nn.functional.avg_pool2d(input=tenOne[0], kernel_size=2, stride=2,
                                                            count_include_pad=False))
            tenTwo.insert(0, torch.nn.functional.avg_pool2d(input=tenTwo[0], kernel_size=2, stride=2,
                                                            count_include_pad=False))

        tenFlow = tenOne[0].new_zeros([tenOne[0].shape[0], 2, int(math.floor(tenOne[0].shape[2] / 2.0)),
                                       int(math.floor(tenOne[0].shape[3] / 2.0))])

        for intLevel in range(len(tenOne)):
            tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear',
                                                           align_corners=True) * 2.0

            if tenUpsampled.shape[2] != tenOne[intLevel].shape[2]: tenUpsampled = torch.nn.functional.pad(
                input=tenUpsampled, pad=[0, 0, 0, 1], mode='replicate')
            if tenUpsampled.shape[3] != tenOne[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(
                input=tenUpsampled, pad=[0, 1, 0, 0], mode='replicate')

            tenFlow = self.netBasic[intLevel](
                torch.cat([tenOne[intLevel], backwarp(tenInput=tenTwo[intLevel], tenFlow=tenUpsampled), tenUpsampled],
                          1)) + tenUpsampled
        return tenFlow

def torch_warp(tensorInput, tensorFlow):
    if tensorInput.device == torch.device('cpu'):
        if str(tensorFlow.size()) not in Backward_tensorGrid_cpu:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            Backward_tensorGrid_cpu[str(tensorFlow.size())] = torch.cat(
                [tensorHorizontal, tensorVertical], 1).cpu()

        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                                tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

        grid = (Backward_tensorGrid_cpu[str(tensorFlow.size())] + tensorFlow)
        return torch.nn.functional.grid_sample(input=tensorInput,
                                               grid=grid.permute(0, 2, 3, 1),
                                               mode='bilinear',
                                               padding_mode='border',
                                               align_corners=True)
    else:
        device_id = tensorInput.device.index
        if str(tensorFlow.size()) not in Backward_tensorGrid[device_id]:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            Backward_tensorGrid[device_id][str(tensorFlow.size())] = torch.cat(
                [tensorHorizontal, tensorVertical], 1).cuda().to(device_id)

        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                                tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

        grid = (Backward_tensorGrid[device_id][str(tensorFlow.size())] + tensorFlow)
        return torch.nn.functional.grid_sample(input=tensorInput,
                                               grid=grid.permute(0, 2, 3, 1),
                                               mode='bilinear',
                                               padding_mode='border',
                                               align_corners=True)

def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)
    return outfeature

class MEBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x

class ME_Spynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = 4
        self.moduleBasic = torch.nn.ModuleList([MEBasic() for _ in range(self.L)])

    def forward(self, im1, im2):
        batchsize = im1.size()[0]
        im1_pre = im1
        im2_pre = im2

        im1_list = [im1_pre]
        im2_list = [im2_pre]
        for level in range(self.L - 1):
            im1_list.append(F.avg_pool2d(im1_list[level], kernel_size=2, stride=2))
            im2_list.append(F.avg_pool2d(im2_list[level], kernel_size=2, stride=2))

        shape_fine = im2_list[self.L - 1].size()
        zero_shape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        flow = torch.zeros(zero_shape, dtype=im1.dtype, device=im1.device)
        for level in range(self.L):
            flow_up = bilinearupsacling(flow) * 2.0
            img_index = self.L - 1 - level
            flow = flow_up + \
                   self.moduleBasic[level](torch.cat([im1_list[img_index],
                                                      torch_warp(im2_list[img_index], flow_up),
                                                      flow_up], 1))

        return flow

