import random
import torch
import torch.nn as nn
from flexible_slim_ops import BasicBlock, set_exist_attr, make_divisible, cut_list02, cut_list03, cut_list, cut_list01, \
    SubpelDSConv2d, DSConv2d
from optical_flow import torch_warp
from utils import get_net_device, get_channel


class Slimsubpel_conv3x3(BasicBlock):
    """1x1 sub-pixel convolution for up-sampling."""
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size=3, stride=1, dilation=1,
                 act_layer=None,
                 bias=True,
                 r=2,
                 downsize=1):
        super(Slimsubpel_conv3x3, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation
        self.act_func = act_layer

        # Basic 2D convolution
        self.conv = SubpelDSConv2d(in_channels_list,
                             out_channels_list,
                             kernel_size=kernel_size,
                             stride=stride,
                             dilation=(dilation, dilation),
                             bias=bias,
                             r=r,
                             downsize=downsize)
        self.pixelshuffle = nn.PixelShuffle(r)
        if act_layer == "LeakyRelu":
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act_layer == "Relu":
            self.act = nn.ReLU(inplace=True)
        elif act_layer == "GELU":
            self.act = nn.GELU()
        else:
            self.act = None
        self.active_out_channel = out_channels_list[-1]  # research_result~

    def forward(self, x):
        x = self.pixelshuffle(self.conv(x))
        if self.act is not None:
            x = self.act(x)
        return x

    def set_active_channels(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'active_out_channel', self.active_out_channel)

class Slimsubpel_conv5x5(BasicBlock):
    """5x5 sub-pixel convolution for up-sampling."""
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size=5, stride=1, dilation=1,
                 act_layer=None,
                 bias=True,
                 r=2,
                 downsize=1):
        super(Slimsubpel_conv5x5, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation
        self.act_func = act_layer

        # Basic 2D convolution
        self.conv = SubpelDSConv2d(in_channels_list,
                             out_channels_list,
                             kernel_size=kernel_size,
                             stride=stride,
                             dilation=(dilation, dilation),
                             bias=bias,
                             r=r,
                             downsize=downsize)
        self.pixelshuffle = nn.PixelShuffle(r)
        if act_layer == "LeakyRelu":
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act_layer == "Relu":
            self.act = nn.ReLU(inplace=True)
        elif act_layer == "GELU":
            self.act = nn.GELU()
        else:
            self.act = None
        self.active_out_channel = out_channels_list[-1]  # research_result~

    def forward(self, x):
        x = self.pixelshuffle(self.conv(x))
        if self.act is not None:
            x = self.act(x)
        return x

    def set_active_channels(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'active_out_channel', self.active_out_channel)

class RDVC_param(nn.Module):
    def __init__(self, in_ch=3, N=192, out_ch=3, l=3, N_channel_list=None):
        super().__init__()
        self.N = int(N)
        self.hidden_list = cut_list03(N_channel_list, l)
        self.rate_level, self.rate_interpolation_coefficient = None, None
        self.prior_rate_level, self.prior_rate_interpolation_coefficient = None, None
        self.isInterpolation = False

        self.param = nn.ModuleList([
            IBasicConv2D(in_channels_list=[in_ch], out_channels_list=self.hidden_list,
                                  kernel_size=(3, 3), stride=1, act_layer="GELU"),
            IBasicConv2D(in_channels_list=self.hidden_list, out_channels_list=self.hidden_list,
                                  kernel_size=(3, 3), stride=1, act_layer="GELU"),
            IBasicConv2D(in_channels_list=self.hidden_list, out_channels_list=self.hidden_list,
                                  kernel_size=(3, 3), stride=1, act_layer="GELU"),
            IBasicConv2D(in_channels_list=self.hidden_list, out_channels_list=[out_ch],
                                  kernel_size=(3, 3), stride=1, act_layer=None),
        ])
        self.width_list = []
        self.layer_num = 0
        for module in self.param:
            if isinstance(module, Slimsubpel_conv3x3):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1
            elif isinstance(module, IBasicConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1

    def forward(self, gaussian_params):
        for module in self.param:  # [b, 128, w//64, h//64] -> [b, 256, w//16, h//16]
            gaussian_params = module(gaussian_params)
        return gaussian_params

class RDVC_fea(nn.Module):
    def __init__(self, in_ch=3, N=192, out_ch=3, N_channel_list=None,):
        super().__init__()
        self.N = int(N)
        self.rate_level, self.rate_interpolation_coefficient = None, None
        self.prior_rate_level, self.prior_rate_interpolation_coefficient = None, None
        self.isInterpolation = False

        self.fea_enc = nn.ModuleList([
            IBasicConv2D(in_channels_list=[in_ch], out_channels_list=N_channel_list,
                                  kernel_size=(3, 3), stride=2, act_layer="GELU"),
            IBasicConv2D(in_channels_list=N_channel_list, out_channels_list=N_channel_list,
                                  kernel_size=(3, 3), stride=2, act_layer="GELU"),
            IBasicConv2D(in_channels_list=N_channel_list, out_channels_list=N_channel_list,
                                  kernel_size=(3, 3), stride=2, act_layer="GELU"),
            IBasicConv2D(in_channels_list=N_channel_list, out_channels_list=[out_ch],
                                  kernel_size=(3, 3), stride=2, act_layer=None),
        ])
        self.width_list = []
        self.layer_num = 0
        for module in self.fea_enc:
            if isinstance(module, Slimsubpel_conv3x3):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1
            elif isinstance(module, IBasicConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1

    def forward(self, gaussian_params):
        for module in self.fea_enc:  # [b, 128, w//64, h//64] -> [b, 256, w//16, h//16]
            gaussian_params = module(gaussian_params)
        return gaussian_params

class RDVC_hs(nn.Module):
    def __init__(self, in_ch=3, N=192, out_ch=3, N_channel_list=None,):
        super().__init__()
        self.N = int(N)

        self.rate_level, self.rate_interpolation_coefficient = None, None
        self.prior_rate_level, self.prior_rate_interpolation_coefficient = None, None
        self.isInterpolation = False

        self.h_s = nn.ModuleList([
            Slimsubpel_conv3x3(in_channels_list=[in_ch], out_channels_list=N_channel_list,
                                  kernel_size=3,
                                  act_layer="GELU", r=2),
            IBasicConv2D(in_channels_list=N_channel_list, out_channels_list=N_channel_list,
                                  kernel_size=(3, 3), stride=1,
                                  act_layer="GELU"),
            Slimsubpel_conv3x3(in_channels_list=N_channel_list, out_channels_list=N_channel_list,
                                  kernel_size=3,
                                  act_layer="GELU", r=2),
            IBasicConv2D(in_channels_list=N_channel_list, out_channels_list=[out_ch],
                                  kernel_size=(3, 3), stride=1,
                                  act_layer=None),
        ])
        self.width_list = []
        self.layer_num = 0
        for module in self.h_s:
            if isinstance(module, Slimsubpel_conv3x3):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1
            elif isinstance(module, IBasicConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1

    def forward(self, gaussian_params):
        for module in self.h_s:  # [b, 128, w//64, h//64] -> [b, 256, w//16, h//16]
            gaussian_params = module(gaussian_params)
        return gaussian_params

class RDVC_ha(nn.Module):
    def __init__(self, in_ch=192, N=192, out_ch=192, N_channel_list=None, ):
        super().__init__()
        self.N = int(N)

        self.h_a = nn.Sequential(
            IBasicConv2D(in_channels_list=[in_ch], out_channels_list=[N],
                         kernel_size=(3, 3), stride=(1, 1), act_layer='GELU'),
            IBasicConv2D(in_channels_list=[N], out_channels_list=[N],
                         kernel_size=(3, 3), stride=(2, 2), act_layer='GELU'),
            IBasicConv2D(in_channels_list=[N], out_channels_list=[N],
                         kernel_size=(3, 3), stride=(1, 1), act_layer='GELU'),
            IBasicConv2D(in_channels_list=[N], out_channels_list=[out_ch],
                         kernel_size=(3, 3), stride=(2, 2), act_layer=None),
        )
        self.width_list = []
        self.layer_num = 0
        for module in self.h_a:
            if isinstance(module, IBasicConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1

    def forward(self, z):
        return self.h_a(z)

class RDVC_ga(nn.Module):
    def __init__(self, in_ch=3, N=192, out_ch=192):
        super().__init__()
        self.N = int(N)
        self.g_a = nn.Sequential(
            RDVCBasicBlock(in_channels_list=[in_ch], out_channels_list=[N], kernel_size=(3, 3), stride=2, resblock=True),

            RDVCBasicBlock(in_channels_list=[N], out_channels_list=[N], kernel_size=(3, 3), stride=2, resblock=True),

            RDVCBasicBlock(in_channels_list=[N], out_channels_list=[N], kernel_size=(3, 3), stride=2, resblock=True),

            IBasicConv2D(in_channels_list=[N], out_channels_list=[out_ch], kernel_size=(5, 5), stride=2, act_layer=None),
        )
        self.width_list = []
        self.layer_num = 0
        for module in self.g_a:
            if isinstance(module, RDVCBasicBlock):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1
            elif isinstance(module, IBasicConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1

    def forward(self, x):
        return self.g_a(x)

class RDVC_gs(nn.Module):
    def __init__(self, in_ch=3, N=128, out_ch=3, N_channel_list=None):
        super().__init__()
        self.N = int(N)
        self.g_s = nn.ModuleList([
            RDVCBasicBlockup(in_channels_list=[N], out_channels_list=N_channel_list, kernel_size=(3, 3), stride=2, resblock=True),

            RDVCBasicBlockup(in_channels_list=N_channel_list, out_channels_list=N_channel_list, kernel_size=(3, 3), stride=2, resblock=True),

            RDVCBasicBlockup(in_channels_list=N_channel_list, out_channels_list=N_channel_list, kernel_size=(3, 3), stride=2, resblock=True),

            Slimsubpel_conv5x5(in_channels_list=N_channel_list, out_channels_list=[out_ch]),
        ])
        self.width_list = []
        self.layer_num = 0
        for module in self.g_s:
            if isinstance(module, Slimsubpel_conv5x5):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1
            elif isinstance(module, IBasicConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1
            elif isinstance(module, RDVCBasicBlockup):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1

    def forward(self, x_hat):
        for module in self.g_s:
            x_hat = module(x_hat)
        return x_hat

def set_channel(modules,num):
    for module in modules:
        if isinstance(module, DSConv2d):
            module.active_out_channel = num
        elif isinstance(module, DsResBlock):
            module.active_out_channel = num
            module.set_active_channels()
        elif isinstance(module, DsResBottleneckBlock):
            module.active_out_channel = num
            module.set_active_channels()
        elif isinstance(module, DsSELayer):
            module.active_out_channel = num
            module.set_active_channels()

class Gain_Module(nn.Module):
    def __init__(self, n=6, N=192, bias=False):
        """
        n: number of scales for quantization levels
        N: number of channels
        """
        super(Gain_Module, self).__init__()
        self.gain_matrix = nn.Parameter(torch.ones(size=[n, N], dtype=torch.float32), requires_grad=True)
        self.bias = bias
        self.ch = N
        self.isFlexibleRate = True if n > 1 else False
        if bias:
            self.bias = nn.Parameter(torch.ones(size=[n, N], dtype=torch.float32), requires_grad=True)

    def forward(self, x, level=None, coeff=None, isInterpolation=False):
        """  level one dim data, coeff two dims datas """
        if isinstance(level, type(x)):
            if isInterpolation:
                coeff = coeff.unsqueeze(1)
                gain1 = self.gain_matrix[level, :]
                gain2 = self.gain_matrix[level + 1, :]
                gain = ((torch.abs(gain1) ** coeff) *
                        (torch.abs(gain2) ** (1 - coeff))).unsqueeze(2).unsqueeze(3)
            else:
                gain = torch.abs(self.gain_matrix[level]).unsqueeze(2).unsqueeze(3)
        else:
            if isInterpolation:
                gain1 = self.gain_matrix[level, :]
                gain2 = self.gain_matrix[level + 1, :]
                gain = ((torch.abs(gain1) ** coeff) *
                        (torch.abs(gain2) ** (1 - coeff))).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            else:
                gain = torch.abs(self.gain_matrix[level, :]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        if self.isFlexibleRate:
            rescaled_latent = gain * x
            if self.bias:
                rescaled_latent += self.bias[level]
            return rescaled_latent
        else:
            return x

class Slimsubpel_conv1x1(BasicBlock):
    """1x1 sub-pixel convolution for up-sampling."""
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size=1, stride=1, dilation=1,
                 act_layer=None,
                 bias=True,
                 r=2,
                 downsize=1):
        super(Slimsubpel_conv1x1, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation
        self.act_func = act_layer

        # Basic 2D convolution
        self.conv = SubpelDSConv2d(in_channels_list,
                             out_channels_list,
                             kernel_size=kernel_size,
                             stride=stride,
                             dilation=(dilation, dilation),
                             bias=bias,
                             r=r,
                             downsize=downsize)
        self.pixelshuffle = nn.PixelShuffle(r)
        if act_layer == "LeakyRelu":
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act_layer == "Relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None
        self.active_out_channel = out_channels_list[-1]  # research_result~

    def forward(self, x):
        x = self.pixelshuffle(self.conv(x))
        if self.act is not None:
            x = self.act(x)
        return x

    def set_active_channels(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'active_out_channel', self.active_out_channel)

class DsResBlock(BasicBlock):
    def __init__(self, in_channels_list, slope=0.01, start_from_relu=True, end_with_relu=False, bottleneck=False, fix_channel=False):
        super(DsResBlock, self).__init__()
        self.in_channels_list = in_channels_list
        self.mid_channels_list = cut_list(in_channels_list)

        self.relu = nn.LeakyReLU(negative_slope=slope)
        if slope < 0.0001:
            self.relu = nn.ReLU()
        if bottleneck:
            self.conv1 = DSConv2d(in_channels_list, self.mid_channels_list, kernel_size=3, stride=1,
                                  dilation=(1, 1), bias=True, downsize=2, fix_channel=fix_channel)
            self.conv2 = DSConv2d(self.mid_channels_list, in_channels_list, kernel_size=3, stride=1,
                                  dilation=(1, 1), bias=True, downsize=1, fix_channel=fix_channel)
        else:
            self.conv1 = DSConv2d(in_channels_list, in_channels_list, kernel_size=3, stride=1,
                                  dilation=(1, 1), bias=True, downsize=1, fix_channel=fix_channel)
            self.conv2 = DSConv2d(in_channels_list, in_channels_list, kernel_size=3, stride=1,
                                  dilation=(1, 1), bias=True, downsize=1, fix_channel=fix_channel)

        self.first_layer = self.relu if start_from_relu else nn.Identity()
        self.last_layer = self.relu if end_with_relu else nn.Identity()

        self.active_out_channel = in_channels_list[-1]

    def forward(self, x):
        out = self.first_layer(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.last_layer(out)
        return x + out

    def set_active_channels(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'active_out_channel', self.active_out_channel)

class DsResBottleneckBlock(BasicBlock):
    def __init__(self, in_channels_list):
        super(DsResBottleneckBlock, self).__init__()
        self.in_channels_list = in_channels_list

        self.block = nn.Sequential(
            IBasicConv2D(in_channels_list=in_channels_list, out_channels_list=in_channels_list,
                         kernel_size=(1, 1), stride=(1, 1), act_layer='LeakyRelu'),
            IBasicConv2D(in_channels_list=in_channels_list, out_channels_list=in_channels_list,
                         kernel_size=(3, 3), stride=(1, 1), act_layer='LeakyRelu'),
            IBasicConv2D(in_channels_list=in_channels_list, out_channels_list=in_channels_list,
                         kernel_size=(1, 1), stride=(1, 1), act_layer=None),
        )

        self.active_out_channel = in_channels_list[-1]

    def forward(self, x):
        residul = self.block(x)
        return x + residul

    def set_active_channels(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'active_out_channel', self.active_out_channel)

class DsConvBlockResidual(BasicBlock):
    def __init__(self, ch_in_list, ch_out_list, se_layer=True):
        super().__init__()
        self.conv = nn.Sequential(
            DSConv2d(ch_in_list, ch_out_list, kernel_size=3, stride=1,
                     dilation=(1, 1), bias=True),
            nn.LeakyReLU(0.01),
            DSConv2d(ch_out_list, ch_out_list, kernel_size=3, stride=1,
                     dilation=(1, 1), bias=True),
            DsSELayer(ch_out_list) if se_layer else nn.Identity(),
        )

        self.up_dim = DSConv2d(ch_in_list, ch_out_list, kernel_size=1, stride=1,
                               dilation=(1, 1), bias=True)

        self.active_out_channel = ch_out_list[-1]

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.up_dim(x)
        return x2 + x1

    def set_active_channels(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'active_out_channel', self.active_out_channel)

class SlimUNet(BasicBlock):
    def __init__(self, in_ch_list):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.in_ch_list = in_ch_list
        mid_list = cut_list(in_ch_list) #//2
        mid_list1 = cut_list01(in_ch_list) #*2


        self.conv1 = DsConvBlockResidual(in_ch_list, mid_list)
        self.conv2 = DsConvBlockResidual(mid_list, in_ch_list)
        self.conv3 = DsConvBlockResidual(in_ch_list, mid_list1)

        self.context_refine = nn.Sequential(
            DsResBlock(mid_list1, 0),
            DsResBlock(mid_list1, 0),
            DsResBlock(mid_list1, 0),
            DsResBlock(mid_list1, 0),
        )

        self.up3 = Slimsubpel_conv1x1(mid_list1, in_ch_list, r=2)
        self.up_conv3 = DsConvBlockResidual(mid_list1, in_ch_list)

        self.up2 = Slimsubpel_conv1x1(in_ch_list, mid_list, r=2)
        self.up_conv2 = DsConvBlockResidual(in_ch_list, in_ch_list)

        self.active_out_channel = in_ch_list[-1]

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2

    def set_active_channels(self):
        for n, m in self.named_children():
            list = ['conv2', 'up3', 'up_conv3', 'up_conv2']  #不变
            list1 = ['conv1', 'up2'] #除2
            list2 = ['conv3', 'context_refine' ] #乘2
            if n in list:
                for m1 in m.modules():
                    m1.active_out_channel = self.active_out_channel

            elif n in list1:
                for m1 in m.modules():
                    m1.active_out_channel = self.active_out_channel // 2

            elif n in list2:
                for m1 in m.modules():
                    m1.active_out_channel = self.active_out_channel * 2

class RDVCBasicBlock(BasicBlock):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride, dilation=1,
                 bias=True, resblock=False):
        super(RDVCBasicBlock, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation
        self.resblock = resblock


        self.conv1 = DSConv2d(in_channels_list, out_channels_list, kernel_size=kernel_size, stride=stride,
                             dilation=(dilation, dilation), bias=bias)
        self.conv2 = DSConv2d(out_channels_list, out_channels_list, kernel_size=3, stride=1,
                              dilation=(1, 1), bias=True)
        self.mlp = nn.Sequential(
            DSConv2d(out_channels_list, out_channels_list, kernel_size=1, stride=1,
                     dilation=(1, 1), bias=True),
            nn.GELU(),
            DSConv2d(out_channels_list, out_channels_list, kernel_size=1, stride=1,
                     dilation=(1, 1), bias=True)
        )
        self.downsample = DSConv2d(in_channels_list, out_channels_list, kernel_size=1, stride=stride,
                     dilation=(1, 1), bias=True)


        if resblock:
            self.block_1 = DsResBottleneckBlock(out_channels_list)
            self.block_2 = DsResBottleneckBlock(out_channels_list)
            self.block_3 = DsResBottleneckBlock(out_channels_list)

        self.leaky_relu = nn.LeakyReLU()
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)

        self.tconv = False
        self.active_out_channel = out_channels_list[-1]  # research_result~

    def forward(self, x):
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu1(out)

        out1 = self.mlp(out)
        out = out + out1
        identity = self.downsample(x)

        out += identity

        if self.resblock:
            out = self.block_1(out)
            out = self.block_2(out)
            out = self.block_3(out)

        return out

    @property
    def config(self):
        return {
            'name': RDVCBasicBlock.__name__,
            'in_channel_list': self.in_channels_list,
            'out_channel_list': self.out_channels_list,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'act_func': None,
            'tconv': self.tconv
        }

    @staticmethod
    def build_from_config(config):
        return RDVCBasicBlock(**config)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        self.active_out_channel = self.out_channels_list[self.channel_choice]
        sub_layer = IBasicConv2D(
            in_channel, self.active_out_channel, self.kernel_size, self.stride, self.dilation, act_layer=self.act_func
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(self.conv.weight.data[:self.active_out_channel, :in_channel, :, :])
        sub_layer.conv.bias.data.copy_(self.conv.bias.data[:self.active_out_channel])
        # act parameters to be added ~
        return sub_layer

    def set_active_channels(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'active_out_channel', self.active_out_channel)

class RDVCBasicBlockup(BasicBlock):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride, dilation=1, resblock=False):
        super(RDVCBasicBlockup, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation
        self.resblock = resblock

        self.conv1 = Slimsubpel_conv3x3(in_channels_list, out_channels_list)
        self.conv2 = DSConv2d(out_channels_list, out_channels_list, kernel_size=3, stride=1,
                              dilation=(1, 1), bias=True)
        self.mlp = nn.Sequential(
            DSConv2d(out_channels_list, out_channels_list, kernel_size=1, stride=1,
                     dilation=(1, 1), bias=True),
            nn.GELU(),
            DSConv2d(out_channels_list, out_channels_list, kernel_size=1, stride=1,
                     dilation=(1, 1), bias=True)
        )

        self.upsample = Slimsubpel_conv1x1(in_channels_list, out_channels_list)

        if resblock:
            self.block_1 = DsResBottleneckBlock(out_channels_list)
            self.block_2 = DsResBottleneckBlock(out_channels_list)
            self.block_3 = DsResBottleneckBlock(out_channels_list)

        self.leaky_relu = nn.LeakyReLU()
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)

        self.tconv = True
        self.active_out_channel = out_channels_list[-1]  # research_result~

    def forward(self, x):
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu1(out)

        out1 = self.mlp(out)
        out = out + out1

        identity = self.upsample(x)

        out += identity

        if self.resblock:
            out = self.block_1(out)
            out = self.block_2(out)
            out = self.block_3(out)
        return out

    # @property
    # def config(self):
    #     return {
    #         'name': RDVCBasicBlockup.__name__,
    #         'in_channel_list': self.in_channels_list,
    #         'out_channel_list': self.out_channels_list,
    #         'kernel_size': self.kernel_size,
    #         'stride': self.stride,
    #         'act_func': None,
    #         'tconv': self.tconv
    #     }

    # @staticmethod
    # def build_from_config(config):
    #     return RDVCBasicBlockup(**config)
    #
    # def get_active_subnet(self, in_channel, preserve_weight=True):
    #     self.active_out_channel = self.out_channels_list[self.channel_choice]
    #     sub_layer = IBasicConv2D(
    #         in_channel, self.active_out_channel, self.kernel_size, self.stride, self.dilation, act_layer=self.act_func
    #     )
    #     sub_layer = sub_layer.to(get_net_device(self))
    #
    #     if not preserve_weight:
    #         return sub_layer
    #
    #     sub_layer.conv.weight.data.copy_(self.conv.weight.data[:self.active_out_channel, :in_channel, :, :])
    #     sub_layer.conv.bias.data.copy_(self.conv.bias.data[:self.active_out_channel])
    #     # act parameters to be added ~
    #     return sub_layer

    def set_active_channels(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'active_out_channel', self.active_out_channel)

class SlimMC(BasicBlock):
    def __init__(self, hidden, lower_bound=32, reduce=8):
        super(SlimMC, self).__init__()

        hidden_list = list(range(lower_bound, hidden+1, reduce))
        print("MC_hidden_channel_num: ", len(hidden_list), " channel : ", hidden_list)

        self.conv_refine = nn.Sequential(
            DSConv2d([67], hidden_list, kernel_size=3, stride=1,
                     dilation=(1, 1), bias=True),
            DsResBlock(hidden_list),
            DsResBlock(hidden_list)
        )

        self.conv_atten = nn.Sequential(
            DSConv2d(hidden_list, hidden_list, kernel_size=3, stride=1,
                     dilation=(1, 1), bias=True),
            DsSELayer(hidden_list),
            DSConv2d(hidden_list, [64], kernel_size=3, stride=1,
                    dilation=(1, 1), bias=True, fix_channel=True),
        )

        self.out_conv = DSConv2d([64], [3], kernel_size=3, stride=1,
                                 dilation=(1, 1), bias=True, fix_channel=True)

        self.lrelu = nn.LeakyReLU(True)

        self.width_list = []
        self.name_list = []
        self.layer_num = 0
        for n, module in self.named_children():
            if isinstance(module, nn.Sequential):
                self.layer_num += 1
                self.name_list.append(n)
                self.width_list.append(module[0].out_channels_list)
            elif isinstance(module, DSConv2d):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1
                self.name_list.append(n)

        self.active_out_channel = hidden_list[-1]

    def forward(self, warped, recon_mv, feature):
        fea = torch_warp(feature, recon_mv)
        x1 = self.conv_refine(torch.concat([warped, fea], dim=1))
        x2 = self.conv_atten(x1)
        x3 = self.out_conv(x2)
        out = x3 + warped

        return x2, out

class SlimRefine(BasicBlock):
    def __init__(self, in_channel=2, hidden_channel=64, out_ch=2, lower_bound=32, reduce=8):
        super().__init__()
        hidden_list = list(range(lower_bound, hidden_channel+1, reduce))
        print("refine_hidden_channel_num: ", len(hidden_list), " channel : ", hidden_list)

        self.refine = nn.Sequential(
            DSConv2d([in_channel], hidden_list, kernel_size=3, stride=1,
                     dilation=(1, 1), bias=True),
            DsResBottleneckBlock(hidden_list),
            DsResBottleneckBlock(hidden_list),
            DsResBottleneckBlock(hidden_list),
            DSConv2d(hidden_list, [out_ch], kernel_size=3, stride=1,
                     dilation=(1, 1), bias=True, fix_channel=True),
        )

        self.width_list = []
        self.name_list = []
        self.layer_num = 0
        for n, module in self.named_children():
            if isinstance(module, nn.Sequential):
                self.layer_num += 1
                self.name_list.append(n)
                self.width_list.append(module[0].out_channels_list)

        self.active_out_channel = hidden_list[-1]

    def forward(self, x, ref_frame):
        return x + self.refine(torch.cat([x, ref_frame], 1))

    def set_active_channels(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'active_out_channel', self.active_out_channel)

class SlimFusion(BasicBlock):
    def __init__(self, in_channel=2, hidden_channel=64, out_ch=2, lower_bound=32, reduce=8):
        super().__init__()
        hidden_list = list(range(lower_bound, hidden_channel+1, reduce))
        print("FeatureFusion_hidden_channel_num: ", len(hidden_list), " channel : ", hidden_list)

        self.Fusion = nn.Sequential(
            DSConv2d([in_channel], hidden_list, kernel_size=3, stride=1,
                     dilation=(1, 1), bias=True),
            DsResBlock(hidden_list),
            DsResBlock(hidden_list),
            DsResBlock(hidden_list),
            DSConv2d(hidden_list, [out_ch], kernel_size=3, stride=1,
                     dilation=(1, 1), bias=True, fix_channel=True),
        )

        self.width_list = []
        self.name_list = []
        self.layer_num = 0
        for n, module in self.named_children():
            if isinstance(module, nn.Sequential):
                self.layer_num += 1
                self.name_list.append(n)
                self.width_list.append(module[0].out_channels_list)

        self.active_out_channel = hidden_list[-1]

    def forward(self, x, ref_frame):
        return x + self.Fusion(torch.cat([x, ref_frame], 1))

    def set_active_channels(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'active_out_channel', self.active_out_channel)

class SlimFeaExt(BasicBlock):
    def __init__(self, in_ch=2, hidden_channel=64, out_ch=2, lower_bound=32, reduce=8):
        super().__init__()
        hidden_list = list(range(lower_bound, hidden_channel+1, reduce))
        print("FeatureExt_hidden_channel_num: ", len(hidden_list), " channel : ", hidden_list)

        self.conv1 = DSConv2d([in_ch], hidden_list, kernel_size=3, stride=1,
                              dilation=(1, 1), bias=True)

        self.rsb1 = nn.Sequential(
            DsResBlock(hidden_list, 0),
            DsResBlock(hidden_list, 0),
            DsResBlock(hidden_list, 0),
        )

        self.conv2 = DSConv2d(hidden_list, [out_ch], kernel_size=3, stride=1,
                              dilation=(1, 1), bias=True, fix_channel=True)

        self.width_list = []
        self.name_list = []
        self.layer_num = 0
        for n, module in self.named_children():
            if isinstance(module, nn.Sequential):
                self.layer_num += 1
                self.name_list.append(n)
                self.width_list.append(module[0].in_channels_list)
            elif isinstance(module, DSConv2d):
                self.layer_num += 1
                self.name_list.append(n)
                self.width_list.append(module.out_channels_list)

        self.active_out_channel = hidden_list[-1]

    def forward(self, x):
        x = self.conv1(x)
        res1 = x + self.rsb1(x)
        out = self.conv2(res1)
        return out

    def set_active_channels(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'active_out_channel', self.active_out_channel)

class Feature_adaptor(nn.Module):
    def __init__(self, hidden):
        super(Feature_adaptor, self).__init__()
        self.feature_adaptor_I = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.feature_adaptor_P = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.lrelu = nn.LeakyReLU(True)

    def forward(self, ref, feature):
        if feature is None:
            feature = self.lrelu(self.feature_adaptor_I(ref))
        else:
            feature = self.lrelu(self.feature_adaptor_P(feature))

        return feature

class SlimRecNet(BasicBlock):
    def __init__(self, in_ch=128, channel=64, out_ch=3, return_fea=True, lower_bound=32, reduce=8):
        super().__init__()
        self.N = in_ch
        hidden_list = list(range(lower_bound, channel+1, reduce))
        print("rec_hidden_channel_num: ", len(hidden_list), " channel : ", hidden_list)

        self.return_fea = return_fea
        self.conv1 = DSConv2d([in_ch], [64], kernel_size=3, stride=1,
                              dilation=(1, 1), bias=True, fix_channel=True)

        self.unet_1 = SlimUNet(hidden_list)
        self.unet_2 = SlimUNet(hidden_list)

        self.recon_conv1 = DSConv2d(hidden_list, [out_ch], kernel_size=3, stride=1,
                                    dilation=(1, 1), bias=True, fix_channel=True)
        self.recon_conv2 = DSConv2d(hidden_list, [out_ch], kernel_size=3, stride=1,
                                    dilation=(1, 1), bias=True, fix_channel=True)

        self.weight1 = nn.Sequential(
            DSConv2d(hidden_list, hidden_list, kernel_size=3, stride=1,
                     dilation=(1, 1), bias=True),
            DsResBlock(hidden_list),
            DSConv2d(hidden_list, [3], kernel_size=3, stride=1,
                     dilation=(1, 1), bias=True, fix_channel=True),
            nn.Sigmoid(),
        )

        self.width_list = []
        self.name_list = []
        self.layer_num = 0

        for n, module in self.named_children():
            if isinstance(module, nn.Sequential):
                self.layer_num += 1
                self.name_list.append(n)
                self.width_list.append(module[0].out_channels_list)
            elif isinstance(module, SlimUNet):
                self.layer_num += 1
                self.name_list.append(n)
                self.width_list.append(module.in_ch_list)
            elif isinstance(module, DSConv2d):
                self.layer_num += 1
                self.name_list.append(n)
                self.width_list.append(module.out_channels_list)

        self.active_out_channel = hidden_list[-1]

    def forward(self, rec_fea, mc_fea):
        feature = self.conv1(torch.cat([rec_fea, mc_fea], 1))
        feature1 = self.unet_1(feature)
        feature2 = self.unet_2(feature1)
        recon1 = self.recon_conv1(rec_fea)
        recon2 = self.recon_conv2(feature2)

        w1 = self.weight1(feature2)
        recon = w1 * recon1 + (1 - w1) * recon2
        if self.return_fea:
            return feature, recon
        else:
            return recon

class RDVCMC(nn.Module):
    def __init__(self, hidden, lower_bound=32, reduce=8):
        super(RDVCMC, self).__init__()

        self.MC_net = SlimMC(hidden, lower_bound=lower_bound, reduce=reduce)

        self.cfg_candidates = {
            'MC_net': {
                'layer_num': self.MC_net.layer_num,
                'name': self.MC_net.name_list,
                'c': self.MC_net.width_list},
        }

    def forward(self, warped, recon_mv, feature):
        fea1, out = self.MC_net(warped, recon_mv, feature)
        return fea1, out

    def sample_active_subnet(self, mode='largest', compute_flops=False):
        assert mode in ['largest', 'random', 'smallest', 'uniform', 'random_uniform']
        if mode == 'random':
            cfg = self._sample_active_subnet(min_net=False, max_net=False)
        elif mode == 'largest':
            cfg = self._sample_active_subnet(max_net=True)
        elif mode == 'smallest':
            cfg = self._sample_active_subnet(min_net=True)
        elif mode == 'uniform':
            cfg = self._sample_active_subnet(uniform=True)
        elif mode == 'random_uniform':
            cfg = self._sample_active_subnet(random_uniform=True)

        if compute_flops:
            cfg['complexity'] = self.compute_active_subnet_flops()
        return cfg

    def _sample_active_subnet(self, min_net=False, max_net=False, uniform=False, random_uniform=False, factor=None):
        sample_cfg = lambda candidates, sample_min, sample_max: \
            min(candidates) if sample_min else (max(candidates) if sample_max else random.choice(candidates))
        cfg = {}
        if not (uniform or random_uniform):
            for k in ['MC_net']:
                cfg[k] = []
                for layer_index in range(0, self.cfg_candidates[k]['layer_num']):
                    cfg[k].append(sample_cfg(self.cfg_candidates[k]["c"][layer_index], min_net, max_net))
        else:
            if random_uniform:
                factor = random.choice(self.uniform_candidates)
            for k in ['MC_net']:
                cfg[k] = []
                for layer_index in range(0, self.cfg_candidates[k]['layer_num']):  # the last layer don't partition!
                    if layer_index == self.cfg_candidates[k]['layer_num'] - 1:
                        cfg[k].append(self.cfg_candidates[k]["c"][layer_index][-1])
                    else:
                        cfg[k].append(int(self.cfg_candidates[k]["c"][layer_index][-1] * factor))
        self.set_active_subnet(cfg)
        return cfg

    def set_active_subnet(self, cfg):
        for layer_index, layer in enumerate(self.MC_net.children()):
            if isinstance(layer, nn.Sequential):
                set_channel(layer, cfg['MC_net'][layer_index])
            elif isinstance(layer, DSConv2d):
                layer.active_out_channel = cfg['MC_net'][layer_index]

    def get_active_subnet_settings(self):
        width = {}
        for k in ['MC_net']:
            width[k] = []
            for layer_index, layer in enumerate(self.MC_net.children()):
                if isinstance(layer, nn.Sequential):
                    width['MC_net'].append(get_channel(layer))
                elif isinstance(layer, DSConv2d):
                    width['MC_net'].append(get_channel(layer))

        return {"width": width}

class RDVCRecNet(nn.Module):
    def __init__(self,in_ch=128, channel=64, out_ch=3, return_fea=True, lower_bound=32, reduce=8):
        super(RDVCRecNet, self).__init__()
        self.N = in_ch
        self.out_ch = out_ch
        self.Rec_net = SlimRecNet(in_ch=in_ch, channel=channel, out_ch=out_ch, return_fea=return_fea, lower_bound=lower_bound, reduce=reduce)
        self.list_len = len(list(range(32, channel+1, 8)))
        self.cfg_candidates = {
            'Rec_net': {
                'layer_num': self.Rec_net.layer_num,
                'name': self.Rec_net.name_list,
                'c': self.Rec_net.width_list},
        }

    def forward(self, rec_fea, mc_fea):
        feature, recon = self.Rec_net(rec_fea, mc_fea)
        return feature, recon

    def sample_active_subnet(self, mode='largest', compute_flops=False, uniform_index=None):
        assert mode in ['largest', 'random', 'smallest', 'uniform']
        if mode == 'random':
            cfg = self._sample_active_subnet(min_net=False, max_net=False)
        elif mode == 'largest':
            cfg = self._sample_active_subnet(max_net=True)
        elif mode == 'smallest':
            cfg = self._sample_active_subnet(min_net=True)
        elif mode == 'uniform':
            cfg = self._sample_active_subnet(uniform=True, uniform_index=uniform_index)

        if compute_flops:
            cfg['complexity'] = self.compute_active_subnet_flops()
        return cfg

    def _sample_active_subnet(self, min_net=False, max_net=False, uniform=False, uniform_index=None):
        sample_cfg = lambda candidates, sample_min, sample_max: \
            min(candidates) if sample_min else (max(candidates) if sample_max else random.choice(candidates))
        cfg = {}
        if not uniform:
            for k in ['Rec_net']:
                cfg[k] = []
                for layer_index in range(0, self.cfg_candidates[k]['layer_num']):
                    cfg[k].append(sample_cfg(self.cfg_candidates[k]["c"][layer_index], min_net, max_net))
        else:
            selected = uniform_index if uniform_index is not None else -1
            for k in ['Rec_net']:
                cfg[k] = []
                for layer_index in range(0, self.cfg_candidates[k]['layer_num']):  # the last layer don't partition!
                    if len(self.cfg_candidates[k]["c"][layer_index]) != 1 and len(self.cfg_candidates[k]["c"][layer_index]) > selected:
                        cfg[k].append(self.cfg_candidates[k]["c"][layer_index][selected])
                    else:
                        cfg[k].append(self.cfg_candidates[k]["c"][layer_index][-1])
        self.set_active_subnet(cfg)
        return cfg

    def set_active_subnet(self, cfg):
        for layer_index, layer in enumerate(self.Rec_net.children()):
            if isinstance(layer, nn.Sequential):
                set_channel(layer, cfg['Rec_net'][layer_index])
            elif isinstance(layer, SlimUNet):
                layer.active_out_channel = cfg['Rec_net'][layer_index]
                layer.set_active_channels()
            elif isinstance(layer, DSConv2d):
                layer.active_out_channel = cfg['Rec_net'][layer_index]

    def get_active_subnet_settings(self):
        width = {}
        for k in ['Rec_net']:
            width[k] = []

            for layer_index, layer in enumerate(self.Rec_net.children()):
                if isinstance(layer, nn.Sequential):
                    width['Rec_net'].append(get_channel(layer))
                elif isinstance(layer, SlimUNet):
                    width['Rec_net'].append(get_channel(layer))
                elif isinstance(layer, DSConv2d):
                    width['Rec_net'].append(get_channel(layer))

        return {"width": width}

class RDVCRefine(nn.Module):
    def __init__(self, in_channel=2, hidden_channel=64, out_ch=2, lower_bound=32, reduce=8):
        super(RDVCRefine, self).__init__()
        self.N = in_channel
        self.out_ch = out_ch
        self.RefineNet = SlimRefine(in_channel=in_channel, hidden_channel=hidden_channel, out_ch=out_ch, lower_bound=lower_bound, reduce=reduce)

        self.list_len = len(list(range(32, hidden_channel+1, 8)))
        self.cfg_candidates = {
            'Refine': {
                'layer_num': self.RefineNet.layer_num,
                'name': self.RefineNet.name_list,
                'c': self.RefineNet.width_list},
        }

    def forward(self, x, ref_frame):
        out = self.RefineNet(x, ref_frame)
        return out

    def sample_active_subnet(self, mode='largest', compute_flops=False, uniform_index=None):
        assert mode in ['largest', 'random', 'smallest', 'uniform']
        if mode == 'random':
            cfg = self._sample_active_subnet(min_net=False, max_net=False)
        elif mode == 'largest':
            cfg = self._sample_active_subnet(max_net=True)
        elif mode == 'smallest':
            cfg = self._sample_active_subnet(min_net=True)
        elif mode == 'uniform':
            cfg = self._sample_active_subnet(uniform=True, uniform_index=uniform_index)

        if compute_flops:
            cfg['complexity'] = self.compute_active_subnet_flops()
        return cfg

    def _sample_active_subnet(self, min_net=False, max_net=False, uniform=False, uniform_index=None):
        sample_cfg = lambda candidates, sample_min, sample_max: \
            min(candidates) if sample_min else (max(candidates) if sample_max else random.choice(candidates))
        cfg = {}
        if not uniform:
            for k in ['Refine']:
                cfg[k] = []
                for layer_index in range(0, self.cfg_candidates[k]['layer_num']):
                    cfg[k].append(sample_cfg(self.cfg_candidates[k]["c"][layer_index], min_net, max_net))
        else:
            selected = uniform_index if uniform_index is not None else -1
            for k in ['Refine']:
                cfg[k] = []
                for layer_index in range(0, self.cfg_candidates[k]['layer_num']):  # the last layer don't partition!
                    if len(self.cfg_candidates[k]["c"][layer_index]) != 1 and len(self.cfg_candidates[k]["c"][layer_index]) > selected:
                        cfg[k].append(self.cfg_candidates[k]["c"][layer_index][selected])
                    else:
                        cfg[k].append(self.cfg_candidates[k]["c"][layer_index][-1])
        self.set_active_subnet(cfg)
        return cfg

    def set_active_subnet(self, cfg):
        for layer_index, layer in enumerate(self.RefineNet.children()):
            if isinstance(layer, nn.Sequential):
                set_channel(layer, cfg['Refine'][layer_index])

    def get_active_subnet_settings(self):
        width = {}
        for k in ['Refine']:
            width[k] = []
            for layer_index, layer in enumerate(self.RefineNet.children()):
                if isinstance(layer, nn.Sequential):
                    width['Refine'].append(get_channel(layer))
        return {"width": width}

class RDVCFusion(nn.Module):
    def __init__(self, in_channel=128, hidden_channel=64, out_ch=64, lower_bound=32, reduce=8):
        super(RDVCFusion, self).__init__()
        self.N = in_channel
        self.out_ch = out_ch
        self.FusionNet = SlimFusion(in_channel=in_channel, hidden_channel=hidden_channel, out_ch=out_ch, lower_bound=lower_bound, reduce=reduce)

        self.list_len = len(list(range(32, hidden_channel+1, 8)))
        self.cfg_candidates = {
            'Fusion': {
                'layer_num': self.FusionNet.layer_num,
                'name': self.FusionNet.name_list,
                'c': self.FusionNet.width_list},
        }

    def forward(self, warp_fea, feature):
        out = self.FusionNet(warp_fea, feature)
        return out

    def sample_active_subnet(self, mode='largest', compute_flops=False, uniform_index=None):
        assert mode in ['largest', 'random', 'smallest', 'uniform']
        if mode == 'random':
            cfg = self._sample_active_subnet(min_net=False, max_net=False)
        elif mode == 'largest':
            cfg = self._sample_active_subnet(max_net=True)
        elif mode == 'smallest':
            cfg = self._sample_active_subnet(min_net=True)
        elif mode == 'uniform':
            cfg = self._sample_active_subnet(uniform=True, uniform_index=uniform_index)

        if compute_flops:
            cfg['complexity'] = self.compute_active_subnet_flops()
        return cfg

    def _sample_active_subnet(self, min_net=False, max_net=False, uniform=False, uniform_index=None):
        sample_cfg = lambda candidates, sample_min, sample_max: \
            min(candidates) if sample_min else (max(candidates) if sample_max else random.choice(candidates))
        cfg = {}
        if not uniform:
            for k in ['Fusion']:
                cfg[k] = []
                for layer_index in range(0, self.cfg_candidates[k]['layer_num']):
                    cfg[k].append(sample_cfg(self.cfg_candidates[k]["c"][layer_index], min_net, max_net))
        else:
            selected = uniform_index if uniform_index is not None else -1
            for k in ['Fusion']:
                cfg[k] = []
                for layer_index in range(0, self.cfg_candidates[k]['layer_num']):  # the last layer don't partition!
                    if len(self.cfg_candidates[k]["c"][layer_index]) != 1 and len(self.cfg_candidates[k]["c"][layer_index]) > selected:
                        cfg[k].append(self.cfg_candidates[k]["c"][layer_index][selected])
                    else:
                        cfg[k].append(self.cfg_candidates[k]["c"][layer_index][-1])
        self.set_active_subnet(cfg)
        return cfg

    def set_active_subnet(self, cfg):
        for layer_index, layer in enumerate(self.FusionNet.children()):
            if isinstance(layer, nn.Sequential):
                set_channel(layer, cfg['Fusion'][layer_index])

    def get_active_subnet_settings(self):
        width = {}
        for k in ['Fusion']:
            width[k] = []
            for layer_index, layer in enumerate(self.FusionNet.children()):
                if isinstance(layer, nn.Sequential):
                    width['Fusion'].append(get_channel(layer))
        return {"width": width}

class RDVCFeaExt(nn.Module):
    def __init__(self, in_ch=128, hidden_channel=64, out_ch=64, lower_bound=32, reduce=8):
        super(RDVCFeaExt, self).__init__()
        self.N = in_ch
        self.out_ch = out_ch
        self.FeaExt = SlimFeaExt(in_ch=in_ch, hidden_channel=hidden_channel, out_ch=out_ch, lower_bound=lower_bound, reduce=reduce)

        self.list_len = len(list(range(32, hidden_channel+1, 8)))
        self.cfg_candidates = {
            'FeaExt': {
                'layer_num': self.FeaExt.layer_num,
                'name': self.FeaExt.name_list,
                'c': self.FeaExt.width_list},
        }

    def forward(self, x):
        out = self.FeaExt(x)
        return out

    def sample_active_subnet(self, mode='largest', compute_flops=False, uniform_index=None):
        assert mode in ['largest', 'random', 'smallest', 'uniform']
        if mode == 'random':
            cfg = self._sample_active_subnet(min_net=False, max_net=False)
        elif mode == 'largest':
            cfg = self._sample_active_subnet(max_net=True)
        elif mode == 'smallest':
            cfg = self._sample_active_subnet(min_net=True)
        elif mode == 'uniform':
            cfg = self._sample_active_subnet(uniform=True, uniform_index=uniform_index)

        if compute_flops:
            cfg['complexity'] = self.compute_active_subnet_flops()
        return cfg

    def _sample_active_subnet(self, min_net=False, max_net=False, uniform=False, uniform_index=None):
        sample_cfg = lambda candidates, sample_min, sample_max: \
            min(candidates) if sample_min else (max(candidates) if sample_max else random.choice(candidates))
        cfg = {}
        if not uniform:
            for k in ['FeaExt']:
                cfg[k] = []
                for layer_index in range(0, self.cfg_candidates[k]['layer_num']):
                    cfg[k].append(sample_cfg(self.cfg_candidates[k]["c"][layer_index], min_net, max_net))
        else:
            selected = uniform_index if uniform_index is not None else -1
            for k in ['FeaExt']:
                cfg[k] = []
                for layer_index in range(0, self.cfg_candidates[k]['layer_num']):  # the last layer don't partition!
                    if len(self.cfg_candidates[k]["c"][layer_index]) != 1 and len(self.cfg_candidates[k]["c"][layer_index]) > selected:
                        cfg[k].append(self.cfg_candidates[k]["c"][layer_index][selected])
                    else:
                        cfg[k].append(self.cfg_candidates[k]["c"][layer_index][-1])
        self.set_active_subnet(cfg)
        return cfg

    def set_active_subnet(self, cfg):
        for layer_index, layer in enumerate(self.FeaExt.children()):
            if isinstance(layer, nn.Sequential):
                set_channel(layer, cfg['FeaExt'][0])
            elif isinstance(layer, DSConv2d):
                layer.active_out_channel = cfg['FeaExt'][0]

    def get_active_subnet_settings(self):
        width = {}
        for k in ['FeaExt']:
            width[k] = []
            for layer_index, layer in enumerate(self.FeaExt.children()):
                if isinstance(layer, nn.Sequential):
                    width['FeaExt'].append(get_channel(layer))
                elif isinstance(layer, DSConv2d):
                    width['FeaExt'].append(get_channel(layer))
        return {"width": width}

class Slim_SE_Block1(BasicBlock):
    def __init__(self, in_channels_list, out_channels_list, se_ratio=0.25, divisor=1, downsize=1, upsize=1):
        super(Slim_SE_Block1, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        reduced_chs = make_divisible((out_channels_list[-1]) * se_ratio, divisor)
        self.fc1 = DSConv2d(in_channels_list, [reduced_chs], kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1),
                            bias=True, downsize=downsize, upsize=upsize)
        self.fc2 = DSConv2d([reduced_chs], out_channels_list, kernel_size=(1, 1), stride=(1, 1), dilation=(1, 1),
                            bias=True, downsize=downsize, upsize=upsize)
        self.attn_act = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

        self.initialize()
        self.active_out_channel = out_channels_list[-1]  # research_result~

    def forward(self, x):
        # 读取批数据图片数量及通道数
        self.set_active_channels()
        y = self.gap(x)
        y = self.relu(self.fc1(y))
        y = self.fc2(y)
        attn = self.attn_act(y)

        return x * attn

    def set_active_channels(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'active_out_channel', self.active_out_channel)

class DsSELayer(BasicBlock):
    def __init__(self, in_channels_list, reduction=8):
        super(DsSELayer, self).__init__()
        hidden_list = cut_list02(in_channels_list, reduction)
        out_channels_list = in_channels_list

        self.fc = nn.Sequential(
            DSConv2d(in_channels_list, hidden_list, kernel_size=(1, 1), stride=(1, 1),
                     dilation=(1, 1), bias=True, downsize=reduction, upsize=1),
            nn.ReLU(inplace=True),
            DSConv2d(hidden_list, out_channels_list, kernel_size=(1, 1), stride=(1, 1),
                     dilation=(1, 1), bias=True, downsize=1, upsize=1),
            nn.Sigmoid()
        )

        self.active_out_channel = out_channels_list[-1]

    def forward(self, x):
        y = torch.mean(x, dim=(-1, -2))
        y = self.fc(y[:, :, None, None])
        return x * y

    def set_active_channels(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'active_out_channel', self.active_out_channel)

class IBasicConv2D(BasicBlock):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride, dilation=1,
                 act_layer=None,
                 bias=True, fix_channel=False
                 ):
        super(IBasicConv2D, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation
        self.act_func = act_layer

        # Basic 2D convolution
        self.conv = DSConv2d(in_channels_list,
                             out_channels_list,
                             kernel_size=kernel_size,
                             stride=stride,
                             dilation=(dilation, dilation),
                             bias=bias, fix_channel=fix_channel)
        if act_layer == "LeakyRelu":
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act_layer == "Relu":
            self.act = nn.ReLU(inplace=True)
        elif act_layer == "GELU":
            self.act = nn.GELU()
        else:
            self.act = None
        self.active_out_channel = out_channels_list[-1]  # research_result~

    def forward(self, x):
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x

    def get_active_subnet(self, in_channel, preserve_weight=True):
        self.active_out_channel = self.out_channels_list[self.channel_choice]
        sub_layer = IBasicConv2D(
            in_channel, self.active_out_channel, self.kernel_size, self.stride, self.dilation, act_layer=self.act_func
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(self.conv.weight.data[:self.active_out_channel, :in_channel, :, :])
        sub_layer.conv.bias.data.copy_(self.conv.bias.data[:self.active_out_channel])
        # act parameters to be added ~
        return sub_layer

    def set_active_channels(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'active_out_channel', self.active_out_channel)

class ResBlock(nn.Module):
    def __init__(self, channel, slope=0.01, start_from_relu=True, end_with_relu=False,
                 bottleneck=False):
        super().__init__()
        self.relu = nn.LeakyReLU(negative_slope=slope)
        if slope < 0.0001:
            self.relu = nn.ReLU()
        if bottleneck:
            self.conv1 = nn.Conv2d(channel, channel // 2, 3, padding=1)
            self.conv2 = nn.Conv2d(channel // 2, channel, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
            self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)
        self.first_layer = self.relu if start_from_relu else nn.Identity()
        self.last_layer = self.relu if end_with_relu else nn.Identity()

    def forward(self, x):
        out = self.first_layer(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.last_layer(out)
        return x + out

class ResBottleneckBlock(nn.Module):
    def __init__(self, channel, slope=0.01):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 1, 1, padding=0)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(channel, channel, 1, 1, padding=0)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if slope < 0.0001:
            self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        return x + out

class GainModule(nn.Module):
    def __init__(self, n=3, N=128):
        super(GainModule, self).__init__()
        self.gain_matrix = nn.Parameter(torch.ones(n, N))

    def forward(self, x, n=None, l=1):
        if l != 1:
            gain1 = self.gain_matrix[n]
            gain2 = self.gain_matrix[n + 1]
            gain = (torch.abs(gain1) ** l) * (torch.abs(gain2) ** (1 - l))
            gain = gain.squeeze(0)
            # print(11, gain.shape)
            # exit()
        else:
            gain = torch.abs(self.gain_matrix[n])
            # print(22, gain.shape)
            # exit()
            # reshaped_gain = gain.unsqueeze(2).unsqueeze(3)

        reshaped_gain = gain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # print(x.shape, reshaped_gain.shape, gain.shape)
        # exit()
        rescaled_latent = reshaped_gain * x
        return rescaled_latent

class GainModule0(nn.Module):
    def __init__(self, n=3, N=128):
        super(GainModule0, self).__init__()
        self.gain_matrix = nn.Parameter(torch.ones(n, N))

    def forward(self, x, n=None, l=1):
        if l != 1:
            gain1 = self.gain_matrix[n]
            gain2 = self.gain_matrix[n[0] + 1]
            gain = (torch.abs(gain1) ** l) * (torch.abs(gain2) ** (1 - l))

        else:
            gain = torch.abs(self.gain_matrix[n])

        reshaped_gain = gain.unsqueeze(2).unsqueeze(3)
        rescaled_latent = reshaped_gain * x
        return rescaled_latent