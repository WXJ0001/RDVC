import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.active_out_channel = None

    def forward(self, *args):
        raise NotImplementedError

    def initialize(self):
        for m in self.modules():
            pass

    @property
    def module_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

def set_exist_attr(m, attr, value):
    if hasattr(m, attr):
        setattr(m, attr, value)

def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

#取一半
def cut_list(in_list):
    new_list = []
    for item in in_list:
        new_list.append(int(item / 2))
    return new_list

#加某数
def cut_list0(in_list,add):
    new_list = []
    for item in in_list:
        new_list.append(int(item + add))
    return new_list

#乘2
def cut_list01(in_list):
    new_list = []
    for item in in_list:
        new_list.append(int(item * 2))
    return new_list

#除某数
def cut_list02(in_list, num):
    new_list = []
    for item in in_list:
        new_list.append(int(item // num))
    return new_list

#乘某数
def cut_list03(in_list, num):
    new_list = []
    for item in in_list:
        new_list.append(int(item * num))
    return new_list


#essential component
class DSConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels_list,  # 输入通道数量的动态变化列表
                 out_channels_list,  # 输出通道数量的动态变化列表
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 downsize=1,
                 upsize=1,
                 fix_channel=False):
        if not isinstance(in_channels_list, (list, tuple)):
            in_channels_list = [in_channels_list]
        if not isinstance(out_channels_list, (list, tuple)):
            out_channels_list = [out_channels_list]  # 转成list or tuple 的数据格式
        super(DSConv2d, self).__init__(
            in_channels=in_channels_list[-1],
            out_channels=out_channels_list[-1],
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)  # 对继承的nn.conv2d 使用输入的参数进行初始化~
        assert self.groups in (1, self.out_channels), \
            'only support regular conv, pwconv and dwconv'
        padding = ((self.stride[0] - 1) + self.dilation[0] * (
                self.kernel_size[0] - 1)) // 2  # 计算same padding 应该填充的宽度
        self.padding = (padding, padding)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.active_out_channel = out_channels_list[-1]
        self.downsize = downsize
        self.upsize = upsize
        self.fix_channel = fix_channel
        self.active_out_channel1 = self.active_out_channel

    def forward(self, x):
        self.running_inc = x.size(1)
        self.active_out_channel1 = self.active_out_channel
        if self.downsize != 1:
            self.active_out_channel1 = int(self.active_out_channel1 // self.downsize)
        elif self.upsize != 1:
            self.active_out_channel1 = int(self.active_out_channel1 * self.upsize)
        elif self.fix_channel:
            self.active_out_channel1 = self.out_channels_list[-1]
        self.running_outc = self.active_out_channel1
        weight = self.weight[:self.running_outc, :self.running_inc]
        bias = self.bias[:self.running_outc] if self.bias is not None else None
        self.running_groups = 1 if self.groups == 1 else self.running_outc
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.running_groups)

class SubpelDSConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels_list,  # 输入通道数量的动态变化列表
                 out_channels_list,  # 输出通道数量的动态变化列表
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 downsize=1,
                 r=1):
        if not isinstance(in_channels_list, (list, tuple)):
            in_channels_list = [in_channels_list]
        if not isinstance(out_channels_list, (list, tuple)):
            out_channels_list = [out_channels_list]  # 转成list or tuple 的数据格式
        super(SubpelDSConv2d, self).__init__(
            in_channels=in_channels_list[-1],
            out_channels=out_channels_list[-1] * r ** 2,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)  # 对继承的nn.conv2d 使用输入的参数进行初始化~
        assert self.groups in (1, self.out_channels), \
            'only support regular conv, pwconv and dwconv'
        padding = ((self.stride[0] - 1) + self.dilation[0] * (
                self.kernel_size[0] - 1)) // 2  # 计算same padding 应该填充的宽度
        self.padding = (padding, padding)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.active_out_channel = out_channels_list[-1]
        self.sub_activate_out_channel = self.active_out_channel * r ** 2
        self.downsize = downsize
        self.r = r

    def forward(self, x):
        self.running_inc = x.size(1)
        self.running_outc = self.active_out_channel * self.r ** 2
        weight = self.weight[:self.running_outc, :self.running_inc]
        bias = self.bias[:self.running_outc] if self.bias is not None else None
        self.running_groups = 1 if self.groups == 1 else self.running_outc
        return F.conv2d(x,
                        weight,
                        bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.running_groups)
