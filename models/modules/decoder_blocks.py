import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from config import Config

config = Config()


class BasicDecBlk(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicDecBlk, self).__init__()
        inter_channels = 64
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)

        x = self.conv_out(x)
        x = self.bn_out(x)
        return x


class BasicDecBlk_lite(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicDecBlk_lite, self).__init__()
        inter_channels = 64

        # Use depthwise separable convolution instead of standard convolution
        self.depthwise_conv_in = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                                           groups=in_channels)
        self.pointwise_conv_in = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.relu_in = nn.ReLU(inplace=True)

        self.depthwise_conv_out = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1,
                                            groups=inter_channels)
        self.pointwise_conv_out = nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Input: Depthwise convolution + pointwise convolution + BN + ReLU
        x = self.depthwise_conv_in(x)
        x = self.pointwise_conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)

        # Output: Depthwise convolution + pointwise convolution + BN
        x = self.depthwise_conv_out(x)
        x = self.pointwise_conv_out(x)
        x = self.bn_out(x)
        return x


class LocalDecBlk(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalDecBlk, self).__init__()
        inter_channels = 64
        self.local_conv = LocalConv2d(in_channels, inter_channels, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.local_conv(x)
        x = self.conv(x)
        x = self.bn_out(x)
        return x


class ResBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=None, inter_channels=64):
        super(ResBlk, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        inter_channels = 64

        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.relu_in = nn.ReLU(inplace=True)

        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_out = nn.BatchNorm2d(out_channels)

        self.conv_resi = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        _x = self.conv_resi(x)
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)

        x = self.conv_out(x)
        x = self.bn_out(x)
        return x + _x


class RF(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel, index=0):
        super(RF, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.index = index

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x





class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class LocalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_windows=16, stride=1, padding=0):
        super(LocalConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels
        self.in_channels = in_channels

        # Initialize convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        # Batch normalization
        self.bn = nn.BatchNorm2d(out_channels)
        # Activation function
        self.relu = nn.ReLU(inplace=True)
        self.num_windows_h = int(math.sqrt(num_windows))
        self.num_windows_w = int(math.ceil(num_windows / self.num_windows_h))

    def forward(self, x):
        b, c, h, w = x.size()
        # Set the number of patch windows to 4x4, so each window size is h//4 * w//4
        window_h, window_w = h // self.num_windows_h, w // self.num_windows_w  # Compute the size of each window

        # Ensure input dimensions are divisible by the number of windows
        assert h % self.num_windows_h == 0 and w % self.num_windows_w == 0, "Input dimensions must be divisible by the number of windows."

        # Divide into windows
        patches = x.unfold(2, window_h, window_h).unfold(3, window_w,
                                                         window_w)  # Shape: (b, c, num_patches_h, num_patches_w, window_h, window_w)

        # Reshape for convolution
        patches = patches.contiguous().view(b * self.num_windows_h * self.num_windows_w, c, window_h, window_w)

        # Apply convolution to each window
        conv_patches = self.conv(patches)  # Shape: (b * num_patches_h * num_patches_w, out_channels, window_h, window_w)

        # Apply batch normalization and ReLU activation
        conv_patches = self.bn(conv_patches)
        conv_patches = self.relu(conv_patches)

        # Restore shape
        conv_patches = conv_patches.view(b, self.num_windows_h, self.num_windows_w, self.out_channels, window_h,
                                         window_w)

        # Concatenate convolution results
        conv_patches = conv_patches.permute(0, 3, 1, 4, 2,
                                            5).contiguous()  # Shape: (b, out_channels, num_patches_h, window_h, num_patches_w, window_w)
        conv_patches = conv_patches.view(b, self.out_channels, h, w)

        return conv_patches
