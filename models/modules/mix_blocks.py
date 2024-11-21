import torch
import torch.nn as nn


class ScMixBlk(nn.Module):
    # Partial Decoder Component (Search Module)
    def __init__(self, channel):
        super(ScMixBlk, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_rc_x1 = BasicConv2d(channel[1], channel[1], 3, padding=1)
        self.conv_rc_x1_reshape = nn.Conv2d(channel[1],channel[1]//2,1)
        self.conv_upsample1 = nn.Sequential(
            BasicConv2d(channel[2], channel[2], 3, padding=1),
            nn.Conv2d(channel[2], channel[1] // 2, 1)
        )  
        self.conv_upsample2 = nn.Sequential(
            BasicConv2d(channel[3], channel[3], 3, padding=1),
            nn.Conv2d(channel[3], channel[1] // 2, 1)
        ) 
        self.conv_upsample3 = nn.Sequential(
            BasicConv2d(channel[2], channel[2], 3, padding=1),
            nn.Conv2d(channel[2], channel[1] // 2, 1)
        ) 
        self.conv_upsample5 = nn.Sequential(
            BasicConv2d(channel[4], channel[4], 3, padding=1),
            nn.Conv2d(channel[4], channel[1] // 2, 1)
        ) 

        self.conv_concat2 = BasicConv2d(channel[1], channel[1], 3, padding=1)
        self.conv_concat3 = BasicConv2d(2 * channel[1], 2 * channel[1], 3, padding=1)
        self.conv4 = BasicConv2d(2 * channel[1], 2 * channel[1], 3, padding=1)
        self.conv_out = nn.Conv2d(2 * channel[1], channel[1],1)

    def forward(self, x1, x2, x3, x4):

        x1_1 = self.conv_rc_x1_reshape(self.conv_rc_x1(x1))
        x2_1 = self.conv_upsample1(self.upsample(x2)) * x1_1
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x3))) * self.conv_upsample3(self.upsample(x2)) * x1_1
        x2_2 = torch.cat((x2_1, x1_1), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x2_2, self.conv_upsample5(self.upsample(self.upsample(x4))), x3_1), 1)
        x3_2 = self.conv_concat3(x3_2)
        x3_2 = self.conv4(x3_2)
        x3_2 = self.conv_out(x3_2) 
        return x3_2, x2_2, x3_1


'''
class PDC_SM(nn.Module):
    # Partial Decoder Component (Search Module)
    def __init__(self, channel):
        super(PDC_SM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_rc_x1 = BasicConv2d(channel[1], channel[1]//2, 3, padding=1)
        self.conv_upsample1 = BasicConv2d(channel[2], channel[1]//2, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel[3], channel[1]//2, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel[2], channel[1]//2, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(channel[4], channel[1]//2, 3, padding=1)

        self.conv_concat2 = BasicConv2d(channel[1], channel[1], 3, padding=1)
        self.conv_concat3 = BasicConv2d(2 * channel[1], 2 * channel[1], 3, padding=1)
        self.conv4 = BasicConv2d(2 * channel[1], channel[1], 3, padding=1)

    def forward(self, x1, x2, x3, x4):

        x1_1 = self.conv_rc_x1(x1)
        x2_1 = self.conv_upsample1(self.upsample(x2)) * x1_1
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x3))) * self.conv_upsample3(self.upsample(x2)) * x1_1
        x2_2 = torch.cat((x2_1, x1_1), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x2_2, self.conv_upsample5(self.upsample(self.upsample(x4))), x3_1), 1)
        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        return x, x2_2, x3_1
'''

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
