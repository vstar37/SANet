import torch
import torch.nn as nn
import torch.nn.functional as F



class SpatialAttention(nn.Module):
    def __init__(self, in_channel):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (1,2880,8,8)
        MaxPool = torch.max(x, dim=1).values
        AvgPool = torch.mean(x, dim=1)

        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  #(1,2,8,8)

        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        x = Ms * x + x

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channel):
        super(ChannelAttention, self).__init__()
        self.conv1 = BasicConv2d(in_channel*in_channel, in_channel, 3, 1, 1)
        self.conv2 = BasicConv2d(in_channel, in_channel, 3, 1, 1)
        self.conv3 = BasicConv2d(in_channel, in_channel, 3, 1, 1)
        self.conv4 = BasicConv2d(in_channel, in_channel, 3, 1, 1)


    def forward(self, x1, x2):
        N, C, W, H = x1.shape
        x0 = x1
        x0_2 = x2
        x1_ = x1.permute(0, 2, 3, 1)     # (22,22,64)
        x2_ = x2.permute(0, 2, 3, 1)     # (22,22,64)
        x1_ = x1_.reshape(N * H * W, C)  # (22*22,64)
        x2_ = x2_.reshape(N * H * W, C)  # (22*22,64)
        a = x1_.unsqueeze(-1)    #  (22*22,64,1)
        b = x2_.unsqueeze(-2)    #  (22*22,1,64)
        c = torch.bmm(a, b)      # (22*22,64,64)
        c = c.reshape(N, W, H, -1) # (22,22,64*64)
        c = c.permute(0, 3, 1, 2)  # (64*64,22,22)
        c_ = self.conv1(c)         # (64,22,22)
        alpha = self.conv2(c_)     # (64,22,22)
        beta = self.conv3(c_)      # (64,22,22)
        ret = alpha * x0 + beta + x0_2
        return ret


# Mixed-resolution attention: use high-resolution Q to compute attention with low-resolution K, then weight low-resolution V to enhance high-resolution features.
class MixResAttention(nn.Module):
    def __init__(self, high_res_channel, low_res_channel):  # 576, 1152
        super(MixResAttention, self).__init__()
        self.high_res_channel = high_res_channel
        self.low_res_channel = low_res_channel
        # Use depthwise convolution to extract Q, K, V with kernel_size=3 and padding=1.
        self.dwConv_Q = nn.Conv2d(high_res_channel, high_res_channel, 3, padding=1, groups=high_res_channel)
        self.dwConv_K = nn.Conv2d(low_res_channel, low_res_channel, 3, padding=1, groups=low_res_channel)
        self.dwConv_V = nn.Conv2d(low_res_channel, low_res_channel, 3, padding=1, groups=low_res_channel)

        # Pointwise convolution to adjust channel dimensions.
        self.pwConv_Q = nn.Conv2d(high_res_channel, high_res_channel, 1)
        self.pwConv_K = nn.Conv2d(low_res_channel, low_res_channel, 1)
        self.pwConv_V = nn.Conv2d(low_res_channel, low_res_channel, 1)

        self.high_to_low = BasicConv2d(high_res_channel, low_res_channel, 1)

    def forward(self, high_res_x, low_res_x):

        # Get spatial dimensions of high_res_x.
        _, high_res_channel, high_res_height, high_res_width = high_res_x.size()
        batch_size, low_res_channel, low_res_height, low_res_width = low_res_x.size()

        # Depthwise convolution.
        Q = self.dwConv_Q(high_res_x)
        K = self.dwConv_K(low_res_x)
        V = self.dwConv_V(low_res_x)

        # Pointwise convolution.
        Q = self.pwConv_Q(Q)
        K = self.pwConv_K(K)
        V = self.pwConv_V(V)

        Q = Q.view(batch_size, high_res_channel, -1)
        K = F.interpolate(K, size=(high_res_height, high_res_height), mode='bilinear', align_corners=False)
        K = K.view(batch_size, low_res_channel, -1)
        V = V.view(batch_size, low_res_channel, -1)
        K = K.transpose(1, 2)
        attention_scores = torch.matmul(Q, K) / (low_res_channel ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention_probs, V)
        out = out.view(batch_size, high_res_channel, low_res_height, low_res_width)
        out = self.high_to_low(out)
        return out



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x