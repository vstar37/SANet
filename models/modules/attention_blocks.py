import torch
import torch.nn as nn
import torch.nn.functional as F



class SpatialAttention(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # (1,2880,8,8)
        # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]  #(1,8,8)
        MaxPool = torch.max(x, dim=1).values  # torch.max 返回的是索引和value， 要用.values去访问值才行！
        AvgPool = torch.mean(x, dim=1)

        # 增加维度, 变成 [N, 1, H, W]
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)

        # 维度拼接 [N, 2, H, W]
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)  #(1,2,8,8)

        # 卷积操作得到空间注意力结果
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        # 与原图通道进行乘积
        x = Ms * x + x

        return x

#待修改
class ChannelAttention(nn.Module):
    def __init__(self, in_channel):
        super(ChannelAttention, self).__init__()
        self.conv1 = BasicConv2d(in_channel*in_channel, in_channel, 3, 1, 1)
        self.conv2 = BasicConv2d(in_channel, in_channel, 3, 1, 1)
        self.conv3 = BasicConv2d(in_channel, in_channel, 3, 1, 1)
        self.conv4 = BasicConv2d(in_channel, in_channel, 3, 1, 1)


    def forward(self, x1, x2):  # 前面是较浅层特征，后面是修正后的较深层特征(64,22,22) & (64,22,22)
        N, C, W, H = x1.shape
        x0 = x1
        x0_2 = x2
        #调整tensor维度
        x1_ = x1.permute(0, 2, 3, 1)     # (22,22,64)
        x2_ = x2.permute(0, 2, 3, 1)     # (22,22,64)
        x1_ = x1_.reshape(N * H * W, C)  # (22*22,64)
        x2_ = x2_.reshape(N * H * W, C)  # (22*22,64)
        a = x1_.unsqueeze(-1)    # 在最后一维插入一维，变成 (22*22,64,1)
        b = x2_.unsqueeze(-2)    # 在倒数第二维插入一维，变成 (22*22,1,64)
        c = torch.bmm(a, b)      # (22*22,64,64)
        c = c.reshape(N, W, H, -1) # (22,22,64*64) ，描述通道相似度
        c = c.permute(0, 3, 1, 2)  # (64*64,22,22)
        c_ = self.conv1(c)         # (64,22,22)
        alpha = self.conv2(c_)     # (64,22,22)
        beta = self.conv3(c_)      # (64,22,22)
        # x_ = self.conv4(x0)
        ret = alpha * x0 + beta + x0_2
        return ret


# 混合分辨率注意力，目的是通过高分辨率Q，计算其和低分辨率图K的关联矩阵， 加权给 低分辨率图V，以增强其高分辨率信息。
class MixResAttention(nn.Module):
    def __init__(self, high_res_channel, low_res_channel):  # 576, 1152
        super(MixResAttention, self).__init__()
        self.high_res_channel = high_res_channel
        self.low_res_channel = low_res_channel
        # 使用深度卷积来提取 Q, K, V，kernel_size=3，padding=1
        self.dwConv_Q = nn.Conv2d(high_res_channel, high_res_channel, 3, padding=1, groups=high_res_channel)
        self.dwConv_K = nn.Conv2d(low_res_channel, low_res_channel, 3, padding=1, groups=low_res_channel)
        self.dwConv_V = nn.Conv2d(low_res_channel, low_res_channel, 3, padding=1, groups=low_res_channel)

        # 逐点卷积用于调整通道维度
        self.pwConv_Q = nn.Conv2d(high_res_channel, high_res_channel, 1)
        self.pwConv_K = nn.Conv2d(low_res_channel, low_res_channel, 1)
        self.pwConv_V = nn.Conv2d(low_res_channel, low_res_channel, 1)

        self.high_to_low = BasicConv2d(high_res_channel, low_res_channel, 1)

    def forward(self, high_res_x, low_res_x):

        # 获取 high_res_x 的空间尺寸
        _, high_res_channel, high_res_height, high_res_width = high_res_x.size()
        batch_size, low_res_channel, low_res_height, low_res_width = low_res_x.size()

        # 深度卷积
        Q = self.dwConv_Q(high_res_x)
        K = self.dwConv_K(low_res_x)
        V = self.dwConv_V(low_res_x)

        # 逐点卷积
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

        # 这里的 attention_probs 矩阵，形状是[1, high_res_channel, low_res_channel] 其中每一行，都代表着高分辨率特征图中 \
        # 一个 channel 对 低分辨率特征图中所有 channel 的相关性权重。
        # 接下来，用 “高分辨率特征图中一个 channel 对 低分辨率特征图中所有 channel 的相关性权重。” 加权给 V 的列向量。
        # 也就是 V(低分辨率特征图) 中 每个channel的 第一个元素上
        # 这里是拼接成 矩阵 的 向量运算。attention_probs中的行 其实是拼接起来的 high_res_channel 与 low_res_channel 的关联性
        # 做矩阵乘法，那么此时实际上是在给V的一列，也就是V中每个channel的第一个元素加权。这合理吗？
        # 这里是逐元素乘法，第一行*第一列：(c1,c1) 加权给 V中 第一个 通道的第一个元素; (c1,c2) 加权给 V中 第二个通道的第一个元素......

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