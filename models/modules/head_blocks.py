import torch
import torch.nn as nn
from config import Config
import torch.nn.functional as F
from models.modules.decoder_blocks import RF, BasicDecBlk
from utils.utils import path_to_image, save_tensor_img, save_tensor_heatmap, save_tensor_img2

config = Config()
channels = config.channels_list


class OutHeaBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(OutHeaBlk, self).__init__()
        inter_channels = 64
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 1, 1, padding=0)
        self.relu_in = nn.ReLU(inplace=True)

        self.conv_out = nn.Conv2d(inter_channels, out_channels, 1, 1, padding=0)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)

        x = self.conv_out(x)
        x = self.bn_out(x)
        return x


class AttBeaBlk(nn.Module):
    def __init__(self, in_channels, out_channels):  # in_channels = weight map channels + feature map channels
        super(AttBeaBlk, self).__init__()
        # 注意力模块，用于融合边缘特征和分割特征
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1),  # 输出一个注意力权重
            nn.Sigmoid()  # 注意力权重归一化到 [0, 1] 之间
        )

    def forward(self, pred_e, sm):
        # 使用双线性插值将边缘特征与分割特征调整到相同尺寸
        # pred_e = F.interpolate(pred_e, size=sm.shape[2:], mode='bilinear', align_corners=False)

        # 融合边缘特征和分割特征
        combined_features = torch.cat((pred_e, sm), dim=1)
        attention_weights = self.attention(combined_features)

        # 对分割特征应用注意力权重
        pred_m = sm * attention_weights

        return pred_m


# 分辨率补充与细节优化模块v1, 对高分辨率原始图像取相应块之后做下采样，拼接到特征图上。
class ResDetHeaBlk_v1(nn.Module):
    def __init__(self, in_channels):
        super(ResDetHeaBlk_v1, self).__init__()
        DecoderBlock = eval(config.dec_blk)
        self.decoder_block = DecoderBlock(in_channels + 64 * 3, in_channels + 64 * 3)
        # self.conv_cat = BasicConv2d(in_channels+64*3, in_channels+64*3, 3, padding=1)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        # self.conv_upsample1 = BasicConv2d(in_channels + 64 * 3, in_channels + 64 * 3, 3, padding=1)
        # self.conv_upsample2 = BasicConv2d(in_channels + 64 * 3, in_channels + 64 * 3, 3, padding=1, dilation=2)
        # self.conv_upsample3 = BasicConv2d(in_channels + 64 * 3, in_channels + 64 * 3, 3, padding=1, dilation=2)
        self.conv_out1 = BasicConv2d(in_channels + 64 * 3, in_channels + 64 * 3, 3, padding=1)
        self.conv_out2 = nn.Conv2d(in_channels + 64 * 3, 1, 1)
        self.conv_pred_m = nn.Conv2d(in_channels, 1, 1)
        self.max_in_channels = in_channels + 64 * 3
        self.conv_concat = BasicConv2d(768 + 576 + 288, in_channels + 64 * 3, 3, padding=1)

    def forward(self, coarse_m, x2_2, x3_1, raw_x):

        # 补充高分辨率patch (在这个位置补充，效果有待商榷)
        coarse_pred_map = self.conv_pred_m(coarse_m)
        coarse_pred_map = coarse_pred_map.sigmoid()
        high_res_patches = self.get_patches_hi_res(coarse_m, raw_x, coarse_pred_map)
        coarse_m = self.batch_concat(coarse_m, high_res_patches)
        coarse_m = self.decoder_block(coarse_m)

        # 特征融合 x3_1 * coarse_m
        coarse_m = torch.cat((x2_2, coarse_m, x3_1), 1)
        coarse_m = self.conv_concat(coarse_m)
        coarse_m = self.conv_out1(coarse_m)
        coarse_m = self.conv_out2(coarse_m)
        coarse_m = self.upsample_8(coarse_m)
        return coarse_m

    def get_patches_hi_res(self, coarse_m, raw_x, coarse_pred_map):
        batch_size, channels_raw, height_raw, width_raw = raw_x.size()
        coarse_patches_size = coarse_pred_map.size(-1) // 8
        raw_x_patch_size = height_raw // 8
        coarse_patches = coarse_pred_map.unfold(2, coarse_patches_size, coarse_patches_size).unfold(3,
                                                                                                    coarse_patches_size,
                                                                                                    coarse_patches_size)
        coarse_patches = coarse_patches.contiguous().view(batch_size, 1, -1, coarse_patches_size, coarse_patches_size)
        raw_patches = raw_x.unfold(2, raw_x_patch_size, raw_x_patch_size).unfold(3, raw_x_patch_size, raw_x_patch_size)
        raw_patches = raw_patches.contiguous().view(batch_size, channels_raw, -1, raw_x_patch_size, raw_x_patch_size)
        foreground_patches = []
        for b in range(batch_size):
            foreground_patches_batch = []
            for i in range(coarse_patches.size(2)):
                patch = coarse_patches[b, :, i, :, :]
                if patch.max() > 0:
                    selected_patch = raw_patches[b, :, i, :, :]
                    selected_patch = selected_patch.unsqueeze(0)
                    foreground_patches_batch.append(selected_patch)
            if foreground_patches_batch:
                high_res_patches_batch = torch.cat(foreground_patches_batch, dim=0)
            else:
                high_res_patches_batch = torch.zeros(1, raw_x.size(1), coarse_m.size(-1), coarse_m.size(-1),
                                                     device=coarse_m.device)
            foreground_patches.append(high_res_patches_batch)
        return foreground_patches

    def batch_concat(self, coarse_m, high_res_patches_lists):
        batched_results = []
        for b in range(coarse_m.size(0)):
            coarse_mb = coarse_m[b].unsqueeze(0)
            for j in range(high_res_patches_lists[b].size(0)):
                high_res_patch = high_res_patches_lists[b][j].unsqueeze(0)
                coarse_mb = torch.cat([coarse_mb, high_res_patch], 1)
            batched_results.append(coarse_mb)

        # batch 遍历结束，得到 batched_results 列表，开始统一通道到最大通道
        for i in range(len(batched_results)):
            if batched_results[i].size(1) != self.max_in_channels:
                padding = torch.zeros(1, self.max_in_channels - batched_results[i].size(1), coarse_m.size(-1),
                                      coarse_m.size(-1), device=coarse_m.device)
                batched_results[i] = torch.cat([batched_results[i], padding], dim=1)

        # 将batched_results列表中元素通道数统一到max_in_channels后,做cat
        return torch.cat(batched_results, 0)


# v2
class ResDetHeaBlk_v2(nn.Module):
    def __init__(self, in_channels):
        super(ResDetHeaBlk_v2, self).__init__()
        self.max_in_channels = in_channels + 64 * 3
        DecoderBlock = eval(config.dec_blk)
        self.decoder_block1 = DecoderBlock(in_channels + 64 * 3, in_channels + 64 * 3)
        self.decoder_block2 = DecoderBlock(in_channels + 64 * 3, in_channels + 64 * 3)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.conv_pred_m1 = nn.Conv2d(in_channels, 1, 1)
        self.conv_pred_m2 = nn.Conv2d(in_channels, 1, 1)
        self.conv_concat1 = BasicConv2d(in_channels + 64 * 3 + channels[1] + channels[1]//2, in_channels + 64 * 3 + channels[1] + channels[1]//2, 3, padding=1)
        self.conv_concat2 = BasicConv2d(in_channels + 64 * 3 + channels[1] + channels[1]//2, in_channels + 64 * 3 + channels[1] + channels[1]//2, 5, padding=2)
        self.conv_concat_reshape1 = nn.Conv2d(in_channels + 64 * 3 + channels[1] + channels[1]//2, in_channels, 1)
        self.conv_concat_reshape2 = nn.Conv2d(in_channels + 64 * 3 + channels[1] + channels[1]//2, in_channels, 1)
        self.conv_out = nn.Conv2d(in_channels, 1, 1)

    def forward(self, coarse_m, x2_2, x3_1, raw_x):
        coarse_m, mid_pred1 = self.hires_cat(coarse_m, raw_x, 0)
        coarse_m = self.decoder_block1(coarse_m)
        coarse_m = torch.cat((x2_2, coarse_m, x3_1), 1)
        coarse_m = self.conv_concat_reshape1(self.conv_concat1(coarse_m))

        coarse_m, mid_pred2 = self.hires_cat(self.conv_upsample1(self.upsample(coarse_m)), raw_x, 1)
        coarse_m = self.decoder_block2(coarse_m)
        coarse_m = self.multi_cat(x2_2, coarse_m, x3_1)
        coarse_m = self.conv_out(coarse_m)
        coarse_m = self.upsample4(coarse_m)
        return coarse_m, mid_pred1, mid_pred2

    def hires_cat(self, coarse_m, raw_x, step):
        if step == 0:
            coarse_pred_map = self.conv_pred_m1(coarse_m)
            mid_pred = coarse_pred_map
            coarse_pred_map = coarse_pred_map.sigmoid()
        else:
            coarse_pred_map = self.conv_pred_m2(coarse_m)
            mid_pred = coarse_pred_map
            coarse_pred_map = coarse_pred_map.sigmoid()

        batch_size, channels_raw, height_raw, width_raw = raw_x.size()
        coarse_patches_size = coarse_pred_map.size(-1) // 8
        raw_x_patch_size = height_raw // 8
        coarse_patches = coarse_pred_map.unfold(2, coarse_patches_size, coarse_patches_size).unfold(3,
                                                                                                    coarse_patches_size,
                                                                                                    coarse_patches_size)
        coarse_patches = coarse_patches.contiguous().view(batch_size, 1, -1, coarse_patches_size, coarse_patches_size)

        raw_patches = raw_x.unfold(2, raw_x_patch_size, raw_x_patch_size).unfold(3, raw_x_patch_size, raw_x_patch_size)
        raw_patches = raw_patches.contiguous().view(batch_size, channels_raw, -1, raw_x_patch_size, raw_x_patch_size)
        foreground_patches = []
        for b in range(batch_size):
            foreground_patches_batch = []
            for i in range(coarse_patches.size(2)):
                patch = coarse_patches[b, :, i, :, :]
                if patch.max() > 0.5:
                    selected_patch = raw_patches[b, :, i, :, :].unsqueeze(0)
                    if selected_patch.size(-1) != coarse_m.size(-1):
                        selected_patch = F.interpolate(selected_patch, coarse_m.size()[-2:], mode='bilinear',
                                                       align_corners=True)
                    foreground_patches_batch.append(selected_patch)
            if foreground_patches_batch:
                high_res_patches_batch = torch.cat(foreground_patches_batch, dim=1)
            else:
                high_res_patches_batch = torch.zeros(1, 64, coarse_m.size(-1), coarse_m.size(-1),
                                                     device=coarse_m.device)
            foreground_patches.append(high_res_patches_batch)

        return self.batch_concat(coarse_m, foreground_patches), mid_pred

    def batch_concat(self, coarse_m, high_res_patches_lists):
        batched_results = []
        target_channels = 64 * 3

        for b in range(coarse_m.size(0)):
            high_res_patches = high_res_patches_lists[b]
            current_channels = high_res_patches.size(1)

            if current_channels < target_channels:
                repeat_times = target_channels // current_channels
                remainder_channels = target_channels % current_channels

                high_res_patches = high_res_patches.repeat(1, repeat_times, 1, 1)

                if remainder_channels > 0:
                    high_res_patches = torch.cat([high_res_patches, high_res_patches[:, :remainder_channels, :, :]],
                                                 dim=1)

                if high_res_patches.size(1) < target_channels:
                    padding_channels = target_channels - high_res_patches.size(1)
                    padding = torch.zeros(1, padding_channels, high_res_patches.size(-2), high_res_patches.size(-1),
                                          device=high_res_patches.device)
                    high_res_patches = torch.cat([high_res_patches, padding], dim=1)

            coarse_mb = coarse_m[b].unsqueeze(0)
            coarse_mb = torch.cat([coarse_mb, high_res_patches], 1)

            if coarse_mb.size(1) < self.max_in_channels:
                padding_channels = self.max_in_channels - coarse_mb.size(1)
                padding = torch.zeros(1, padding_channels, coarse_m.size(-2), coarse_m.size(-1), device=coarse_m.device)
                coarse_mb = torch.cat([coarse_mb, padding], dim=1)

            batched_results.append(coarse_mb)

        return torch.cat(batched_results, 0)

    def multi_cat(self, x2_2, coarse_m, x3_1):
        x2_2 = F.interpolate(x2_2, coarse_m.size()[-2:], mode='bilinear', align_corners=True)
        x3_1 = F.interpolate(x3_1, coarse_m.size()[-2:], mode='bilinear', align_corners=True)
        coarse_m = torch.cat((x2_2, coarse_m, x3_1), 1)
        coarse_m = self.conv_concat_reshape2(self.conv_concat2(coarse_m))
        return coarse_m

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x




