import torch
import torch.nn as nn
from config import Config
import torch.nn.functional as F
from models.modules.decoder_blocks import RF, BasicDecBlk, BasicDecBlk_lite, LocalDecBlk
from utils.utils import path_to_image, save_tensor_img, save_tensor_heatmap, save_tensor_img2

config = Config()


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
        # print(self.index)
        # print(config.dec_traget_size[self.index])
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)

        x = self.conv_out(x)
        x = self.bn_out(x)

        # 逐渐放大尺寸，对齐输入尺寸
        # x = F.interpolate(x, size=244, mode='bilinear', align_corners=False)
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


#串行策略 整体解码
class ResDetHeaBlk_v2(nn.Module):
    def __init__(self, in_channels):
        super(ResDetHeaBlk_v2, self).__init__()
        self.max_in_channels = in_channels + 64 * 3
        # BasicDecBlk_lite
        self.decoder1 = BasicDecBlk_lite(in_channels + 64 * 3, in_channels + 64 * 3)
        self.decoder2 = BasicDecBlk_lite(in_channels + 64 * 3, in_channels + 64 * 3)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.conv_pred_m1 = nn.Conv2d(in_channels, 1, 1)
        self.conv_pred_m2 = nn.Conv2d(in_channels, 1, 1)
        self.conv_concat1 = BasicConv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2,
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, 3, padding=1)
        self.conv_concat2 = BasicConv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2,
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, 5, padding=2)
        self.conv_concat_reshape1 = nn.Conv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, in_channels, 1)
        self.conv_concat_reshape2 = nn.Conv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, in_channels, 1)
        self.conv_out = nn.Conv2d(in_channels, 1, 1)

        # 用于缓存 foreground_patches
        self.cached_foreground_patches = None

    def forward(self, coarse_m, x2_2, x3_1, raw_x):
        coarse_m, mid_pred1 = self.hires_cat(coarse_m, raw_x, 0)
        coarse_m = self.decoder1(coarse_m)
        coarse_m = torch.cat((x2_2, coarse_m, x3_1), 1)
        coarse_m = self.conv_concat_reshape1(self.conv_concat1(coarse_m))

        coarse_m, mid_pred2 = self.hires_cat(self.conv_upsample1(self.upsample(coarse_m)), raw_x, 1)
        coarse_m = self.decoder2(coarse_m)
        coarse_m = self.multi_cat(x2_2, coarse_m, x3_1)
        coarse_m = self.conv_out(coarse_m)
        coarse_m = self.upsample4(coarse_m)
        return coarse_m, mid_pred1, mid_pred2

    def hires_cat(self, coarse_m, raw_x, step):
        if step == 0:
            coarse_pred_map = self.conv_pred_m1(coarse_m)
            mid_pred = coarse_pred_map
            coarse_pred_map = coarse_pred_map.sigmoid()
            self.cached_foreground_patches = self.compute_foreground_patches(coarse_pred_map, raw_x, coarse_m.size())
        else:
            mid_pred = self.conv_pred_m2(coarse_m)
            # 直接使用缓存的 foreground_patches
        return self.batch_concat(coarse_m, self.cached_foreground_patches), mid_pred

    def compute_foreground_patches(self, coarse_pred_map, raw_x, coarse_m_size):
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
                    foreground_patches_batch.append(selected_patch)
            if foreground_patches_batch:
                high_res_patches_batch = torch.cat(foreground_patches_batch, dim=1)
            else:
                high_res_patches_batch = torch.zeros(1, 64, coarse_m_size[-1], coarse_m_size[-1],
                                                     device=coarse_pred_map.device)
            foreground_patches.append(high_res_patches_batch)
        return foreground_patches

    def batch_concat(self, coarse_m, high_res_patches_lists):
        batched_results = []
        target_channels = 64 * 3
        for b in range(coarse_m.size(0)):
            high_res_patches = high_res_patches_lists[b]

            # 如果 patch 尺寸和 coarse_m 尺寸不匹配，进行调整
            if high_res_patches.size(-1) != coarse_m.size(-1):
                high_res_patches = F.interpolate(high_res_patches, size=coarse_m.size()[-2:], mode='bilinear',
                                                 align_corners=True)

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


# 滑动窗口卷积，限制感受野，增强局部特征理解能力。并行策略
class ResDetHeaBlk_v3(nn.Module):
    def __init__(self, in_channels):
        super(ResDetHeaBlk_v3, self).__init__()
        self.max_in_channels = in_channels + 64 * 3
        # 替换 BasicDecBlk_lite 的实现
        self.decoder1 = ResDetDecoder(in_channels + 64 * 3)
        self.decoder2 = ResDetDecoder(in_channels + 64 * 3)
        # self.decoder_lite = BasicDecBlk_lite(in_channels, in_channels)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(in_channels, in_channels, 3, padding=1)

        self.conv_pred_m1 = nn.Conv2d(in_channels, 1, 1)
        self.conv_pred_m2 = nn.Conv2d(in_channels, 1, 1)
        self.conv_concat1 = BasicConv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2,
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, 3, padding=1)
        self.conv_concat2 = BasicConv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2,
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, 5, padding=2)
        self.conv_concat_reshape1 = nn.Conv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, in_channels, 1)
        self.conv_concat_reshape2 = nn.Conv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, in_channels, 1)
        self.conv_out = nn.Conv2d(in_channels, 1, 1)

        # 用于缓存 foreground_patches
        self.cached_foreground_patches = None

    def forward(self, coarse_m, x2_2, x3_1, raw_x):
        coarse_m_low_res = coarse_m
        coarse_m_high_res = self.upsample(coarse_m_low_res)
        # low-res banch
        coarse_m_low_res, mid_pred1 = self.hires_cat(coarse_m_low_res, raw_x, 0)
        coarse_m_low_res = self.decoder1(coarse_m_low_res)
        coarse_m_low_res = torch.cat((x2_2, coarse_m_low_res, x3_1), 1)
        coarse_m_low_res = self.conv_concat_reshape1(self.conv_concat1(coarse_m_low_res))

        # hi-res banch
        coarse_m_high_res, mid_pred2 = self.hires_cat(self.conv_upsample1(coarse_m_high_res), raw_x, 1)
        coarse_m_high_res = self.decoder2(coarse_m_high_res)
        coarse_m_high_res = self.multi_cat(x2_2, coarse_m_high_res, x3_1)

        # fine_m = coarse_m fusion + decode
        coarse_m_low_res = self.conv_upsample2(self.upsample(coarse_m_low_res))
        fine_m = coarse_m_low_res + coarse_m_high_res
        # fine_m = torch.nn.BatchNorm2d(fine_m.size(1))(fine_m)
        #fine_m = self.decoder_lite(fine_m)

        # output fine_m
        fine_m = self.conv_out(fine_m)
        fine_m = self.upsample4(fine_m)
        return fine_m, mid_pred1, mid_pred2

    def hires_cat(self, coarse_m, raw_x, step):
        if step == 0:
            coarse_pred_map = self.conv_pred_m1(coarse_m)
            mid_pred = coarse_pred_map
            coarse_pred_map = coarse_pred_map.sigmoid()
            self.cached_foreground_patches = self.compute_foreground_patches(coarse_pred_map, raw_x, coarse_m.size())
        else:
            mid_pred = self.conv_pred_m2(coarse_m)
            # 直接使用缓存的 foreground_patches
        return self.batch_concat(coarse_m, self.cached_foreground_patches), mid_pred

    def compute_foreground_patches(self, coarse_pred_map, raw_x, coarse_m_size):
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
                    foreground_patches_batch.append(selected_patch)
            if foreground_patches_batch:
                high_res_patches_batch = torch.cat(foreground_patches_batch, dim=1)
            else:
                high_res_patches_batch = torch.zeros(1, 64, coarse_m_size[-1], coarse_m_size[-1],
                                                     device=coarse_pred_map.device)
            foreground_patches.append(high_res_patches_batch)
        return foreground_patches

    def batch_concat(self, coarse_m, high_res_patches_lists):
        batched_results = []
        target_channels = 64 * 3
        for b in range(coarse_m.size(0)):
            high_res_patches = high_res_patches_lists[b]

            # 如果 patch 尺寸和 coarse_m 尺寸不匹配，进行调整
            if high_res_patches.size(-1) != coarse_m.size(-1):
                high_res_patches = F.interpolate(high_res_patches, size=coarse_m.size()[-2:], mode='bilinear',
                                                 align_corners=True)

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


class ResDetDecoder(nn.Module):
    def __init__(self, in_channels):
        super(ResDetDecoder, self).__init__()
        self.basic_decoder = BasicDecBlk(in_channels, in_channels)
        self.local_decoder = LocalDecBlk(in_channels, in_channels)

    def forward(self, x):
        x = self.basic_decoder(x + self.local_decoder(x))
        # x = self.local_decoder(x + self.basic_decoder(x))
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


'''
if __name__ == "__main__":
    # coarse_m, x2_2, x3_1, raw_x
    coarse_m = torch.randn(4, config.channels_list[1], 64, 64)
    x2_2 = torch.randn(4, config.channels_list[1], 64, 64)
    x3_1 = torch.randn(4, config.channels_list[1]//2, 64, 64)
    raw_x = torch.randn(4, 3, 512, 512)


    net = ResDetHeaBlk_v2()
    logits = net(coarse_m, x2_2, x3_1, raw_x)



    def hires_cat(self, coarse_m, raw_x, step):
        if step == 0:
            coarse_pred_map = self.conv_pred_m1(coarse_m)
            mid_pred = coarse_pred_map
            coarse_pred_map = coarse_pred_map.sigmoid()

            visualization

            visual_coarse_m = coarse_pred_map
            res = nn.functional.interpolate(
                visual_coarse_m,
                size=config.visual_size,
                mode='bilinear',
                align_corners=True
            )
            save_tensor_heatmap(res, '/home/amos/PycharmProjects/COD_TEST/visualzation/coarse_m1.jpg')

            visualization end       

        else:
            coarse_pred_map = self.conv_pred_m2(coarse_m)
            mid_pred = coarse_pred_map
            coarse_pred_map = coarse_pred_map.sigmoid()

            visualization

            visual_coarse_m = coarse_pred_map
            res = nn.functional.interpolate(
                visual_coarse_m,
                size=config.visual_size,
                mode='bilinear',
                align_corners=True
            )
            save_tensor_heatmap(res, '/home/amos/PycharmProjects/COD_TEST/visualzation/coarse_m2.jpg')

            visualization end       

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

  
                                    #Visualization of foreground patches
                            # Save the first foreground patch of the batch for visualization
                            if foreground_patches_batch:
                                for j, patch in enumerate(foreground_patches_batch):
                                    visual_patch = patch[0, :, :, :].cpu().detach()
                                    visual_patch = (visual_patch - visual_patch.min()) / (
                                                visual_patch.max() - visual_patch.min())  # Normalize
                                    save_tensor_img2(visual_patch,
                                                    f'/home/amos/PycharmProjects/COD_TEST/visualzation/output/foreground_patch_b{b}_j{j}.jpg')


                                    #Reconstructing image from patches

                            reconstructed_image = torch.zeros((channels_raw, height_raw, width_raw), device=coarse_m.device)
                            patch_count = torch.zeros((1, height_raw, width_raw), device=coarse_m.device)

                            patch_size = raw_x_patch_size
                            num_patches = int(height_raw // patch_size)

                            for i in range(num_patches):
                                for j in range(num_patches):
                                    if coarse_patches[b, :, i * num_patches + j, :, :].max() > 0.5:
                                        patch = raw_patches[b, :, i * num_patches + j, :, :].squeeze()
                                        reconstructed_image[:, i * patch_size:(i + 1) * patch_size,
                                        j * patch_size:(j + 1) * patch_size] += patch
                                        patch_count[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] += 1

                            patch_count[patch_count == 0] = 1
                            reconstructed_image /= patch_count

                            # Save the reconstructed image
                            save_tensor_img2(reconstructed_image,
                                            f'/home/amos/PycharmProjects/COD_TEST/visualzation/reconstructed_image_b{b}.jpg')
                '''


'''
串行策略 局部解码
class ResDetHeaBlk_v3(nn.Module):
    def __init__(self, in_channels):
        super(ResDetHeaBlk_v3, self).__init__()
        self.max_in_channels = in_channels + 64 * 3
        # 替换 BasicDecBlk_lite 的实现
        self.decoder1 = ResDetDecoder(in_channels + 64 * 3)
        self.decoder2 = ResDetDecoder(in_channels + 64 * 3)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.conv_pred_m1 = nn.Conv2d(in_channels, 1, 1)
        self.conv_pred_m2 = nn.Conv2d(in_channels, 1, 1)
        self.conv_concat1 = BasicConv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2,
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, 3, padding=1)
        self.conv_concat2 = BasicConv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2,
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, 5, padding=2)
        self.conv_concat_reshape1 = nn.Conv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, in_channels, 1)
        self.conv_concat_reshape2 = nn.Conv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, in_channels, 1)
        self.conv_out = nn.Conv2d(in_channels, 1, 1)

        # 用于缓存 foreground_patches
        self.cached_foreground_patches = None

    def forward(self, coarse_m, x2_2, x3_1, raw_x):
        coarse_m, mid_pred1 = self.hires_cat(coarse_m, raw_x, 0)
        coarse_m = self.decoder1(coarse_m)
        coarse_m = torch.cat((x2_2, coarse_m, x3_1), 1)
        coarse_m = self.conv_concat_reshape1(self.conv_concat1(coarse_m))

        coarse_m, mid_pred2 = self.hires_cat(self.conv_upsample1(self.upsample(coarse_m)), raw_x, 1)
        coarse_m = self.decoder2(coarse_m)
        coarse_m = self.multi_cat(x2_2, coarse_m, x3_1)
        coarse_m = self.conv_out(coarse_m)
        coarse_m = self.upsample4(coarse_m)
        return coarse_m, mid_pred1, mid_pred2

    def hires_cat(self, coarse_m, raw_x, step):
        if step == 0:
            coarse_pred_map = self.conv_pred_m1(coarse_m)
            mid_pred = coarse_pred_map
            coarse_pred_map = coarse_pred_map.sigmoid()
            self.cached_foreground_patches = self.compute_foreground_patches(coarse_pred_map, raw_x, coarse_m.size())
        else:
            mid_pred = self.conv_pred_m2(coarse_m)
            # 直接使用缓存的 foreground_patches
        return self.batch_concat(coarse_m, self.cached_foreground_patches), mid_pred

    def compute_foreground_patches(self, coarse_pred_map, raw_x, coarse_m_size):
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
                    foreground_patches_batch.append(selected_patch)
            if foreground_patches_batch:
                high_res_patches_batch = torch.cat(foreground_patches_batch, dim=1)
            else:
                high_res_patches_batch = torch.zeros(1, 64, coarse_m_size[-1], coarse_m_size[-1],
                                                     device=coarse_pred_map.device)
            foreground_patches.append(high_res_patches_batch)
        return foreground_patches

    def batch_concat(self, coarse_m, high_res_patches_lists):
        batched_results = []
        target_channels = 64 * 3
        for b in range(coarse_m.size(0)):
            high_res_patches = high_res_patches_lists[b]

            # 如果 patch 尺寸和 coarse_m 尺寸不匹配，进行调整
            if high_res_patches.size(-1) != coarse_m.size(-1):
                high_res_patches = F.interpolate(high_res_patches, size=coarse_m.size()[-2:], mode='bilinear',
                                                 align_corners=True)

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
'''


'''
# 滑动窗口卷积，限制感受野，增强局部特征理解能力。并行策略
class ResDetHeaBlk_v3(nn.Module):
    def __init__(self, in_channels):
        super(ResDetHeaBlk_v3, self).__init__()
        self.max_in_channels = in_channels + 64 * 3
        # 替换 BasicDecBlk_lite 的实现
        self.decoder1 = ResDetDecoder(in_channels + 64 * 3)
        self.decoder2 = ResDetDecoder(in_channels + 64 * 3)
        # self.decoder_lite = BasicDecBlk_lite(in_channels, in_channels)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(in_channels, in_channels, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(in_channels, in_channels, 3, padding=1)

        self.conv_pred_m1 = nn.Conv2d(in_channels, 1, 1)
        self.conv_pred_m2 = nn.Conv2d(in_channels, 1, 1)
        self.conv_concat1 = BasicConv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2,
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, 3, padding=1)
        self.conv_concat2 = BasicConv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2,
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, 5, padding=2)
        self.conv_concat_reshape1 = nn.Conv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, in_channels, 1)
        self.conv_concat_reshape2 = nn.Conv2d(
            config.channels_list[1] + 64 * 3 + config.channels_list[1] + config.channels_list[1] // 2, in_channels, 1)
        self.conv_out = nn.Conv2d(in_channels, 1, 1)

        # 用于缓存 foreground_patches
        self.cached_foreground_patches = None

    def forward(self, coarse_m, x2_2, x3_1, raw_x):
        coarse_m_low_res = coarse_m
        coarse_m_high_res = self.upsample(coarse_m_low_res)
        # low-res banch
        coarse_m_low_res, mid_pred1 = self.hires_cat(coarse_m_low_res, raw_x, 0)
        coarse_m_low_res = self.decoder1(coarse_m_low_res)
        coarse_m_low_res = torch.cat((x2_2, coarse_m_low_res, x3_1), 1)
        coarse_m_low_res = self.conv_concat_reshape1(self.conv_concat1(coarse_m_low_res))

        # hi-res banch
        coarse_m_high_res, mid_pred2 = self.hires_cat(self.conv_upsample1(coarse_m_high_res), raw_x, 1)
        coarse_m_high_res = self.decoder2(coarse_m_high_res)
        coarse_m_high_res = self.multi_cat(x2_2, coarse_m_high_res, x3_1)

        # fine_m = coarse_m fusion + decode
        coarse_m_low_res = self.conv_upsample2(self.upsample(coarse_m_low_res))
        fine_m = coarse_m_low_res + coarse_m_high_res
        # fine_m = torch.nn.BatchNorm2d(fine_m.size(1))(fine_m)
        #fine_m = self.decoder_lite(fine_m)

        # output fine_m
        fine_m = self.conv_out(fine_m)
        fine_m = self.upsample4(fine_m)
        return fine_m, mid_pred1, mid_pred2
'''