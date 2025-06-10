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


# v2, 做两次cat (是否做权重共享？不应该做权重共享，因为我这里之所以分两个阶段，就是要区分它们。做共享了不是白区分)
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
        # self.conv_upsample2 = BasicConv2d(in_channels, in_channels, 3, padding=1, dilation=2)
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
            '''
            visualization
            
            visual_coarse_m = coarse_pred_map
            res = nn.functional.interpolate(
                visual_coarse_m,
                size=(589, 617),
                mode='bilinear',
                align_corners=True
            )
            res = (res * 255)
            save_tensor_img(res, '/home/amos/PycharmProjects/COD_TEST/visualzation/coarse_m1.jpg')
            
            visualization end       
            '''
        else:
            coarse_pred_map = self.conv_pred_m2(coarse_m)
            mid_pred = coarse_pred_map
            coarse_pred_map = coarse_pred_map.sigmoid()
            '''
            visualization
            
            visual_coarse_m = coarse_pred_map
            res = nn.functional.interpolate(
                visual_coarse_m,
                size=(589, 617),
                mode='bilinear',
                align_corners=True
            )
            res = (res * 255)
            save_tensor_img(res, '/home/amos/PycharmProjects/COD_TEST/visualzation/coarse_m2.jpg')
            
            visualization end       
            '''
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
            '''
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





'''
if __name__ == "__main__":
    # coarse_m, x2_2, x3_1, raw_x
    coarse_m = torch.randn(4, 576, 64, 64)
    x2_2 = torch.randn(4, 576, 64, 64)
    x3_1 = torch.randn(4, 288, 64, 64)
    raw_x = torch.randn(4, 3, 512, 512)


    net = ResDetHeaBlk_v2()
    logits = net(coarse_m, x2_2, x3_1, raw_x)

'''

'''
    整合你之前提供的信息，ResDetHeaBlk可以说是对高分辨率原始图像取相应块之后做下采样，拼接到特征图上。但是如果我不想折损分辨率信息，不想在这里对原始高分辨率patches做下采样。我想修改get_patches函数，让其返回的patches分辨率等于输入的coarse_m的分辨率，实现方式如下：
    
    作用：获取高分辨率patch，以进行concat
    实现方式：
    先备份coarse_map，然后对coarse_map计算 sigmoid，并乘以255，形成一个coarse_pred_map.
    然后对粗糙预测图 coarse_pred_map[64,64] 与 原始输入图片raw_x[512,512] 分别做等比例分割
        例如：第一次concat时，将 coarse_pred_map 分割成 8*8个patch ，每个patch的分辨率都是8*8；将raw_x 分割成8*8个patch，每个patch的分辨率都是64*64
            第二次concat时,由于 coarse_pred_map 经过 upsample 和 conv_upsample1 已经变成了 [128,128],因此做8*8patch 之后每个patch的分辨率是16*16，将raw_x 分割成8*8个patch，每个patch的分辨率都是64*64。
            第三次concat时,由于 coarse_pred_map 经过 upsample 和 conv_upsample2 已经变成了 [256,256],因此做8*8patch 之后每个patch的分辨率是32*32，将raw_x 分割成8*8个patch，每个patch的分辨率都是64*64。
    然后对于每一次 concat 寻找coarse_pred_map 的 patches中所有的包含前景信息patches，并根据等比例分割选取raw_x中相应位置的patches
    此时还可能面临raw_x的patches尺寸==coarse_pred_map的尺寸的情况，如果这种情况发生，我们不需要进行额外处理。因为它可以直接在forward中进行concat
    此时可能面临raw_x的patches尺寸<coarse_pred_map的尺寸的情况，如果这种情况发生，一般是因为输入的coarse_map已经经过若干次上采样处理，变得很大了。我们需要将raw_x patches进行上采样，使其分辨率等于coarse_pred_map的尺寸。
    在我们的设计上raw_x的patches尺寸不会发生>coarse_pred_map尺寸的情况，如果发生请抛出报错。

        例如：假如第一次concat时，coarse_pred_map选中了三个包含前景信息的patches，相应的我们找到了raw_x上三个相对位置的patches。此时raw_x的三个patches的分辨率是64*64，等于coarse_pred_map的尺寸，我们就不需要进行进一步的patch。
            然而第一次concat时，假如coarse_pred_map又选中了三个包含前景信息的patches，相应的我们找到了raw_x上三个相对位置的patches。此时raw_x的三个patches的分辨率是64*64，然而coarse_pred_map经过上采样已经变成128*128，大于raw_x的三个patches的分辨率
            此时我们需要raw_x的三个patches做上采样，使其分辨率等于coarse_pred_map的分辨率。

    最后将这些选中的raw_x中的patches做concat，作为get_patches的返回值。

'''

'''

对于这俩特征图，patch块数确实都是64块。大图和小图做的都是八行八列分块。
块数量一致是因为要根据低分辨率预测图的前景信息来定位，其前景块号。再利用低分辨率图的前景块号，以及二者是等比分割的关系，去512大图片 raw_x 上来选取相同块号的patch。
这些patch就是模型粗糙预测的存在伪装物体的块号，并且具有高分辨率信息。
然后是拼接阶段，这一步骤不需要二者patch的尺寸一致，而需要大图的patch尺寸等于小图的整个图尺寸。
将大图的带有可疑前景信息的高分辨率patch直接沿着通道维度 cat 到小图整张图上。
在这个例子中，大图分完八行八列，每个patch的分辨率固定是64*64。
小图的增长图分辨率，则由于分三次二倍上采样所以在每次cat之前分辨率是 64，128，256。
所以第一次cat时二者都是64不需要动
后两次则是要将大图的patch二倍上采样，才能cat在小图的通道维度上。
'''
