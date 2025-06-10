import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from models.backbones.build_backbone import build_backbone
from models.modules.decoder_blocks import BasicDecBlk, RF
from models.modules.head_blocks import OutHeaBlk, AttBeaBlk, ResDetHeaBlk_v2, ResDetHeaBlk_v3
from models.modules.lateral_blocks import GussianLatBlk, KPConvLatBlk
from models.modules.attention_blocks import SpatialAttention, MixResAttention
from models.modules.mix_blocks import ScMixBlk
from utils.utils import save_tensor_img_for_latblk


class SANet(nn.Module):
    def __init__(self, backbone_pretrained=True):
        super(SANet, self).__init__()
        self.config = Config()
        self.epoch = 1
        self.backbone = build_backbone(self.config.backbone, self.config.is_pretrained_backbone)
        self.decoder = Decoder(self.config.channels_list)

    def forward_preprocess(self, x):
        if self.config.backbone in ['vgg16', 'vgg16bn', 'resnet50']:
            x1 = self.backbone.conv1(x)
            x2 = self.backbone.conv2(x1)
            x3 = self.backbone.conv3(x2)
            x4 = self.backbone.conv4(x3)
        else:
            if self.config.backbone in ['MambaVision_b_1k', 'MambaVision_l_1k']:
                _, outs = self.backbone(x)
                x1, x2, x3, x4 = outs
            else:
                x1, x2, x3, x4 = self.backbone(x)
            if self.config.mul_scl_ipt == 'cat':
                B, C, H, W = x.shape
                x1_, x2_, x3_, x4_ = self.backbone(
                    F.interpolate(x, size=(H // 2, W // 2), mode='bilinear', align_corners=True))
                x1 = torch.cat([x1, F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)], dim=1)
                x2 = torch.cat([x2, F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)], dim=1)
                x3 = torch.cat([x3, F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)], dim=1)
                x4 = torch.cat([x4, F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)], dim=1)
            elif self.config.mul_scl_ipt == 'add':
                B, C, H, W = x.shape
                x1_, x2_, x3_, x4_ = self.backbone(
                    F.interpolate(x, size=(H // 2, W // 2), mode='bilinear', align_corners=True))
                x1 = x1 + F.interpolate(x1_, size=x1.shape[2:], mode='bilinear', align_corners=True)
                x2 = x2 + F.interpolate(x2_, size=x2.shape[2:], mode='bilinear', align_corners=True)
                x3 = x3 + F.interpolate(x3_, size=x3.shape[2:], mode='bilinear', align_corners=True)
                x4 = x4 + F.interpolate(x4_, size=x4.shape[2:], mode='bilinear', align_corners=True)
        if self.config.mul_scl_sc:
            x4 = torch.cat(
                (
                    *[
                         F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=True),
                         F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=True),
                         F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True),
                     ][-self.config.mul_scl_sc_num:],
                    x4
                ),
                dim=1
            )
        if self.config.mul_lev_ipt:
            x4_ = x4
            x4 = torch.cat(
                (
                    *[
                         F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=True),
                         F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=True),
                         F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True),
                     ][-self.config.mul_scl_sc_num:],
                    x4
                ),
                dim=1
            )
            x1 = torch.cat(
                (
                    F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True),
                    x2
                ),
                dim=1
            )
            x2 = torch.cat(
                (
                    F.interpolate(x2, size=x3.shape[2:], mode='bilinear', align_corners=True),
                    x3
                ),
                dim=1
            )
            x3 = torch.cat(
                (
                    F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True),
                    x4_
                ),
                dim=1
            )
            
        return x1, x2, x3, x4

    def forward(self, x):
        ########## Encoder ##########
        x1, x2, x3, x4 = self.forward_preprocess(x)
        ########## Decoder ##########
        features = [x, x1, x2, x3, x4]
        pred_m = self.decoder(features)
        return pred_m


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.config = Config()
        LateralBlock = eval(self.config.lat_blk)
        DecoderBlock = eval(self.config.dec_blk)
        AttentionBlock = eval(self.config.att_blk)
        HeadBlock = eval(self.config.hea_blk)

        self.lateral_block2 = LateralBlock(channels[2], 2)
        self.lateral_block1 = LateralBlock(channels[1], 1)

        self.decoder_block4 = DecoderBlock(channels[4], channels[4])
        self.decoder_block3 = DecoderBlock(channels[3], channels[3])
        self.decoder_block2 = DecoderBlock(channels[2], channels[2])
        self.decoder_block1 = DecoderBlock(channels[1], channels[1])
        if self.config.lat_blk == 'KPConvLatBlk':
            self.edge_shape1_2 = nn.Conv2d(channels[1], channels[2],1)
            self.edge_shape2_1 = nn.Conv2d(channels[2], channels[1],1)

        self.mix_block1 = ScMixBlk(channels)

        if self.config.att_blk == 'MixResAttention':
            self.attention_block2 = AttentionBlock(channels[1], channels[2])
            self.attention_block3 = SpatialAttention(channels[4])
        else:
            self.attention_block1 = AttentionBlock(channels[1])
            self.attention_block4 = AttentionBlock(channels[4])

        if self.config.hea_blk == 'ResDetHeaBlk_v2' or self.config.hea_blk == 'ResDetHeaBlk_v3':
            self.head_block = HeadBlock(channels[1])
        else:
            print('EORROR: not find head block!')

    def forward(self, features):
        # fds = []
        x, x1, x2, x3, x4 = features


        # --- Get edge weights ---
        e1 = self.lateral_block1(x1)
        e2 = self.lateral_block2(x2)

        # Structure weight denoising
        if self.config.lat_blk == 'GussianLatBlk' and self.config.lat_blk_filter:
            threshold = 1.0
            e1 = torch.where(e1 >= threshold, e1,
                             torch.tensor(1.0, dtype=torch.float32))
            e2 = torch.where(e2 >= threshold, e2,
                             torch.tensor(1.0, dtype=torch.float32))

        # ---- Get pred_m ----
        if self.config.lat_blk == 'GussianLatBlk':
            x1 = x1 * e1
            x2 = x2 * e2
        elif self.config.lat_blk == 'KPConvLatBlk':
            e1_2 = F.interpolate(e1, size=x2.shape[-2:], mode='bilinear', align_corners=False)
            e2_1 = F.interpolate(e2, size=x1.shape[-2:], mode='bilinear', align_corners=False)
            e1_2 = self.edge_shape1_2(e1_2)
            e2_1 = self.edge_shape2_1(e2_1)

            x1 = x1 + e1 + e2_1
            x2 = x2 + e2 + e1_2

        if (self.config.att_blk ==
                'MixResAttention'):
            x2 = self.attention_block2(x1, x2)
            x4 = self.attention_block3(x4)
        else:
            x1 = self.attention_block1(x1)
            x4 = self.attention_block4(x4)

        # Apply attention before decoding
        x1 = self.decoder_block1(x1)
        x2 = self.decoder_block2(x2)
        x3 = self.decoder_block3(x3)
        x4 = self.decoder_block4(x4)

        pred_m, x2_2, x3_1 = self.mix_block1(x1, x2, x3, x4)
        if self.config.hea_blk == 'ResDetHeaBlk_v2' or self.config.hea_blk == 'ResDetHeaBlk_v3':
            pred_m, mid_pred1, mid_pred2 = self.head_block(pred_m, x2_2, x3_1, x)
            return pred_m, mid_pred1, mid_pred2

        else:
            pred_m = self.head_block(x)

        return pred_m


def main():
    # Debug code: run a forward pass once
    model = SANet()
    input_tensor = torch.randn(1, 3, 512, 512)  # batch_size=1, channels=3, height=512, width=512
    # Run forward pass
    output = model(input_tensor)

    print("Input tensor shape:", input_tensor.shape)

    x1, x2, x3, x4 = model.forward_preprocess(input_tensor)
    print("x1 shape:", x1.shape)
    print("x2 shape:", x2.shape)
    print("x3 shape:", x3.shape)
    print("x4 shape:", x4.shape)

    if isinstance(output, tuple):
        for i, out in enumerate(output):
            print(f"Output {i} shape:", out.shape)
    else:
        print("Output shape:", output.shape)

#if __name__ == "__main__":
#    main()