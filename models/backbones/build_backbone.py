import os
import torch
import torch.nn as nn
from config import Config
from torchvision.models import vgg16, vgg16_bn, VGG16_Weights, VGG16_BN_Weights, resnet50, ResNet50_Weights
from models.backbones.pvt_v2 import pvt_v2_b0, pvt_v2_b2, pvt_v2_b5, pvt_v2_b1
from models.backbones.swin_v1 import swin_v1_b, swin_v1_l, swin_v1_t, swin_v1_s
from collections import OrderedDict
from transformers import AutoModel

config = Config()


def build_backbone(bb_name, pretrained=True, params_settings=''):

    if pretrained is True:
        print('Loading the pretrained backbone setting from config - backbone:{}...'.format(bb_name))
    if bb_name == 'vgg16':
        bb_net = list(vgg16(pretrained=VGG16_Weights.DEFAULT if pretrained else None).children())[0]
        bb = nn.Sequential(OrderedDict({'conv1': bb_net[:4], 'conv2': bb_net[4:9], 'conv3': bb_net[9:16], 'conv4': bb_net[16:23]}))
    elif bb_name == 'vgg16bn':
        bb_net = list(vgg16_bn(pretrained=VGG16_BN_Weights.DEFAULT if pretrained else None).children())[0]
        bb = nn.Sequential(OrderedDict({'conv1': bb_net[:6], 'conv2': bb_net[6:13], 'conv3': bb_net[13:23], 'conv4': bb_net[23:33]}))
    elif bb_name == 'resnet50':
        bb_net = list(resnet50(pretrained=ResNet50_Weights.DEFAULT if pretrained else None).children())
        bb = nn.Sequential(OrderedDict({'conv1': nn.Sequential(*bb_net[0:3]), 'conv2': bb_net[4], 'conv3': bb_net[5], 'conv4': bb_net[6]}))
    elif bb_name == 'MambaVision_b_1k':
        bb = AutoModel.from_pretrained("nvidia/MambaVision-B-1K", trust_remote_code=True)
    elif bb_name == 'MambaVision_l_1k':
        bb = AutoModel.from_pretrained("nvidia/MambaVision-L-1K", trust_remote_code=True)

    else:
        bb = eval('{}({})'.format(bb_name, params_settings))
        if pretrained:
            bb = load_weights(bb)
    return bb


def load_weights(model):
    save_model = torch.load(config.backbone_weights_dir)

    model_dict = model.state_dict()

    state_dict = {k: v if v.size() == model_dict[k].size() else model_dict[k] for k, v in save_model.items() if
                  k in model_dict.keys()}

    # If no matching state_dict found, handle multiple keys
    if not state_dict:
        save_model_keys = list(save_model.keys())
        if len(save_model_keys) == 1:
            sub_item = save_model_keys[0]
        else:
            print(f"Multiple or no keys found in save_model: {save_model_keys}")
            sub_item = None

        if sub_item and sub_item in save_model:
            sub_item_data = save_model[sub_item]
            state_dict = {k: v if v.size() == model_dict[k].size() else model_dict[k] for k, v in sub_item_data.items()
                          if k in model_dict.keys()}
            if not state_dict:
                print(f"No matching state_dict found in the '{sub_item}' item.")
                return None
            print(f"Found correct weights in the '{sub_item}' item of loaded state_dict.")
        else:
            print('Weights are not successfully loaded. Check the state dict of weights file.')
            return None

    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model



#test region
# build_backbone(config.backbone)
