import logging
import os
import torch
from torchvision import transforms
import numpy as np
import random
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import imageio

def path_to_image(path, size=(244, 244), color_type=['rgb', 'gray'][0]):
    if color_type.lower() == 'rgb':
        image = cv2.imread(path)
    elif color_type.lower() == 'gray':
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        print('Select the color_type to return, either to RGB or gray image.')
        return
    if size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    if color_type.lower() == 'rgb':
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
    else:
        image = Image.fromarray(image).convert('L')
    return image


def check_state_dict(state_dict, unwanted_prefix='_orig_mod.'):
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict


def generate_smoothed_gt(gts):
    epsilon = 0.001
    new_gts = (1 - epsilon) * gts + epsilon / 2
    return new_gts


class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger('SANet')
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def info(self, txt):
        self.logger.info(txt)

    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, path, filename="latest.pth"):
    torch.save(state, os.path.join(path, filename))


def save_tensor_img(tenor_im, path):
    '''
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    # im = im.convert('L')
    im.save(path)
    '''
    # 将张量转换为numpy数组，并转换为uint8类型
    im = tenor_im.cpu().clone().squeeze().numpy().astype('uint8')    # 添加通道维度
    # print('im.shape = {}'.format(im.shape))
    # 保存图像
    imageio.imwrite(path, im)


def save_tensor_heatmap(tensor, path):
    # Convert tensor to numpy array
    array = tensor.cpu().detach().numpy().squeeze(0).squeeze(0)
    # Create heatmap
    plt.imshow(array, cmap='hot', interpolation='nearest')
    plt.axis('off')  # Turn off axis
    plt.colorbar()   # Add color bar
    plt.savefig(path, bbox_inches='tight', pad_inches=0.0)  # Save image
    plt.close()     # Close plot to release memory


def save_tensor_img2(tensor_im, path):
    """
    Save a torch tensor as an image file.

    Args:
        tensor_im (torch.Tensor): The tensor to save as an image.
        path (str): The file path to save the image to.
    """
    # Check the shape of the tensor
    if tensor_im.dim() == 4:  # Batch of images
        tensor_im = tensor_im.squeeze(0)

    if tensor_im.dim() == 3:  # Single image
        # Convert to numpy array
        np_img = tensor_im.cpu().numpy()

        # Normalize and scale to [0, 255]
        np_img = (np_img - np_img.min()) / (np_img.max() - np_img.min()) * 255.0
        np_img = np_img.astype(np.uint8)

        if np_img.shape[0] == 1:  # Grayscale
            np_img = np_img.squeeze(0)
        elif np_img.shape[0] == 3:  # RGB
            np_img = np_img.transpose(1, 2, 0)

        imageio.imwrite(path, np_img)
    else:
        raise ValueError("Unexpected tensor shape: {}".format(tensor_im.shape))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_tensor_img_for_latblk(tensor_im, save_dir):
    """
    将输入的 PyTorch tensor 批量可视化并保存为图像文件。

    参数:
        tensor_im (torch.Tensor): 输入的 PyTorch tensor，包含一批图像。
        save_dir (str): 要保存图像的目录路径。

    返回:
        无，将每张图像保存在指定目录下。
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 对每张图像进行遍历并保存
    for i in range(tensor_im.size(0)):
        # 获取单张图像的 tensor
        im_tensor = tensor_im[i].cpu().clone()

        # 将 tensor 转换为 PIL 图像
        image_pil = TF.to_pil_image(im_tensor.squeeze(0))

        # 保存图像
        image_pil.save(os.path.join(save_dir, f'image_{i}.png'))


