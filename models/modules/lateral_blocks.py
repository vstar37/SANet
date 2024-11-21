import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
from config import Config
from models.modules.decoder_blocks import BasicDecBlk
from sklearn.cluster import DBSCAN

from sklearn.neighbors import NearestNeighbors

config = Config()


# KPConvLatBlk class
class KPConvLatBlk(nn.Module):
    def __init__(self, out_channels=64, index=0):
        super(KPConvLatBlk, self).__init__()
        self.index = index
        self.conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.decoder = BasicDecBlk(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):

        dog = difference_of_gaussians(x, kernel_size=3, sigma1=3.0 * (0.9 ** self.index),
                                      sigma2=3.0 * (0.9 ** (self.index + 1)), index=self.index)

        mask = dog_dynamic_threshold(dog)
        mask = expand_mask(mask)
        masked_x = x * mask
        decoded_features = self.decoder(masked_x)
        output = decoded_features
        return output


class GussianLatBlk(nn.Module):
    def __init__(self, out_channels=64, index=0):
        super(GussianLatBlk, self).__init__()
        # self.conv = nn.Conv2d(1, out_channels, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.index = index

    def forward(self, x):
        dog = difference_of_gaussians(x=x, kernel_size=3, sigma1=3.0 * (0.9 ** self.index),
                                      sigma2=3.0 * (0.9 ** (self.index + 1)), index=self.index)
        # dog = self.conv(dog)
        dog = 2.0 * min_max_norm(dog)  # 这里应该用normal 而非 sigmoid
        return dog


def dog_dynamic_threshold(dog, sigma_factor=3, num_samples=1000, downsample_factor=2):
    """
    对 DOG tensor 进行动态阈值处理，先归一化，再通过多级处理减少计算复杂度。
    sigma_factor：±σ原则的 σ 值，用于异常值检测。
    downsample_factor: 下采样比例
    """
    # Step 1: 下采样以减少计算量
    downsampled_dog = F.avg_pool2d(dog, kernel_size=downsample_factor, stride=downsample_factor)
    normalized_dog = abs_norm(downsampled_dog)

    # Step 2: 计算均值和标准差，应用 ±3σ 原则过滤异常值
    mean = normalized_dog.mean(dim=(2, 3), keepdim=True)
    std = normalized_dog.std(dim=(2, 3), keepdim=True)
    lower_bound = mean - sigma_factor * std
    upper_bound = mean + sigma_factor * std
    mask = (normalized_dog >= lower_bound) & (normalized_dog <= upper_bound)
    filtered_dog = normalized_dog * mask.float()

    # Step 3: 计算非零点的排序和分位数
    flat_filtered_dog = filtered_dog.view(filtered_dog.size(0), -1)
    sorted_values, indices = torch.sort(flat_filtered_dog, dim=1)

    # 寻找第一个非零元素的索引
    first_nonzero_index = (sorted_values > 0).nonzero(as_tuple=True)[1].min()

    if first_nonzero_index.numel() == 0:
        print(r'ERROR! KSIP Can\'t find any mask!')
        return torch.zeros_like(filtered_dog)

    # 从第一个非零元素到最后的所有点
    nonzero_sorted_values = sorted_values[:, first_nonzero_index:]
    quantile_index = int(0.2 * nonzero_sorted_values.size(1))
    threshold_value = nonzero_sorted_values[:, quantile_index]

    # Step 4: 从20%点开始均匀抽样num_samples个点
    valid_points = flat_filtered_dog[flat_filtered_dog >= threshold_value.unsqueeze(1)]
    valid_indices = torch.nonzero(flat_filtered_dog >= threshold_value.unsqueeze(1), as_tuple=True)
    num_valid_points = valid_points.size(0)

    if num_samples <= num_valid_points:
        step = num_valid_points // num_samples
        sampled_indices = valid_indices[1][::step][:num_samples]
    else:
        sampled_indices = valid_indices[1]

    # Step 5: 构建最终的掩膜
    final_mask = torch.zeros_like(filtered_dog)
    # 构建一个索引张量，形状为 [batch_size, num_elements]
    batch_indices = valid_indices[0]
    final_mask.view(final_mask.size(0), -1).scatter_(1, sampled_indices.unsqueeze(0).expand(final_mask.size(0), -1), 1)

    final_mask = final_mask.view(downsampled_dog.size(0), downsampled_dog.size(1), *downsampled_dog.shape[2:])
    upsampled_mask = F.interpolate(final_mask, size=dog.shape[-2:], mode='bilinear', align_corners=False)

    return upsampled_mask

#dog动态计算阈值，并过滤掉低于阈值的点
def dog_dynamic_threshold_pointcloud(dog, sigma_factor=3, num_samples=1000):
    """
    对 DOG tensor 进行动态阈值处理，先归一化，然后排除异常值和后20%。
    sigma_factor：±σ原则的 σ 值，用于异常值检测。
    """
    # 归一化 DOG tensor
    normalized_dog = abs_norm(dog)

    # 计算均值和标准差
    mean = normalized_dog.mean(dim=(2, 3), keepdim=True)
    std = normalized_dog.std(dim=(2, 3), keepdim=True)

    # 根据 ±3σ 原则计算阈值
    lower_bound = mean - sigma_factor * std
    upper_bound = mean + sigma_factor * std

    # 创建掩膜，过滤异常值
    mask = (normalized_dog >= lower_bound) & (normalized_dog <= upper_bound)
    filtered_dog = normalized_dog * mask.float()

    # 计算所有点的分位数阈值 (最小20%点)
    sorted_values, _ = torch.sort(filtered_dog.view(filtered_dog.size(0), -1), dim=1)
    quantile_index = int(0.2 * sorted_values.size(1))  # 获取最小20%点的索引
    threshold_value = sorted_values[:, quantile_index].view(-1, 1, 1, 1)  # 适应形状


    # 过滤掉低于阈值的点
    final_mask = filtered_dog.clone()  # 复制一个新的tensor，保持原有数值
    final_mask[filtered_dog < threshold_value] = 0  # 将小于阈值的点设置为0，其他点保持原值

    # 将 tensor 展平为 2D 形状以便于抽样
    flat_dog = final_mask.view(final_mask.size(0), -1)  # [batch_size, num_elements]

    # 筛选出非零点的索引
    nonzero_indices = (flat_dog != 0).nonzero(as_tuple=True)

    # 确保我们有足够的点进行抽样
    if len(nonzero_indices[0]) > num_samples:
        # 从非零点中随机抽取 num_samples 个点
        sampled_indices = torch.randperm(len(nonzero_indices[0]))[:num_samples]
        sampled_mask = torch.zeros_like(final_mask)
        sampled_mask.view(final_mask.size(0), -1)[
            nonzero_indices[0][sampled_indices], nonzero_indices[1][sampled_indices]] = \
        final_mask.view(final_mask.size(0), -1)[
            nonzero_indices[0][sampled_indices], nonzero_indices[1][sampled_indices]]
        return sampled_mask
    else:
        return final_mask

def gussian_prompting(x, kernel_size, sigma1, index):
    blur1 = gaussian_conv(x, kernel_size=kernel_size, sigma=sigma1, id=index)
    prompt = x - blur1
    return prompt


def difference_of_gaussians(x, kernel_size, sigma1, sigma2, index):
    blur1 = gaussian_conv(x, kernel_size=kernel_size, sigma=sigma1, id=index)
    blur2 = gaussian_conv(x, kernel_size=kernel_size, sigma=sigma2, id=index)
    dog = blur1 - blur2
    # print('dog.shape = {}'.format(dog.shape))
    dog = abs_norm(dog)

    return dog


def gaussian_conv(x, kernel_size, sigma, id):
    channels = x.size(1)  # 获取输入特征图的通道数
    if config.gus_ker_type == '2d':
        gaussian_kernel = _get_2d_kernel(kernel_size, sigma)
        gaussian_kernel = np.float32(gaussian_kernel)

        # 调整高斯核以匹配输入通道数，且每个通道独立卷积
        gaussian_kernel = np.repeat(gaussian_kernel[np.newaxis, np.newaxis, ...], channels, axis=0)  # 复制核以匹配通道数
        gaussian_kernel = torch.from_numpy(gaussian_kernel).to(x.device)

        # 注意：这里 channels = x.size(1)，而不是 groups 参数
        x_smoothed = F.conv2d(x, gaussian_kernel, padding=kernel_size // 2, groups=channels)
        return x_smoothed
    else:
        gaussian_kernel = _get_3d_kernel(sigma, kernel_size, channels).to(dtype=x.dtype, device=x.device)
        x_smoothed = F.conv2d(x, gaussian_kernel, padding=kernel_size // 2, groups=1)
        return x_smoothed


def _get_2d_kernel(size, sigma):
    interval = (2 * sigma + 1.) / size
    x = np.linspace(-sigma - interval / 2., sigma + interval / 2., size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def _get_3d_kernel(sigma, size, channels):
    """
    Generate a 3D Gaussian kernel.

    Args:
    - sigma: standard deviation of the Gaussian distribution.
    - size: size of the kernel in each spatial dimension (should be odd).
    - channels: number of channels.

    Returns:
    - kernel: 3D Gaussian kernel.
    """
    kernel = np.zeros((channels, size, size))
    center = size // 2
    z_center = channels // 2
    cov = np.diag([sigma ** 2, sigma ** 2, sigma ** 2])  # Covariance matrix
    for z in range(channels):
        for x in range(size):
            for y in range(size):
                # Calculate the coordinates relative to the center
                coord = np.array([x - center, y - center, z - z_center])
                # Calculate the value of the Gaussian function at this point
                kernel[z, x, y] = st.multivariate_normal.pdf(coord, mean=np.zeros(3), cov=cov)
    kernel /= np.sum(kernel)
    return torch.tensor(kernel).unsqueeze(0)


def min_max_norm(in_):
    """
        normalization
    :param: in_
    :return:
    """
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-8)


def cluster_keypoints_dbscan(mask, dog, eps=5, min_samples=5):
    """
    聚类剩余的关键点，使用 DBSCAN 算法基于数值和空间距离划分点簇，并找到点簇的质心。

    参数:
    mask: 动态阈值过滤后的 mask，保留高特征值点的位置。
    dog: DOG 提取的高频特征图，保持与 mask 尺寸一致。
    eps: DBSCAN 聚类的邻域半径。
    min_samples: DBSCAN 聚类的最小核心点邻居数。

    返回:
    centroids: 提取的点簇质心列表。
    """

    # Step 1: 将 mask 中非零点提取出来，找到这些点的坐标及其在 dog 中的对应数值
    indices = torch.nonzero(mask, as_tuple=False)  # 提取所有非零点的坐标 (b, c, h, w)
    values = dog[mask > 0]  # 获取这些点在 dog 中对应的数值，返回值是一维tensor

    # 计算动态阈值
    std_v = values.std()
    dynamic_threshold = 0.8 * std_v  # alpha * std_v

    # Step 2: 将坐标转换为 numpy 格式以供 DBSCAN 使用
    indices_np = indices.cpu().numpy()

    # Step 3: 使用 DBSCAN 进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters_labels = dbscan.fit_predict(indices_np)

    # Step 4: 根据聚类结果筛选出不同簇的质心
    centroids = []
    for cluster_id in set(clusters_labels):
        if cluster_id == -1:
            continue  # 跳过噪声点
        cluster_indices = indices[clusters_labels == cluster_id]

        # 获取当前簇内的点在 dog 中对应的值
        cluster_values = torch.tensor([dog[tuple(point)].item() for point in cluster_indices])

        # Step 5: 计算簇的中位数点，作为质心
        median_value = torch.median(cluster_values)
        closest_point_idx = torch.argmin(torch.abs(cluster_values - median_value))
        centroid = cluster_indices[closest_point_idx]
        centroids.append(centroid)

    return centroids

def create_binary_mask(keypoints, shape, radius=3):
    """
    根据关键点坐标生成二值化掩膜矩阵
    :param keypoints: 质心坐标列表 [(b, c, h, w), ...]
    :param shape: 输入特征图的形状 (batch_size, channels, height, width)
    :param radius: 质心周围的区域半径，默认 3 表示 3x3 的局部范围
    :return: 二值化掩膜矩阵，形状与输入特征图一致
    """
    batch_size, channels, height, width = shape
    # 初始化与输入特征图形状一致的二值掩膜矩阵
    binary_mask = torch.zeros(shape, dtype=torch.float32)

    # 遍历每个质心坐标
    for (b, c, h, w) in keypoints:
        # Step 1: 确定高度和宽度的局部区域，限制在特征图的边界内
        h_min = max(h - radius, 0)
        h_max = min(h + radius, height - 1)
        w_min = max(w - radius, 0)
        w_max = min(w + radius, width - 1)

        # Step 2: 确定通道的局部范围，限制在通道数边界内
        c_min = max(c - radius, 0)
        c_max = min(c + radius, channels - 1)

        # Step 3: 将质心周围的区域 (在通道、高度、宽度三个维度上) 设置为 1
        binary_mask[b, c_min:c_max + 1, h_min:h_max + 1, w_min:w_max + 1] = 1

    return binary_mask


def expand_mask(mask):
    """
    对值为1的点进行3*3*3扩散。
    :param mask: 输入掩膜，形状为 [batch_size, channels, height, width]，值为0或1
    :return: 扩散后的掩膜
    """
    # 扩展卷积核 [1, 1, 3, 3, 3]，表示只对当前值为1的点进行扩散
    kernel = torch.ones((1, 1, 3, 3, 3), device=mask.device)

    # 增加一个批次和通道维度
    mask = mask.unsqueeze(1)  # [batch_size, 1, channels, height, width]

    # 使用F.conv3d对掩膜进行卷积，保持同样大小 (padding=1)
    expanded_mask = F.conv3d(mask, kernel, padding=1)

    # 将卷积后非零值区域设置为1，保持二值化
    expanded_mask = (expanded_mask > 0).float()

    # 去掉增加的通道维度
    expanded_mask = expanded_mask.squeeze(1)

    return expanded_mask

def abs_norm(in_):
    """
    Normalize the input based on absolute values, ensuring that larger absolute values approach 1
    and smaller absolute values approach 0.

    :param in_: Input tensor
    :return: Normalized tensor
    """
    # Step 1: Take the absolute values of the input
    abs_in = in_.abs()

    # Step 2: Get the max and min values along the last two dimensions (height and width)
    max_abs = abs_in.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(abs_in)
    min_abs = abs_in.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(abs_in)

    # Step 3: Perform min-max normalization on the absolute values
    norm_in = (abs_in - min_abs) / (max_abs - min_abs + 1e-8)

    return norm_in


