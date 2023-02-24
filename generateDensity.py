# 这个文件用的Python版本是python3.7，用Python3.10出了些问题
# 功能：将指定图片转成密度图

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import cv2
import scipy.ndimage
import scipy.io


def gaussian_filter_density(gt):
    # 初始化密度图
    density = np.zeros(gt.shape, dtype=np.float32)

    # 获取gt中不为0的元素的个数
    gt_count = np.count_nonzero(gt)

    # 如果gt全为0，就返回全0的密度图
    if gt_count == 0:
        return density

    sigma = 16
    density += scipy.ndimage.filters.gaussian_filter(gt, sigma, mode='constant')
    return density


def density_map(img, gt):
    k = np.zeros((img.shape[0], img.shape[1]))
    for i in range(len(gt)):  # 生成头部点注释图
        if gt[i][0] < img.shape[1] and gt[i][1] < img.shape[0]:
            k[int(gt[i][1])][int(gt[i][0])] += 1
    k = gaussian_filter_density(k)
    return k


if __name__ == '__main__':
    #   路径设置
    mat_from_path = f"待检测图片mat标签文件路径"
    img_from_path = f"待检测图片路径"
    density_img_to_path = f"密度图存放路径"

    gt = scipy.io.loadmat(mat_from_path)
    img = cv2.imread(img_from_path)

    groundtruth = density_map(img, gt['annPoints'])
    plt.figure(2)

    plt.imshow(groundtruth, cmap=CM.jet)
    plt.savefig(density_img_to_path)
    plt.show()


