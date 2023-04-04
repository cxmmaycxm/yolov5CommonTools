# 这个文件用的Python版本是python3.7，用Python3.10出了些问题
# 功能：将指定图片转成密度图
# 参考链接：https://blog.csdn.net/qq_37713366/article/details/108276460

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import cv2
import scipy.io
import os
import matplotlib.image as pltImg
import scipy.io as sio
from scipy.ndimage import filters
from PIL import Image
import math
import color_set as c

error_list = []
warning_list = []


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
    non_zero2 = [np.nonzero(density)]
    return density


def density_map(img, gt):
    k = np.zeros((img.shape[0], img.shape[1]))
    for i in range(len(gt)):  # 生成头部点注释图
        if gt[i][0] < img.shape[1] and gt[i][1] < img.shape[0]:
            k[int(gt[i][1])][int(gt[i][0])] += 1
    k = gaussian_filter_density(k)
    return k


# 数据标签文件来自txt，文件中存放点的x,y,w,h（相对图片位置）
def create_density_txt(img_from_path, txt_from_path):
    # 图片和标签文件读取
    img = pltImg.imread(img_from_path)
    data = []
    with open(txt_from_path, 'r', encoding='utf-8') as f:
        data_list = f.readlines()
    img_h = float(img.shape[0])
    img_w = float(img.shape[1])
    for temp_data in data_list:
        temp_data_list = temp_data.split(' ')
        # x = float(temp_data_list[1]) * img_w
        # y = float(temp_data_list[2]) * img_h

        x = float(temp_data_list[0])
        y = float(temp_data_list[1])

        location = [x, y]
        data.append(location)
    gts = np.array(data)
    return density_map(img, gts)


# 数据标签文件来自matlab，文件中存放点的x,y轴坐标
def create_density_mat(img_from_path, mat_from_path):
    # 图片和标签文件读取
    img = pltImg.imread(img_from_path)
    data = sio.loadmat(mat_from_path)
    gts = data['annPoints']
    return density_map(img, gts)


# 创建多张密度图
def create_maps(imgs_from_path, labels_from_path, density_imgs_to_path, is_mat=False):
    #   路径设置
    imgs = os.listdir(imgs_from_path)
    imgs.sort()
    files_name = [img.split('.')[0] for img in imgs]
    img_suffix = imgs[0].split('.')[1]
    suffix = 'mat' if is_mat else 'txt'
    combine_dens_imgs_to_path = f'{density_imgs_to_path}/combine'

    if not os.path.exists(density_imgs_to_path):
        os.mkdir(density_imgs_to_path)
    if not os.path.exists(combine_dens_imgs_to_path):
        os.mkdir(combine_dens_imgs_to_path)

    for file_name in files_name:
        img_from_path = f'{imgs_from_path}/{file_name}.{img_suffix}'
        label_from_path = f'{labels_from_path}/{file_name}.{suffix}'
        density_img_to_path = f'{density_imgs_to_path}/{file_name}_density.jpg'
        combine_den_img_to_path = f'{combine_dens_imgs_to_path}/{file_name}_density.jpg'
        create_map(img_from_path, label_from_path, density_img_to_path, combine_den_img_to_path, is_mat)


# 创建单张密度图
def create_map(img_from_path, label_from_path, density_img_to_path, combine_den_img_to_path, is_mat=False):
    # 判断文件是否存在
    if not os.path.exists(img_from_path):
        err_msg = f'{img_from_path} is not exist'
        print(f'{c.red_begin}error:{c.color_end}{err_msg}')
        error_list.append(err_msg)
    elif not os.path.exists(label_from_path):
        err_msg = f'{label_from_path} is not exist'
        print(f'{c.red_begin}error:{c.color_end}{err_msg}')
        error_list.append(err_msg)
    elif os.path.exists(density_img_to_path):
        warning_msg = f'{density_img_to_path} is exist'
        print(f'{c.yellow_begin}warning:{c.color_end}{warning_msg}')
        warning_list.append(warning_msg)
    else:
        print(f'{c.green_begin}begin create density map : {c.color_end} {density_img_to_path}')
        # 通过mat或txt中的人头标签进行密度图生成
        den_map = create_density_mat(img_from_path, label_from_path) if is_mat else create_density_txt(img_from_path,
                                                                                                       label_from_path)
        # 密度图展示和保存
        plt.figure(2)
        plt.imshow(den_map, cmap=CM.jet)
        plt.axis('off')
        plt.savefig(density_img_to_path, bbox_inches='tight', pad_inches=0)
        # plt.show()

        # 将密度图和原图融合保存
        heatmap = cv2.imread(density_img_to_path)
        img = cv2.imread(img_from_path)

        combine = cv2.addWeighted(cv2.resize(heatmap, (img.shape[1], img.shape[0])), 0.5, img, 0.5, 0)
        cv2.imwrite(combine_den_img_to_path, combine)
        # cv2.imshow('1', combine)
        # cv2.waitKey(0)


if __name__ == '__main__':
    # 路径设置
    is_mat = False
    txts_from_path = f"../labels"  # 待检测图片标签文件路径(txt/mat)
    imgs_from_path = f"../jhu_crowd_v2.0_yolo/images/exp"  # 待检测图片路径
    density_imgs_to_path = f"../density_imgs"  # 密度图存放路径
    combine_den_img_to_path = f'{density_imgs_to_path}/combine'  # 存放密度图和原图融合的新图的地方

    # create_map(img_from_path, txt_from_path, density_img_to_path, combine_den_img_to_path, is_mat=False)
    create_maps(imgs_from_path, txts_from_path, density_imgs_to_path, is_mat)  # is_mat:标签文件 is mat/txt文件

    print(f'{c.yellow_begin}warning message list{c.color_end}:{len(warning_list)}')
    print(warning_list)

    print(f'{c.red_begin}error message list{c.color_end}:{len(error_list)}')
    print(error_list)


