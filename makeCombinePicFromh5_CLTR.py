# 功能：用于CLTR模型中，将.h5文件中的image和kpoint数据转换成原图片和其密度图的融合图
# 具体操作：
# 1.读取CLTR/npydata/文件下，要转换数据集的npy文件（里面放的是该数据集的所有图片路径）
# 2.读取数据集图片路径，并获取其相对应的.h5文件（该.h5文件是用CLTR模型的prepare_nwpu.py或prepare_jhu.py生成的，里面存放的是该图片的image和kpoint，image是reshape后的图片数据，kpoint是该图片人头标签点）
# 3.将.h5文件中的image和kpoint读取出来，并用kpoint去做密度图
# 4.将密度图和原图image融合成一张图，并且保存密度图和融合图

import os.path

import PIL.Image
import cv2
import numpy
import numpy as np
import h5py
from matplotlib import pyplot as plt
from matplotlib import cm as CM

import readnpy as rn
import generateDensityMap as gdm
import readh as rh


def combine_source_pic(npy_file_path, density_img_to):
    # 密度图和原图融合图的文件夹
    combine_den_img_to = f'{density_img_to}/combine'

    # 文件夹创建
    if not os.path.exists(density_img_to):
        os.mkdir(density_img_to)
    if not os.path.exists(combine_den_img_to):
        os.mkdir(combine_den_img_to)

    # 读取要处理的h5文件
    print('img_list')
    img_list = rn.read_npy(npy_file_path)
    print('h5_list')
    h5_list = [file.replace('images_2048', 'gt_detr_map_2048').replace('jpg', 'h5') for file in img_list]
    print(f'{len(h5_list)}\n{h5_list}')

    # 开始处理文件
    for file_path in h5_list:
        # 保存路径处理
        _, file_name = os.path.split(file_path)
        file_name = file_name.replace('h5', 'jpg')
        density_img_to_path = f'{density_img_to}/{file_name}'
        combine_den_img_to_path = f'{combine_den_img_to}/{file_name}'
        # HDF5的读取：
        h5files = rh.main(file_path)
        print(h5files)
        img = h5files['/image']
        kpoint = h5files['/kpoint']
        img = np.array(img)
        kpoint_data = np.array(kpoint, dtype=numpy.float64)
        # non_zero = [np.nonzero(data)]
        # print(non_zero)
        # 获取密度图
        den_map = gdm.gaussian_filter_density(kpoint_data)
        # non_zero2 = [np.nonzero(den_map)]
        # print(non_zero2)

        # 密度图展示和保存
        plt.figure(2)
        plt.imshow(den_map, cmap=CM.jet)
        plt.axis('off')
        plt.savefig(density_img_to_path, bbox_inches='tight', pad_inches=0)
        # plt.show()

        # 将密度图和原图融合保存
        heatmap = cv2.imread(density_img_to_path)
        combine = cv2.addWeighted(cv2.resize(heatmap, (img.shape[1], img.shape[0])), 0.5, img, 0.5, 0)
        cv2.imwrite(combine_den_img_to_path, combine)


if __name__ == '__main__':
    # 基本路径
    npy_file = 'jhu_test_try'  # 要做处理的npy文件名
    npy_file_path = f'../CLTR/npydata/{npy_file}.npy'  # npy路径
    density_img_to = '../jhu_crowd_v2.0-001/jhu_crowd_v2.0/test/test_try_density'  # 得到的密度图存放位置

    combine_source_pic(npy_file_path, density_img_to)
