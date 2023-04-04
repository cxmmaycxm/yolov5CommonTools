# 功能：将指定图片转成密度图
# 特点：使用了自适应高斯核，头部大小通常与拥挤场景中两个头部中心距离有关，所以使用相邻头部的平均距离作为参数。sigmas与头部间距离成正比。
# 参考链接：https://zhuanlan.zhihu.com/p/39424587

import numpy as np
import os
import matplotlib.image as cv2
import scipy.io as sio
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import math
from matplotlib import cm as CM

#   字体颜色控制
red_begin = '\033[1;31;40m'
green_begin = '\033[1;32;40m'
yellow_begin = '\033[1;33;40m'
blue_begin = '\033[1;34;40m'
color_end = '\033[0m'

error_list = []
warning_list = []


# 自适应高斯卷积核生成密度图
def gaussian_filter_density(gts, d_map_h, d_map_w):
    # 筛选出合规的有标签点（不合规：超出图片范围；标签点为0，既点处无人头标签）
    res = np.zeros(shape=[d_map_h, d_map_w])
    bool_res = (gts[:, 0] < d_map_w) & (gts[:, 1] < d_map_h) if len(gts) > 0 else True
    for k in range(len(gts)):
        gt = gts[k]
        if bool_res[k]:
            res[int(gt[1])][int(gt[0])] = 1
    pts = np.array(list(zip(np.nonzero(res)[1], np.nonzero(res)[0])))
    # 先挑出检测人数为1或者无人的预测结果
    use_kdtree = True
    distances = np.array([[0]])
    pts_len = len(pts)
    if pts_len < 2:
        use_kdtree = False
    # 使用KDTree算法选出最邻近4个点，利用其计算出sigmas（4点与参考点距离之和*0.075）（如果没有4个点，就取图总数量-1个临近点）
    if use_kdtree:
        neig = 4 if pts_len > 4 else pts_len - 1
        print(f'{yellow_begin}points\'s num: {pts_len}, n_neighbors : {neig}{color_end}') if neig is not 4 else None
        neighbors = NearestNeighbors(n_neighbors=neig, algorithm='kd_tree', leaf_size=1200)
        neighbors.fit(pts.copy())
        distances, _ = neighbors.kneighbors()
    map_shape = [d_map_h, d_map_w]
    density = np.zeros(shape=map_shape, dtype=np.float32)
    sigmas = distances.sum(axis=1) * 0.075
    # 对有标签的每个点使用高斯函数（sigma自适应），得出密度图
    for i in range(len(pts)):
        pt = pts[i]
        pt2d = np.zeros(shape=map_shape, dtype=np.float32)
        pt2d[pt[1]][pt[0]] = 1
        density += filters.gaussian_filter(pt2d, sigmas[i], mode='constant')
    return density


# 数据标签文件来自txt，文件中存放点的x,y,w,h（相对图片位置）
def create_density_txt(img_from_path, txt_from_path):
    # 图片和标签文件读取
    img = cv2.imread(img_from_path)
    data = []
    with open(txt_from_path, 'r', encoding='utf-8') as f:
        data_list = f.readlines()
    img_h = float(img.shape[0])
    img_w = float(img.shape[1])
    for temp_data in data_list:
        temp_data_list = temp_data.split(' ')
        x = float(temp_data_list[1]) * img_w
        y = float(temp_data_list[2]) * img_h
        location = [x, y]
        data.append(location)
    gts = np.array(data)

    # 将图片缩小1/4后，生成密度图
    d_map_h = math.floor(math.floor(img_h / 2.0) / 2.0)
    d_map_w = math.floor(math.floor(img_w / 2.0) / 2.0)
    den_map = gaussian_filter_density(gts / 4, d_map_h, d_map_w)

    return den_map


# 数据标签文件来自matlab，文件中存放点的x,y轴坐标
def create_density_mat(img_from_path, mat_from_path):
    # 图片和标签文件读取
    img = cv2.imread(img_from_path)
    data = sio.loadmat(mat_from_path)
    gts = data['annPoints']

    # 将图片缩小1/4后，生成密度图
    d_map_h = math.floor(math.floor(float(img.shape[0]) / 2.0) / 2.0)
    d_map_w = math.floor(math.floor(float(img.shape[1]) / 2.0) / 2.0)
    den_map = gaussian_filter_density(gts / 4, d_map_h, d_map_w)

    # 以原图大小来生成密度图（速度会比上面1/4时慢）
    # d_map_h = math.floor(img.shape[0])
    # d_map_w = math.floor(img.shape[1])
    # den_map = create_density(gts, d_map_h, d_map_w)
    return den_map


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
        print(f'{red_begin}error:{color_end}{err_msg}')
        error_list.append(err_msg)
    elif not os.path.exists(label_from_path):
        err_msg = f'{label_from_path} is not exist'
        print(f'{red_begin}error:{color_end}{err_msg}')
        error_list.append(err_msg)
    elif os.path.exists(density_img_to_path):
        warning_msg = f'{density_img_to_path} is exist'
        print(f'{yellow_begin}warning:{color_end}{warning_msg}')
        warning_list.append(warning_msg)
    else:
        print(f'{green_begin}begin create density map : {color_end} {density_img_to_path}')
        # 通过mat或txt中的人头标签进行密度图生成
        den_map = create_density_mat(img_from_path, label_from_path) if is_mat else create_density_txt(img_from_path,label_from_path)

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
    is_mat = False
    txts_from_path = f"../labels"  # 待检测图片标签文件路径(txt/mat)
    imgs_from_path = f"../jhu_crowd_v2.0_yolo/images/exp"  # 待检测图片路径
    density_imgs_to_path = f"../density_imgs"  # 密度图存放路径
    combine_den_img_to_path = f'{density_imgs_to_path}/combine' # 存放密度图和原图融合的新图的地方

    # create_map(img_from_path, txt_from_path, density_img_to_path, combine_den_img_to_path, is_mat=False)
    create_maps(imgs_from_path, txts_from_path, density_imgs_to_path, is_mat)  # is_mat:标签文件 is mat/txt文件

    print(f'{yellow_begin}warning message list{color_end}')
    print(warning_list)

    print(f'{red_begin}error message list{color_end}')
    print(error_list)
