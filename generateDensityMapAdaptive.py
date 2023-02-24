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


# 自适应高斯卷积核生成密度图
def create_density(gts, d_map_h, d_map_w):
    # 筛选出合规的有标签点（不合规：超出图片范围；标签点为0，既点处无人头标签）
    res = np.zeros(shape=[d_map_h, d_map_w])
    bool_res = (gts[:, 0] < d_map_w) & (gts[:, 1] < d_map_h)
    for k in range(len(gts)):
        gt = gts[k]
        if bool_res[k]:
            res[int(gt[1])][int(gt[0])] = 1
    pts = np.array(list(zip(np.nonzero(res)[1], np.nonzero(res)[0])))
    # 使用KDTree算法选出最邻近4个点，利用其计算出sigmas（4点与参考点距离之和*0.075）
    neighbors = NearestNeighbors(n_neighbors=4, algorithm='kd_tree', leaf_size=1200)
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


if __name__ == '__main__':
    #   路径设置
    mat_from_path = f"待检测图片mat标签文件路径"
    img_from_path = f"待检测图片路径"
    density_img_to_path = f"密度图存放路径"

    # 图片和标签文件读取
    img = cv2.imread(img_from_path)
    data = sio.loadmat(mat_from_path)
    gts = data['annPoints']
    shape = img.shape

    # 将图片缩小1/4后，生成密度图
    d_map_h = math.floor(math.floor(float(img.shape[0]) / 2.0) / 2.0)
    d_map_w = math.floor(math.floor(float(img.shape[1]) / 2.0) / 2.0)
    den_map = create_density(gts / 4, d_map_h, d_map_w)

    # 以原图大小来生成密度图（速度会比上面1/4时慢）
    # d_map_h = math.floor(img.shape[0])
    # d_map_w = math.floor(img.shape[1])
    # den_map = create_density(gts, d_map_h, d_map_w)

    # 密度图展示和保存
    plt.figure(2)
    plt.imshow(den_map, cmap=CM.jet)
    plt.savefig(density_img_to_path)
    plt.show()
