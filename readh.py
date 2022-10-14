import cv2
import numpy as np
import h5py

if __name__ == '__main__':
    file_path = '/media/cxm/PapersCode/datasets/UCF-QNRF/UCF-QNRF_ECCV18/train_data/Images/00_img_0001.h5'
    # HDF5的读取：
    f = h5py.File(file_path, 'r')  # 打开h5文件
    # 可以查看所有的主键
    for key in f.keys():
        print(f[key].name)
        print(f[key].shape)
        # print(f[key].value)

    # # 遍历文件中的一级组
    # for group in f.keys():
    #     print(group)
    #     # 根据一级组名获得其下面的组
    #     group_read = f[group]
    #     # 遍历该一级组下面的子组
    #     for subgroup in group_read.keys():
    #         print(subgroup)
    #         # 根据一级组和二级组名获取其下面的dataset
    #         dset_read = f[group + '/' + subgroup]
    #         # 遍历该子组下所有的dataset
    #         for dset in dset_read.keys():
    #             # 获取dataset数据
    #             dset1 = f[group + '/' + subgroup + '/' + dset]
    #             print(dset1.name)
    #             data = np.array(dset1)
    #             print(data.shape)
    #             x = data[..., 0]
    #             y = data[..., 1]
