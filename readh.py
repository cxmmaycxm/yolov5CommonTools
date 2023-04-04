import cv2
import numpy as np
import h5py

def main(file_path):
    print(file_path)
    # HDF5的读取：
    f = h5py.File(file_path, 'r')  # 打开h5文件
    direc = {}
    # 可以查看所有的主键
    for key in f.keys():
        # print(f[key].name)
        # print(f[key].shape)
        # print(f[key].value)
        data = np.array(f[key])
        x = data[..., 0]
        y = data[..., 1]
        # print(x)
        # print(y)
        direc[f[key].name] = f[key]
    return direc

if __name__ == '__main__':
    file_path = '你h5文件的完整路径.h5'
    main(file_path)
