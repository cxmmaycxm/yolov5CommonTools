import numpy as np


def main():
    file_path = '你npy文件的完整地址.npy'
    file = np.load(file_path)
    print(file)


if __name__ == '__main__':
    main()
