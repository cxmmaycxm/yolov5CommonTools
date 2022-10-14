import numpy as np


def main():
    file_path = '/media/cxm/PapersCode/datasets/NWPU_localization/NWPU_localization/gt_npydata_2048/0001.npy'
    file = np.load(file_path)
    print(file)


if __name__ == '__main__':
    main()
