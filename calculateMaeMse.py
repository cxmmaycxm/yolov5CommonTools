# 功能：计算MAE/MSE
import math
import os.path
from yoloDetectResult import  get_detect_result,detect_result

#   字体颜色控制
red_begin = '\033[1;31;40m'
green_begin = '\033[1;32;40m'
yellow_begin = '\033[1;33;40m'
blue_begin = '\033[1;34;40m'
color_end = '\033[0m'

# 计算源数据和检测数据每张图的总人数
def count_source_detect_people(source_data_path, result_save_file, source_file_from_path, file_suffixs, detect_data_path, detect_label_from, detect_path):
    source_file_names, source_labels = [], []
    detect_file_names, detect_labels = [], []
    if os.path.exists(source_data_path):
        source_file_names, source_labels = get_detect_result(source_file_from_path, file_suffixs, source_file_from_path, source_file_from_path, result_save_file)
        print(f'{red_begin}源文件的{result_save_file}文件已存在{color_end}')
    else:
        source_file_names, source_labels = detect_result(source_file_from_path, file_suffixs, source_file_from_path, source_file_from_path, result_save_file)
        print(f'{green_begin}源文件{result_save_file}写入成功{color_end}')

    if os.path.exists(detect_data_path):
        detect_file_names, detect_labels = get_detect_result(detect_label_from, file_suffixs, detect_label_from, detect_path, result_save_file)
        print(f'{red_begin}检测文件的{result_save_file}文件已存在{color_end}')
    else:
        detect_file_names, detect_labels = detect_result(detect_label_from, file_suffixs, detect_label_from, detect_path, result_save_file)
        print(f'{green_begin}检测文件{result_save_file}写入成功{color_end}')
    return source_file_names, source_labels, detect_file_names, detect_labels


def main():
    mae = 0.0
    mse = 0.0
    file_suffixs = ['txt']

    # 路径设置
    detect_exp = 'exp30'
    detect_path = f'../yolov5/runs/detect/{detect_exp}' # yolov5检测结果路径
    source_file_from_path = '/../datasets/jhu_crowd_v2.0-001/jhu_crowd_v2.0_yolo/labels/test' # 数据集标签文件路径
    detect_label_from = f'{detect_path}/labels' # 检测结果标签文件所在地
    calculate_save_file = 'calculateMaeMse.txt'
    result_save_file = 'detectResult.txt' # 文件里面每行存放着一张图片及图片中的总人数，可由yoloDetectResult.py生成
    # 计算文件路径和保存文件路径
    source_data_path = f'{source_file_from_path}/{result_save_file}' # gt源文件的detectResult.txt路径
    detect_data_path = f'{detect_path}/{result_save_file}' # 检测结果文件的detectResult.txt路径
    calculate_to_path = f'{detect_path}/{calculate_save_file}' # 存放计算后MAE/MSE的txt文件路径

    # 统计源数据和检测数据每张图片的总人数、文件列表名
    source_file_names, source_labels, detect_file_names, detect_labels = count_source_detect_people(source_data_path, result_save_file, source_file_from_path, file_suffixs, detect_data_path, detect_label_from, detect_path)

    # 根据检测结果，计算mae和mse
    print(f'{green_begin}--calculate MAE , MSE--{color_end}')
    for file_name in detect_file_names:
        count = detect_labels[file_name]
        gt_count = source_labels[file_name]
        mae += abs(count - gt_count)
        mse += abs(count - gt_count) * abs(count - gt_count)
        print(f'{green_begin}{file_name}:{color_end}MAE:{mae}   MSE:{mse}')

    mae = mae / len(detect_file_names)
    mse = math.sqrt(mse / len(detect_file_names))

    print(f'{yellow_begin}MAE:{mae}   MSE:{mse}{color_end}')
    # 保存结果
    with open(calculate_to_path, 'w') as f:
        f.write(f'MAE:{mae}   MSE:{mse}')

    return mae, mse


if __name__ == '__main__':
    main()
