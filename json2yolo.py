import json
import os
import cv2
import numpy as np

#   字体颜色控制
red_begin = '\033[1;31;40m'
green_begin = '\033[1;32;40m'
yellow_begin = '\033[1;33;40m'
blue_begin = '\033[1;34;40m'
color_end = '\033[0m'
error_list = []  # 转换失败的图片文件列表


#   将标签转换成YOLO指定形式
def trans_labels2yolo(h_img, w_img, human_num, points, boxes, yolo_labels):
    cycle = 1
    print(f'{green_begin}weight:{color_end} {w_img}  {green_begin}height:{color_end} {h_img}')
    for temp_num in range(human_num):
        [x, y] = points[temp_num]
        [x1, y1, x2, y2] = boxes[temp_num]
        w = x2-x1
        h = y2-y1
        #   将原标签转成yolo格式的标签
        [yolo_x, yolo_y, yolo_w, yolo_h] = [x / w_img, y / h_img, w / w_img, h / h_img]
        #   非法标签检测
        for yolo_temp in [yolo_x, yolo_y, yolo_w, yolo_h]:
            if yolo_temp > 1 or yolo_temp < 0:
                print(
                    f'{red_begin}{cycle}行：{color_end}\n'
                    f'[point:({x}, {y}), boxes:({x1}, {y1}, {x2},{y2}), wh:({w}, {h})]\n'
                    f' ---> [{yolo_x}, {yolo_y}, {yolo_w}, {yolo_h}]')
                break
        #   将yolo格式标签存进list
        yolo_labels.append(f'0 {yolo_x} {yolo_y} {yolo_w} {yolo_h}')
        cycle += 1


#   展示转换后标签人头边界框位置
def show_img_boxes(img, labels, is_yolo=True):
    size = img.shape
    h_img, w_img = size[0], size[1]
    print(f'{green_begin}img size: {color_end} [{h_img}, {w_img}]')
    #   是YOLO标签是进行转化
    if is_yolo:
        for label in labels:
            temp = []
            for strip_label in label.split(' '):
                temp.append(float(strip_label))
            [obj_class, yolo_x, yolo_y, yolo_w, yolo_h] = temp
            #   展示标注图片，看看转换是否正确
            x1 = int((yolo_x - yolo_w / 2) * w_img)  # x_center - width/2
            y1 = int((yolo_y - yolo_h / 2) * h_img)  # y_center - height/2
            x2 = int((yolo_x + yolo_w / 2) * w_img)  # x_center + width/2
            y2 = int((yolo_y + yolo_h / 2) * h_img)  # y_center + height/2
            # print(x1, ",", y1, ",", x2, ",", y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
    else:
        for label in labels:
            [x, y, w, h] = label
            #   展示标注图片，看看转换是否正确
            x1 = int(x - w / 2)  # x_center - width/2
            y1 = int(y - h / 2)  # y_center - height/2
            x2 = int(x + w / 2)  # x_center + width/2
            y2 = int(y + h / 2)  # y_center + height/2

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imshow('windows', img)
    cv2.waitKey(0)


#   打开文件进行转换工作，并保存完整数据（这里默认图片是.jpg后缀，后续再考虑优化）
def open_save_labels(img_from_path, label_from_path, label_to_path, img_name):
    #   原始标签,原始标签标准化，和转换后的标签存放的list
    source_labels = []
    format_source_labels = []
    yolo_labels = []
    #   完善文件路径
    file_name = img_name.split('.')[0]
    img_from = f'{img_from_path}/{img_name}'
    label_from = f'{label_from_path}/{file_name}.json'
    label_to = f'{label_to_path}/{file_name}.txt'
    #   读取图片文件，获取图片大小
    print(f'{green_begin}----------------------{green_begin}{file_name}.jpg{color_end} --------------------{color_end}')
    img = cv2.imread(img_from)
    size = img.shape
    h_img, w_img = size[0], size[1]
    print(f'{green_begin}img size: {color_end} [{h_img}, {w_img}]')
    #   判断目标标签文件是否存在
    if os.path.isfile(label_to):
        msg = f'{yellow_begin}warning：{color_end} {file_name}.txt 已完成标签转换'
        print(msg)
        error_list.append(msg)
    #   判断原标签文件是否存在
    elif os.path.isfile(label_from):
        #   读取标签文件
        with open(label_from, 'r') as f:
            source_labels = json.load(f)
            f.close()
        #   获取标签
        human_num = source_labels['human_num']
        points = source_labels['points']
        boxes = source_labels['boxes']
        #   进行标签的转换
        trans_labels2yolo(h_img, w_img, human_num, points, boxes, yolo_labels)
        # show_img_boxes(img, yolo_labels)
        # show_img_boxes(img, format_source_labels, is_yolo=False)
        #   将转换好后的yolo格式标签存进文件里
        with open(label_to, 'w') as f:
            for label_str in yolo_labels:
                f.write(f'{label_str}\n')
            print(f'--{file_name}.txt 写入成功')
            f.close()
    else:
        msg = f'{red_begin}error：{color_end} {img_name} 无标签文件'
        print(msg)
        error_list.append(msg)


def main():
    #   路径设置
    root_path = '你数据集所在目录'
    img_from_path = f'{root_path}/images'
    label_from_path = f'{root_path}/jsons'
    label_to_path = f'{root_path}/labels'

    #   获取文件列表
    images = os.listdir(img_from_path)
    total = len(images)
    print(f'-- {blue_begin} the number of images : {color_end} {total}')
    print(f'-- images : {images} ')
    # open_save_labels(img_from_path, label_from_path, label_to_path, '0734.jpg')
    for img_name in images:
        #   进行文件后缀的判断
        suffix = img_name.split('.')[1]
        if suffix == 'jpg' or suffix == 'png':
            #   这边进行批量的文件转换，以下是单个文件的转换流程
            open_save_labels(img_from_path, label_from_path, label_to_path, img_name)
    #   转换失败文件的输入
    print(f'--{green_begin}总文件数：{total}， 失败：{len(error_list)} {color_end}')
    if len(error_list) > 0:
        print(f'{red_begin}失败列表：{color_end}')
        for msg in error_list:
            print(msg)


if __name__ == '__main__':
    main()
