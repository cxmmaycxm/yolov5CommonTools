"""
    实现效果：将YOLOv5中detect出来的结果进行转换
    原格式：一张图片对应一个同名txt，txt中存储检测结果
        例如：3610.jpg --> labels/3610.txt（3610.txt中共有316行），3611.jpg --> labels/3611.txt（3611.txt中共有30行）
            3610.txt :  0 0.480452 0.509121 0.0191138 0.029316          -----
                        0 0.884883 0.468404 0.0199826 0.0221498             |
                        0 0.306907 0.471661 0.016073 0.0273616              |——> 316行
                        ...                                                 |
                        0 0.388575 0.421498 0.00825369 0.0169381        -----

            3611.txt :  0 0.77725 0.36475 0.00483333 0.008              -----
                        0 0.830083 0.365375 0.00516667 0.00925              |
                        0 0.679583 0.398125 0.00516667 0.00875              |——> 30行
                        ...                                                 |
                        0 0.524917 0.441375 0.00616667 0.01075          -----

    转换后：只有一个detectResult.txt，一行存放一张图片的检测结果
        例如： detectResult.txt  :   3610 316
                                    3611 30
                                    ...
"""

import os

#   字体颜色控制
red_begin = '\033[1;31;40m'
green_begin = '\033[1;32;40m'
yellow_begin = '\033[1;33;40m'
blue_begin = '\033[1;34;40m'
color_end = '\033[0m'


#   获取指定文件夹下的所有文件名
def get_file_name(path):
    file_name = os.listdir(path)
    return file_name


#   获取所有检测图片的名字
def get_img_name(path):
    images = get_file_name(path)
    suffixs = ['jpg', 'png', 'jpeg', 'gif', 'bmp']
    #   整理图片文件，删除后缀
    imgs = [img.split('.')[0] for img in images if img.split('.')[1] in suffixs]
    #   将图片按名字排好序
    imgs.sort()
    print(f'-- {blue_begin} the number of images : {color_end} {len(imgs)}')
    print(f'-- images : {imgs} ')
    return imgs


def get_labels(path, imgs_name):
    labels = {}
    for name in imgs_name:
        labels_path = f'{path}/{name}.txt'
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                labels[name] = len(f.readlines())
                f.close()
        else:
            labels[name] = 0

    print(f'-- {blue_begin} labels : {color_end} {labels}')
    return labels


def save_file(path, content):
    with open(path, 'w') as f:
        f.write(content)
        print(f'-- 写入成功')
        f.close()


def save_labels(path, labels):
    content = ""
    for name in labels.keys():
        content = f'{content}{name} {labels[name]}\n'
    save_file(path, content)


#   入口函数
def main():
    #   路径设置
    detect_result_save_file = 'detectResult.txt'
    img_from_path = '检测图片的路径'
    label_from_path = f'检测结果标签存放路径'
    label_to_path = f'最后结果文件存放处'
    #   获取所有检测图片的名字
    imgs_name = get_img_name(img_from_path)
    #   读取检测结果
    labels = get_labels(label_from_path, imgs_name)
    #   保存检测结果
    save_labels(f'{label_to_path}/{detect_result_save_file}', labels)


#   测试性能专用函数
def test():
    print('Test time.\n')
    labels_name = ['111.jpg', '22.png', '36.jpg']
    suffix = ['png', 'jpg', 'jpeg', 'bmp']
    print([name.split('.')[0] for name in labels_name if name.split('.')[1] in suffix])


if __name__ == '__main__':
    main()
    # test()
