# 检查图片用PIL打开是否会出错

from PIL import Image
import os
import io

#   字体颜色控制
red_begin = '\033[1;31;40m'
green_begin = '\033[1;32;40m'
yellow_begin = '\033[1;33;40m'
blue_begin = '\033[1;34;40m'
color_end = '\033[0m'
error_list = []  # 失败图片错误信息列表


# 校验图片的字节流
def check_img_byte(error_img, imgs_path, img_name):
    file_name = f'{imgs_path}/{img_name}'
    try:
        with open(file_name, 'rb') as image_file:
            image_byte = image_file.read()  # 获得图片字节流，可以从文件或数据接口获得

        image_file = io.BytesIO(image_byte)  # 使用BytesIO把字节流转换为文件对象
        image = Image.open(image_file)  # 检查文件是否能正常打开
        image.verify()  # 检查文件完整性
        image_file.close()
        image.close()
    except:
        try:
            image_file.close()
        except:
            pass
        try:
            image.close()
        except:
            error_msg = f'{red_begin}{file_name}{color_end} check_img_byte error'
            error_img.append(file_name)
            error_list.append(error_msg)
            pass
        raise
    else:
        print(f'{green_begin}{img_name}{color_end} OK')


# 校验可直接打开的文件
def check_img(error_img, imgs_path, img_name):
    file_name = f'{imgs_path}/{img_name}'
    try:
        image = Image.open(file_name)  # 检查文件是否能正常打开
        image.verify()  # 检查文件完整性
        image.close()
    except:
        try:
            image.close()
        except:
            error_msg = f'{red_begin}{file_name}{color_end} check_img error'
            error_img.append(file_name)
            error_list.append(error_msg)
            pass
        raise
    else:
        print(f'{green_begin}{img_name}{color_end} OK')


def main():
    # 出错图片名字列表
    error_img = []

    #   路径设置
    imgs_path = f'图片文件夹路径'

    #   获取文件列表
    images = os.listdir(imgs_path)
    total = len(images)
    print(f'-- {blue_begin} the number of images : {color_end} {total}')
    print(f'-- images : {images} ')
    for img_name in images:
        #   进行文件后缀的判断
        suffix = img_name.split('.')[1]
        if suffix == 'jpg' or suffix == 'png':
            # check_img(error_img, imgs_path, img_name)
            check_img_byte(error_img, imgs_path, img_name)
    #   失败文件的输入
    print(f'--{green_begin}总文件数：{total}， 失败：{len(error_img)} {color_end}')
    if len(error_img) > 0:
        print(f'{red_begin}失败列表：{color_end}')
        for msg in error_img:
            print(msg)


if __name__ == '__main__':
    main()
