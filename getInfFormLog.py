# 从CLTR模型的train log中获取test epoch的val数据集mae、mse
import re
import matplotlib.pyplot as plt


def save_change_to_img(img_save_path, xlabel, ylabels, ylabel_name):
    epoch = xlabel['value']
    epoch_name = xlabel['label']
    shapes = ['ro--', 'rs--', 'go--', 'gs--', 'bo--', 'bs--']
    plt.figure(figsize=(20,10))
    for i in range(len(ylabels)):
        y = ylabels[i]
        shape = shapes[i]
        plt.plot(epoch, y['value'], shape, alpha=0.5, linewidth=1, label=y['name'])

    plt.legend()
    plt.xlabel(epoch_name)
    plt.ylabel(ylabel_name)
    plt.savefig(img_save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


def save_to_txt(save_path, content_dire):
    with open(save_path, 'w') as f:
        for i in range(len(content_dire['epoch'])):
            f.write(f"epoch:{content_dire['epoch'][i]} MAE:{content_dire['mae'][i]} MSE:{content_dire['mse'][i]}\n")
    print(f'save to {save_path}')


def get_epoch_mae_mse_from_txt(label_from):
    contents = []
    with open(label_from, 'r') as f:
        contents = f.readlines()
        print(contents)
    epoch = [int(re.findall(r"epoch:(.*) MAE", content)[0]) for content in contents]
    mae = [float(re.findall(r"MAE:(.*) MSE", content)[0]) for content in contents]
    mse = [float(re.findall(r"MSE:(.*)\n", content)[0]) for content in contents]
    return {'epoch': epoch, 'mae': mae, 'mse': mse}


def get_epoch_mae_mse_from_log(label_from):
    contents = []
    with open(label_from, 'r') as f:
        contents = f.readlines()
    new_contents = [content for content in contents if 'Testing Epoch' in content]

    epoch = [int(re.findall(r"Testing Epoch:\[(.*)/1500\]", content)[0]) for content in new_contents]
    mae = [float(re.findall(r"\t mae=(.*)\t mse=", content)[0]) for content in new_contents]
    mse = [float(re.findall(r"mse=(.*)\t best_mae", content)[0]) for content in new_contents]

    return {'epoch': epoch, 'mae': mae, 'mse': mse}


def get_epoch_loss_from_log(label_from):
    contents = []
    with open(label_from, 'r') as f:
        contents = f.readlines()
    new_contents = [content for content in contents if 'Training Epoch' in content]

    epoch = [int(re.findall(r"Training Epoch:\[(.*)/1500\]", content)[0]) for content in new_contents]
    loss = [float(re.findall(r"\t loss=(.*)\t lr=", content)[0]) for content in new_contents]

    return {'epoch': epoch, 'loss': loss}


if __name__ == '__main__':
    # jhu
    name = 'jhu_val_20230601_all'
    root = '../CLTR/save_file/jhu_20230601'
    # nwpu
    # name = 'nwpu_val_20230307_all'
    # root = '../CLTR/save_file/nwpu_20230307'
    from_file = f'{root}/1.log'
    to_file = f'{root}/{name}.txt'
    from_file2 = f'{root}/{name}.txt'.replace('val', 'test')
    img_path = f'{root}/{name}.jpg'
    loss_img_path = f'{root}/{name.replace("all", "loss")}.jpg'
    val_contents = get_epoch_mae_mse_from_log(from_file)
    print(val_contents)
    # save_to_txt(to_file, val_contents)
    # test_contents = get_epoch_mae_mse_from_txt(from_file2)
    # print(test_contents)
    # 将变化曲线保存下来
    ylabels = [{'label': 'mae', 'name': 'val mae', 'value': val_contents['mae']},
               {'label': 'mse', 'name': 'val mse', 'value': val_contents['mse']}]
               # {'label': 'mae', 'name': 'test mae', 'value': test_contents['mae']},
               # {'label': 'mse', 'name': 'test mse', 'value': test_contents['mse']}]
    save_change_to_img(img_path, xlabel={'label': 'epoch', 'value': val_contents['epoch']}, ylabels=ylabels, ylabel_name='mae/mse')
    # 保存loss文件
    loss_contents = get_epoch_loss_from_log(from_file)
    loss_ylabels = [{'label': 'loss', 'name': 'loss', 'value': loss_contents['loss']}]
    save_change_to_img(loss_img_path, xlabel={'label': 'epoch', 'value': loss_contents['epoch']}, ylabels=loss_ylabels, ylabel_name='loss')


