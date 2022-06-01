import argparse
import numpy as np
import torch
from torch.autograd import Variable
from model.deeplab import Resnet_model
from dataset.Vaihingen_dataset import VaihingenDataSet
from torch.utils import data
import os
from PIL import Image
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from utils.metricsRM import runningScore
import torch.nn.functional as F
import torch.nn.functional as F
matplotlib.use('TkAgg')
from sklearn.manifold import TSNE
'''对基准方法训练模型的预测结果，选择图片中的像素点，绘制特征TSEN分布图'''
# IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = np.array((81.19, 81.80, 120.48), dtype=np.float32)
DATA_DIRECTORY = r'D:\ISPRS\Vaihingen\\16\\512_512\\16_nooverlap'
DATA_LIST_PATH = r'D:\ISPRS\Vaihingen\\16\\512_512\\16_nooverlap\\16_1.txt'
SAVE_PATH = 'G:\wangming\试验文件夹\组内服务器\snapshots_gcn_entropy\\tsne'
# SNAPSHOT_DIR = "G:\wangming\EGA_data——工作一\model\Baseline\snapshots_resnet\P2V_70000.pth"
    # r'G:\wangming\EGA_data——工作一\model\NoAdapt\ISPRS_epoch22.pth'
SNAPSHOT_DIR=r"G:\wangming\EGA_data——工作一\model\Graph_Reasoning\baseline+EGA\P2V_72000.pth"
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'

IGNORE_LABEL = 255
NUM_CLASSES = 6
NUM_STEPS = 500  # Number of images in the validation set.
SET = 'val'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


cty_running_metrics = runningScore(NUM_CLASSES)
valid_colors = [[255, 255, 255],
                [0, 0, 255],
                [0, 255, 0],
                [255, 255, 0],
                [255, 0, 0],
                [0, 255, 255]]  # every class correspond RGB value
label_colours = dict(zip(range(6), valid_colors))  # create dictionary , which return tuple queue


def plot_embedding(data, label, title):
    # “data为n * 2
    # 矩阵，label为n * 1
    # 向量，对应着data的标签, title未使用”   6个类别
    fig = plt.figure()
    ax = plt.subplot(111)
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []
    type4_y = []
    type5_x = []
    type5_y = []
    type6_x = []
    type6_y = []
    for i in range(data.shape[0]):
        if label[i] == 0:
            type1_x.append(data[i][0])
            type1_y.append(data[i][1])
        if label[i] == 1:
            type2_x.append(data[i][0])
            type2_y.append(data[i][1])
        if label[i] == 2:
            type3_x.append(data[i][0])
            type3_y.append(data[i][1])
        if label[i] == 3:
            type4_x.append(data[i][0])
            type4_y.append(data[i][1])
        if label[i] == 4:
            type5_x.append(data[i][0])
            type5_y.append(data[i][1])
        if label[i] == 5:
            type6_x.append(data[i][0])
            type6_y.append(data[i][1])

    type1 = plt.scatter(type1_x, type1_y, s=3, c='k', marker='o' )
    type2 = plt.scatter(type2_x, type2_y, s=3, c='b', marker='o' )
    type3 = plt.scatter(type3_x, type3_y, s=5, c='g', marker='o')
    type4 = plt.scatter(type4_x, type4_y, s=5, c='y', marker='o')
    type5 = plt.scatter(type5_x, type5_y, s=5, c='r', marker='o')
    type6 = plt.scatter(type6_x, type6_y, s=5, c='c', marker='o')
    plt.legend((type1, type2, type3, type4, type5, type6),
               ('Imp. surf.', 'Build.', 'Tree', 'Car', 'Clu.', 'Low veg.', ),
               loc=(0.97, 0.5))


    # plt.xticks(np.linspace(int(x_min[0]), math.ceil(x_max[0]), 5))
    # plt.yticks(np.linspace(int(x_min[1]), math.ceil(x_max[1]), 5))
    plt.xticks()
    plt.yticks()
    # plt.title(title)

    # ax.spines['right'].set_visible(False)  # 去除右边框
    # ax.spines['top'].set_visible(False)  # 去除上边框
    return fig

def plot_2D(data, label,name):
    # “data为提取的特征数据，epoch未使用”
    #n_samples, n_features = data.shape
    print('Computing t-SNE embedding')
    # tsne = TSNE(n_components=2, learning_rate=1) #使用TSNE对特征降到二维

    tsne = TSNE(n_components=2, init='pca', perplexity=80,learning_rate=1) #使用TSNE对特征降到二维
    #t0 = time()
    result = tsne.fit_transform(data) #降维后的数据

    result_min, result_max=result.min(0), result.max(0)
    result_norm=(result-result_min)/(result_max-result_min)  # 归一化

    #画图
    fig = plot_embedding(result_norm, label,
                         't-SNE embedding of the digits (time %.2fs)')
                         #% (time() - t0))
    fig.subplots_adjust(right=0.8)  #图例过大，保存figure时无法保存完全，故对此参数进行调整
    # plt.savefig("G:\wangming\EGA_data——工作一\model\Baseline\\tsne_1\\" + str(name) + ".png")
        # "G:\wangming\EGA_data——工作一\model\\NoAdapt\\tsne_1\\" + str(name) + ".png")
    plt.savefig("G:\wangming\EGA_data——工作一\model\Graph_Reasoning\\baseline+EGA\\tsne_1\\" + str(name) + ".png")


def main():
    """Create the model and start the evaluation process."""

    print("evalute by metric of ", SNAPSHOT_DIR)
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    '''模型'''
    model = Resnet_model(num_classes=args.num_classes)
    '''加载训练的模型'''
    saved_state_dict = torch.load(SNAPSHOT_DIR)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    testloader = data.DataLoader(
        VaihingenDataSet(args.data_dir, args.data_list, crop_size=(512, 512), mean=IMG_MEAN, scale=False,
                         mirror=False, set=args.set),
        batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

    with torch.no_grad():
        for index, batch in enumerate(testloader):
            # if index % 100 == 0:
            #     print('%d processd' % index)
            image, label, _, _, name = batch
            output = model(Variable(image).cuda())

            output = interp(output).cpu().data[0].numpy()  # class,hw

            output = output.transpose(1, 2, 0)  # h,w,class  transpose
            predict = output.reshape([-1, output.shape[2]])
            print(predict.shape)

            gt = label.data.cpu().data[0].numpy()  # h,w
            pre_label = gt.reshape([-1, 1])
            print("pre_label", pre_label.shape)
            # p_label=p_label[0:5,:]

            name = name[0] + ".tif"

            p_data = np.zeros((4096, 6))
            p_label = np.zeros((4096, 1))
            for i in range(4096): # 每个64个像素点去一个像素，不然512*512的图片，像素点过多
                p_data[i, :] = predict[i * 64, :]
                p_label[i, :] = pre_label[i * 64, :]
            print(p_data)
            print(p_label)
            plot_2D(p_data, p_label, name)





if __name__ == '__main__':
    main()
