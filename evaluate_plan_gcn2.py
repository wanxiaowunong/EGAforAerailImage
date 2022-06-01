import argparse
import numpy as np
import torch
from torch.autograd import Variable
from model.GCN_AL_model2 import GCN_Resnet
from dataset.Vaihingen_dataset import VaihingenDataSet
from torch.utils import data
import os
from PIL import Image
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from utils.metricsRM import runningScore

'''添加GCN网络的模型测试代码'''
#IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = np.array((81.19, 81.80, 120.48), dtype=np.float32)
source="Potsdam"

if source=="Potsdam":
   DATA_DIRECTORY = r'D:\ISPRS\Vaihingen\\16\\512_512\\16_nooverlap'
   DATA_LIST_PATH = r'D:\ISPRS\Vaihingen\\16\\512_512\\16_nooverlap\\16_nooverlap.txt'
   SNAPSHOT_DIR='./snapshots_gcn2_entropy_PM_0.002'


else:
   DATA_DIRECTORY = '/Share/home/E19201070/data/Potsdam_512512_nooverlap'
   DATA_LIST_PATH = '/Share/home/E19201070/data/Potsdam_512512_nooverlap/24_nooverlap.txt'
   SNAPSHOT_DIR='./snapshots_gcn_V2P'

# if source=="Potsdam":
#    DATA_DIRECTORY = '/Share/home/E19201070/data/ISPRS512_512/Vaihingen_nooverlap'
#    DATA_LIST_PATH = '/Share/home/E19201070/data/ISPRS512_512/Vaihingen_nooverlap/16_nooverlap.txt'
#    SNAPSHOT_DIR='./snapshots_gcn2_entropy_advmin_PM_0.6_5'
#
#
# else:
#    DATA_DIRECTORY = '/Share/home/E19201070/data/Potsdam_512512_nooverlap'
#    DATA_LIST_PATH = '/Share/home/E19201070/data/Potsdam_512512_nooverlap/24_nooverlap.txt'
#    SNAPSHOT_DIR='./snapshots_gcn_V2P'
SAVE_PATH = './result/Vaihingen'


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


def main():
    """Create the model and start the evaluation process."""

    print("evalute by metric of ", SNAPSHOT_DIR)
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    iter = []
    Impre = []
    Build = []
    Tree = []
    Car = []
    Clutter = []
    LowVge = []
    mIoU = []


    for i in range(1, 51):

        if source=="Potsdam":
            model_path =os.path.join(SNAPSHOT_DIR, 'P2V_gcn{0:d}.pth'.format(i * 2000))
        else:
            model_path =os.path.join(SNAPSHOT_DIR, 'V2P_gcn{0:d}.pth'.format(i * 2000))

        iter.append(i * 2000)


        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        model = GCN_Resnet(num_classes=args.num_classes)

        saved_state_dict = torch.load(model_path)
        model.load_state_dict(saved_state_dict)

        model.eval()
        model.cuda()

        testloader = data.DataLoader(
            VaihingenDataSet(args.data_dir, args.data_list, crop_size=(512, 512), mean=IMG_MEAN, scale=False,
                             mirror=False, set=args.set),
            batch_size=1, shuffle=False, pin_memory=True)

        interp = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

        print("<====epoch:",i*2000,"====>")
        with torch.no_grad():
            for index, batch in enumerate(testloader):
                # if index % 100 == 0:
                #     print('%d processd' % index)
                image,label, _, _, name = batch
                output = model(Variable(image).cuda())  #n,c,h,w

                output = interp(output).data.max(1)[1].cpu().numpy()  #n,hw
                gt = label.data.cpu().numpy() #n,h,w

                cty_running_metrics.update(gt, output)


                # name = name[0].split('/')[-1]

            cty_score, cty_class_iou = cty_running_metrics.get_all()
            cty_running_metrics.reset()

            miou=0
            for j in range(len(cty_class_iou)):
                miou=miou+cty_class_iou[j]
            mIoU.append(miou/NUM_CLASSES)
            Impre.append(cty_class_iou[0])
            Build.append(cty_class_iou[1])
            Tree.append(cty_class_iou[2])
            Car.append(cty_class_iou[3])
            Clutter.append(cty_class_iou[4])
            LowVge.append(cty_class_iou[5])
            f_result = open(os.path.join(SNAPSHOT_DIR, '_value.txt'), 'a')
            f_result.write('<======epoch: {}======>\n'.format(i * 2000))
            f_result.write("Imprevious:{}\n".format(cty_class_iou[0]))
            f_result.write('Buildings:{}\n'.format(cty_class_iou[1]))
            f_result.write('Tree     :{}\n'.format(cty_class_iou[2]))
            f_result.write('Car      :{}\n'.format(cty_class_iou[3]))
            f_result.write('Clutter  :{}\n'.format(cty_class_iou[4]))
            f_result.write('LowVegetable:{}\n'.format(cty_class_iou[5]))
            for k, v in cty_score.items():
                f_result.write("{}{}\n".format(k, v))
                print(k, v)
            f_result.close()

        matplotlib.use('Agg')
        plt.title('Potsdam-to-Vaihingen mIoU')
        plt.plot(iter, Impre, 'm-o', label='Impre')
        plt.plot(iter, Build, 'b-o', label='Build')
        plt.plot(iter, Tree, 'g-o', label='Tree')
        plt.plot(iter, Car, 'y-o', label='Car')
        plt.plot(iter, Clutter, 'r-o', label='Clutter')
        plt.plot(iter, LowVge, 'c-o', label='LowVge')
        plt.plot(iter, mIoU, 'k--*', label='mIoU', linewidth=4)
        # plt.legend(loc='upper left')
        plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)
        plt.ylabel('mIoU', size=14)
        plt.xlabel('iter', size=14)
        plt.grid()
        # plt.show()
        plt.savefig(os.path.join(SNAPSHOT_DIR, 'IoU.png'), format='png')
        plt.close()
        if source=="Potsdam":
            plt.title('P2V mIoU')
        else:
            plt.title('V2P mIoU')
        plt.plot(iter, mIoU, 'k--*', label='mIoU', linewidth=2)
        for a, b in zip(iter, mIoU):
            plt.text(a, b, b, ha='center', va='bottom', fontsize=4)
        plt.legend(bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0)
        plt.ylabel('mIoU', size=14)
        plt.xlabel('iter', size=14)
        plt.grid()
        plt.savefig(os.path.join(SNAPSHOT_DIR, 'mIoU.png'), format='png')
        plt.close()


if __name__ == '__main__':
    main()
