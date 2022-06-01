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
import torch.nn.functional as F

# IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = np.array((81.19, 81.80, 120.48), dtype=np.float32)
DATA_DIRECTORY = r'D:\ISPRS\Vaihingen\\16\\512_512\\16_nooverlap'
DATA_LIST_PATH = r'D:\ISPRS\Vaihingen\\16\\512_512\\16_nooverlap\\16_nooverlap.txt'
SAVE_PATH = r'./result'
SNAPSHOT_DIR = r'./snapshots_gcn2_entropy_PM_0.004/P2V_gcn94000.pth'
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
IGNORE_LABEL = 255
NUM_CLASSES = 6
NUM_STEPS = 500  # Number of images in the validation set.
SET = 'val'

'''根据评测指标的模型，保存所有分割结果'''

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

def decode_color(img):
    temp = img[:, :]  # h,w
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, NUM_CLASSES):
        r[temp == l] = label_colours[l][0]  # get each channel value
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))  # h,w,3
    rgb[:, :, 0] = r/255
    rgb[:, :, 1] = g/255
    rgb[:, :, 2] = b/255
    return rgb
    
def prob_2_entropy1(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    # h, w = prob.size()
    out = -torch.mul(prob, torch.log2(prob + 1e-30))  # n,c,h,w
    out = out.mean(1)
    # out = out.squeeze(0)
    return out

def main():
    """Create the model and start the evaluation process."""

    print("evalute by metric of ", SNAPSHOT_DIR)
    args = get_arguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    model = GCN_Resnet(num_classes=args.num_classes)

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
            name = name[0]
            output = model(Variable(image).cuda())

            """define entropy heatmap"""
            pred1 = F.softmax(interp(output), dim=1)  # n,c,h,w
            entropy_map1 = prob_2_entropy1(pred1)
            heatmap_batch = entropy_map1.cpu().data.numpy()
            # print(entropy_map1.shape,entropy_map1)
            heatmap_tmp = heatmap_batch[0, :, :]  / (np.max(heatmap_batch[0, :, :]))
            fig = plt.figure()
            plt.axis('off')
            heatmap = plt.imshow(heatmap_tmp, cmap='viridis')
            fig.colorbar(heatmap)
            # fig.savefig('%s/%s_heatmap.png' % (SAVE_PATH, name))
            # torch_vis_color(name[0], entropy_map1, "entropy", 6, 6, SAVE_PATH, colormode=2, margining=1)


            # output = interp(output).cpu().data[0].numpy()  # class,hw
            #
            # output = output.transpose(1, 2, 0)  # h,w,class
            # output = np.argmax(output, axis=2)  # h,w
            output= interp(output).data[0].max(0)[1].cpu().numpy()
            gt = label.data.cpu().data[0].numpy()  # h,w

            """define compare"""
            compare = (output == gt)
            compare = compare.astype(int)

            # heatmap_tmp = heatmap_batch[0, :, :] / np.max(heatmap_batch[0, :, :])
            fig = plt.figure()
            plt.axis('off')
            heatmap = plt.imshow(compare, cmap='gray')
            fig.colorbar(heatmap)
            # fig.savefig('%s/%s_compare.png' % (SAVE_PATH, name))


            cty_running_metrics.update(gt, output)  # pingce
            #image
            image_c = Image.open(os.path.join(DATA_DIRECTORY,"image/"+name+".tif"))
            image_c=np.asarray(image_c)
            image_c=image_c/255
            #GT
            gt_c=decode_color(gt)
            gt_c=np.asarray(gt_c)
            #predict
            output_c = decode_color(output)
            output_c = np.asarray(output_c)
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(image_c)
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(gt_c)
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(output_c)
            plt.axis('off')
            
            plt.savefig(os.path.join(SAVE_PATH, name))
            plt.close()
            plt.figure()
            plt.plot()
            plt.imshow(output_c)
            plt.axis('off')
            plt.savefig(os.path.join(SAVE_PATH, name + "_predit.tif"))
            plt.close()

        cty_score, cty_class_iou = cty_running_metrics.get_all()
        cty_running_metrics.reset()

        f_result = open(os.path.join(SAVE_PATH, '_value.txt'), 'a')
        f_result.write('<======epoch: {}======>\n'.format(SNAPSHOT_DIR))
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


if __name__ == '__main__':
    main()
