import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp

import scipy.sparse as sp
from scipy import sparse

from model.deeplab import Resnet_model, FCDiscriminator_model
from model.domain_classifier import DomainClassifier
from utils.loss import CrossEntropy2d
from utils.loss import WeightedBCEWithLogitsLoss

from dataset.Potsdam_dataset import PotsdamDataSet
from dataset.Vaihingen_dataset import VaihingenDataSet
from utils.metricsRM import runningScore
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
'''基准方法上添加EGA，考虑局部特征对齐来缩小全局特征误对齐的影响'''
# IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

IMG_MEAN_P = np.array((85.92, 92.55, 86.55), dtype=np.float32)
IMG_MEAN_V = np.array((81.19, 81.80, 120.48), dtype=np.float32)
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4

IGNORE_LABEL = 255

MOMENTUM = 0.9
NUM_CLASSES = 6
cty_running_metrics = runningScore(NUM_CLASSES)

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './snapshots_resnet_entropy_plot_d_all/'

# Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 2.5e-4  # 0.00025
LEARNING_RATE_D = 1e-4
LEARNING_RATE_Dcl = 1e-4
NUM_STEPS = 100000
NUM_STEPS_STOP = 100000  # Use damping instead of early stopping
PREHEAT_STEPS = int(NUM_STEPS_STOP / 20)
POWER = 0.9
RANDOM_SEED = 1234

SOURCE = 'Potsdam'
TARGET = 'Vaihingen'
SET = 'train'

if SOURCE == 'Potsdam':
    INPUT_SIZE_SOURCE = '512,512'
    DATA_DIRECTORY = '/Share/home/E19201070/data/ISPRS512_512/Potsdam'  # Potsdam_512512_nooverlap
    DATA_LIST_PATH = '/Share/home/E19201070/data/ISPRS512_512/Potsdam/24.txt'  # /Potsdam_512512_nooverlap/24_nooverlap.txt
    Lambda_weight = 0.01
    Lambda_adv = 0.001
    Lambda_local = 5
    Epsilon = 0.6
elif SOURCE == 'SYNTHIA':
    INPUT_SIZE_SOURCE = '1280,760'
    DATA_DIRECTORY = '/Share/home/E19201070/data/ISPRS512_512/Vaihingen'
    DATA_LIST_PATH = '/Share/home/E19201070/data/ISPRS512_512/Vaihingen/16.txt'
    Lambda_weight = 0.01
    Lambda_adv = 0.001
    Lambda_local = 10
    Epsilon = 0.4

INPUT_SIZE_TARGET = '512,512'
DATA_DIRECTORY_TARGET = '/Share/home/E19201070/data/ISPRS512_512/Vaihingen'
DATA_LIST_PATH_TARGET = '/Share/home/E19201070/data/ISPRS512_512/Vaihingen/17.txt'

DATA_DIRECTORY_val = '/Share/home/E19201070/data/ISPRS512_512/Vaihingen'
DATA_LIST_PATH_val = '/Share/home/E19201070/data/ISPRS512_512/Vaihingen/16.txt'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    # parser.add_argument("--model", type=str, default=MODEL,
    #                     help="available options : ResNet")
    parser.add_argument("--source", type=str, default=SOURCE,
                        help="available options : Potsdam, Vaihingen")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : Vaihingen")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size-source", type=str, default=INPUT_SIZE_SOURCE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    # parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
    #                     help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=7,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label):  # crossentropy loss
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d(NUM_CLASSES).cuda()  # to(device)
    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate_D, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


'''calculate information entropy map'''
def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    # h, w = prob.size()
    out = -torch.mul(prob, torch.log2(prob + 1e-30))  # n,c,h,w
    out = out.mean(1)
    out = out.squeeze(0)
    return out


def main():
    print("train_resnet ", SNAPSHOT_DIR)
    """Create the model and start the training."""
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    h, w = map(int, args.input_size_source.split(','))
    input_size_source = (h, w)

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)

    cudnn.enabled = True

    # Create Network
    '''generator'''
    model = Resnet_model(num_classes=args.num_classes)

    model.train()
    model.cuda()

    '''discriminator'''
    cudnn.benchmark = True
    model_dcl=DomainClassifier()
    model_dcl.train()
    model_dcl.cuda()

    # Init D
    model_D = FCDiscriminator_model(num_classes=args.num_classes)
    # =============================================================================
    #    #for retrain
    #    saved_state_dict_D = torch.load(RESTORE_FROM_D)
    #    model_D.load_state_dict(saved_state_dict_D)
    # =============================================================================

    model_D.train()
    model_D.cuda()

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    if args.source == 'Potsdam':
        trainloader = data.DataLoader(
            PotsdamDataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                           crop_size=input_size_source,
                           scale=True, mirror=True, mean=IMG_MEAN_P),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = data.DataLoader(
            VaihingenDataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                             crop_size=input_size_source,
                             scale=True, mirror=True, mean=IMG_MEAN_V),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(VaihingenDataSet(args.data_dir_target, args.data_list_target,
                                                    max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                    crop_size=input_size_target,
                                                    scale=True, mirror=True, mean=IMG_MEAN_V,
                                                    set=args.set),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)

    targetloader_iter = enumerate(targetloader)

    optimizer = optim.SGD(model.optim_parameters(args.learning_rate),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_dcl = optim.Adam(model_dcl.parameters(), lr=LEARNING_RATE_Dcl, betas=(0.9, 0.99))
    optimizer_dcl.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    weighted_bce_loss = WeightedBCEWithLogitsLoss()

    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # Labels for Adversarial Training
    source_label = 0
    target_label = 1

    loss_seg_tmp = []
    loss_adv_tmp = []
    loss_D_s_tmp = []
    loss_D_t_tmp = []
    loss_D_tmp = []
    loss_dcl_tmp = []
    i_iter_tmp = []
    epoch = 0
    epoch_tmp = []
    loss_seg_tmp1 = []
    loss_adv_tmp1 = []
    loss_D_s_tmp1 = []
    loss_D_t_tmp1 = []
    loss_D_t_tmp_together = []
    loss_tmp = []
    i_iter_tmp1 = []

    loss_seg_tmp0 = 0
    loss_adv_tmp0 = 0
    loss_tmp0 = 0
    loss_D_s_tmp0 = 0
    loss_D_t_tmp0 = 0
    loss_dcl_tmp0=0

    for i_iter in range(args.num_steps):

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)

        optimizer_dcl.zero_grad()
        adjust_learning_rate_D(optimizer_dcl,i_iter)
        damping = 1 - i_iter / NUM_STEPS

        # ======================================================================================
        # train G
        # ======================================================================================

        # Remove Grads in D
        for param in model_D.parameters():
            param.requires_grad = False
        for param in model_dcl.parameters():
            param.requires_grad = False


        # Train with Source
        _, batch = next(trainloader_iter)
        images_s, labels_s, _, _, _ = batch
        images_s = Variable(images_s).cuda()  # to(device)
        pred_source = model(images_s)

        pred_source = interp_source(pred_source)

        # Segmentation Loss
        loss_seg = loss_calc(pred_source, labels_s)
        loss_seg.backward()

        # Train with Target
        _, batch = next(targetloader_iter)
        images_t, labels_t, _, _, _ = batch
        images_t = Variable(images_t).cuda()  # to(device)
        pred_target = model(images_t)

        pred_target = interp_target(pred_target)  # n,c,h,w

        pred = F.softmax(pred_target, dim=1)  # n,c,h,w
        # pred = pred.cpu().data[0].numpy()  # c,h,w
        # pred = pred.transpose(1, 2, 0)  # h,w,c
        # prob = np.max(pred, axis=2)  # h,w
        '''calculate information map'''
        entropy_map = prob_2_entropy(pred)

        D_out = interp_target(model_D(F.softmax(pred_target, dim=1)))

        if (i_iter > PREHEAT_STEPS):
            loss_adv = weighted_bce_loss(D_out,
                                         Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(),
                                         entropy_map, Epsilon, Lambda_local)
        else:
            loss_adv = bce_loss(D_out,
                                Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda())
        loss_adv = loss_adv * Lambda_adv * damping
        loss_adv.backward()

        # ======================================================================================
        # train D
        # ======================================================================================

        # Bring back Grads in D
        for param in model_dcl.parameters():
            param.requires_grad = True
        for param in model_D.parameters():
            param.requires_grad = True

        prob_dcl_real=model_dcl(F.softmax(pred_source.detach(), dim=1))
        prob_dcl_fake = model_dcl(F.softmax(pred_target.detach(), dim=1))

        loss_dcl=bce_loss(prob_dcl_real, Variable(torch.FloatTensor(prob_dcl_real.data.size()).fill_(source_label)).cuda()).cuda() \
                 + bce_loss(prob_dcl_fake, Variable(torch.FloatTensor(prob_dcl_fake.data.size()).fill_(target_label)).cuda()).cuda()
        loss_dcl.backward()

        # Train with Source
        pred_source = pred_source.detach()

        D_out_s = interp_source(model_D(F.softmax(pred_source, dim=1)))

        loss_D_s = bce_loss(D_out_s,
                            Variable(torch.FloatTensor(D_out_s.data.size()).fill_(source_label)).cuda())

        loss_D_s.backward()

        # Train with Target
        pred_target = pred_target.detach()
        entropy_map = entropy_map.detach()

        D_out_t = interp_target(model_D(F.softmax(pred_target, dim=1)))

        '''Adaptive Adversarial Loss'''
        if (i_iter > PREHEAT_STEPS):
            loss_D_t = weighted_bce_loss(D_out_t,
                                         Variable(
                                             torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(),
                                         entropy_map, Epsilon, Lambda_local)
        else:
            loss_D_t = bce_loss(D_out_t,
                                Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda())

        loss_D_t.backward()

        optimizer.step()
        optimizer_dcl.step()
        optimizer_D.step()

        print('exp = {}'.format(args.snapshot_dir))
        print(
            'iter = {}/{}, loss_seg = {} loss_adv = {},  loss_D_s = {} loss_D_t = {}'.format(
                i_iter, args.num_steps, loss_seg, loss_adv, loss_D_s, loss_D_t))

        f_loss = open(osp.join(args.snapshot_dir, 'loss.txt'), 'a')
        f_loss.write('iter = {}/{}, loss_seg = {} loss_adv = {},  loss_D_s = {} loss_D_t = {}\n'.format(
            i_iter, args.num_steps, loss_seg, loss_adv, loss_D_s, loss_D_t))
        f_loss.close()

        loss_seg_tmp0 += loss_seg.item()
        loss_adv_tmp0 += loss_adv.item()
        loss_D_s_tmp0 += loss_D_s.item()
        loss_D_t_tmp0 += loss_D_t.item()
        loss_dcl_tmp0 += loss_dcl.item()
        loss_tmp0 = loss_tmp0 + loss_seg.item() + loss_adv.item()

        if (i_iter + 1) % 936 == 0:
            epoch = epoch + 1
            epoch_tmp.append(epoch)
            plt.title('adv_loss')
            loss_adv_tmp.append(loss_adv_tmp0 / 936)
            loss_adv_tmp0 = 0
            plt.plot(epoch_tmp, loss_adv_tmp, label='loss_adv')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
            plt.grid()
            plt.savefig(os.path.join(args.snapshot_dir, 'adv_loss.png'))
            plt.close()

            f_loss = open(osp.join(args.snapshot_dir, 'loss_d.txt'), 'a')
            f_loss.write(str((loss_D_s_tmp0 + loss_D_t_tmp0) / (2 * 936)) + '\n')
            f_loss.close()

            f_loss = open(osp.join(args.snapshot_dir, 'loss_all.txt'), 'a')
            f_loss.write(str(loss_tmp0 / 936) + '\n')
            f_loss.close()

            f_loss = open(osp.join(args.snapshot_dir, 'loss_dcl.txt'), 'a')
            f_loss.write(str(loss_dcl_tmp0 / 936) + '\n')
            f_loss.close()

            plt.title('')
            loss_tmp.append(loss_dcl_tmp0/ 936)
            loss_dcl_tmp0 = 0
            plt.plot(epoch_tmp, loss_tmp, label='loss_dcl')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
            plt.grid()
            plt.savefig(os.path.join(args.snapshot_dir, 'loss_dcl.png'))
            plt.close()

            plt.title('')
            loss_tmp.append(loss_tmp0 / 936)
            loss_tmp0 = 0
            plt.plot(epoch_tmp, loss_tmp, label='loss_EGA')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
            plt.grid()
            plt.savefig(os.path.join(args.snapshot_dir, 'loss.png'))
            plt.close()

            plt.title('D_loss')
            loss_D_tmp.append((loss_D_s_tmp0 + loss_D_t_tmp0) / (2 * 936))
            plt.plot(epoch_tmp, loss_D_tmp, label='loss_D')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
            plt.grid()
            plt.savefig(os.path.join(args.snapshot_dir, 'D_loss.png'))
            plt.close()

            plt.title('D_loss')
            loss_D_s_tmp.append(loss_D_s_tmp0 / 936)
            loss_D_s_tmp0 = 0
            loss_D_t_tmp.append(loss_D_t_tmp0 / 936)
            loss_D_t_tmp0 = 0
            plt.plot(epoch_tmp, loss_D_s_tmp, label='loss_D_s')
            plt.plot(epoch_tmp, loss_D_t_tmp, label='loss_D_t')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
            plt.grid()
            plt.savefig(os.path.join(args.snapshot_dir, 'D_loss1.png'))
            plt.close()

            plt.title('segmentation_loss')
            loss_seg_tmp.append(loss_seg_tmp0 / 936)
            loss_seg_tmp0 = 0
            plt.plot(epoch_tmp, loss_seg_tmp, label='loss_seg')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
            plt.grid()
            plt.savefig(os.path.join(args.snapshot_dir, 'seg_loss.png'))
            plt.close()

        if i_iter > 0:
            i_iter_tmp1.append(i_iter)
            plt.title('adv_loss_all')
            loss_adv_tmp1.append(loss_adv.item())
            plt.plot(i_iter_tmp1, loss_adv_tmp1, label='loss_adv')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
            plt.grid()
            plt.savefig(os.path.join(args.snapshot_dir, 'adv_loss_all.png'))
            plt.close()

            plt.title('D_loss_all')
            loss_D_s_tmp1.append(loss_D_s.item())
            loss_D_t_tmp1.append(loss_D_t.item())
            plt.plot(i_iter_tmp1, loss_D_s_tmp1, label='loss_D_s')
            plt.plot(i_iter_tmp1, loss_D_t_tmp1, label='loss_D_t')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
            plt.grid()
            plt.savefig(os.path.join(args.snapshot_dir, 'D_loss_all.png'))
            plt.close()

            plt.title('D_loss_togethertogether')
            loss_D_t_tmp_together.append((loss_D_s.item() + loss_D_t.item()) / 2)
            plt.plot(i_iter_tmp1, loss_D_t_tmp_together, label='loss_D')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
            plt.grid()
            plt.savefig(os.path.join(args.snapshot_dir, 'D_loss_together.png'))
            plt.close()

            plt.title('segmentation_loss_all')
            loss_seg_tmp1.append(loss_seg.item())
            plt.plot(i_iter_tmp1, loss_seg_tmp1, label='loss_seg')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
            plt.grid()
            plt.savefig(os.path.join(args.snapshot_dir, 'seg_loss_all.png'))
            plt.close()

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'P2V_' + str(args.num_steps) + '.pth'))
            # torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'P2V_' + str(args.num_steps) + '_D.pth'))
            break

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model_dcl.state_dict(), osp.join(args.snapshot_dir, 'dcl_' + str(args.num_steps) + '.pth'))
            # torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'P2V_' + str(args.num_steps) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'P2V_' + str(i_iter) + '.pth'))
            # torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'P2V_' + str(i_iter) + '_D.pth'))


if __name__ == '__main__':
    main()
