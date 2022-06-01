import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import math

from torch.autograd import Variable
from torch.nn import Parameter
from model.resnet import Res_Deeplab
from model.resnet import Classifier_Module
'''deeplab 网络结构'''
class Resnet_model(nn.Module):
    def __init__(self,num_classes):
        super(Resnet_model, self).__init__()
        self.n_classes = num_classes

        Seg_Model = Res_Deeplab(num_classes=self.n_classes)

        self.layer0 = nn.Sequential(Seg_Model.conv1, Seg_Model.bn1, Seg_Model.relu, Seg_Model.maxpool)
        self.layer1 = Seg_Model.layer1  # 3
        self.layer2 = Seg_Model.layer2  # 4
        self.layer3 = Seg_Model.layer3  # 23
        self.layer4 = Seg_Model.layer4  # 3
        self.classifier=Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], self.n_classes)


    def forward(self, x):
        inp_shape = x.shape[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_map = self.layer4(x)
        #print(x.shape)  # feature_map

        out = self.classifier(feature_map)

        return out

    def get_1x_lr_params_NOscale(self):
        b = []

        b.append(self.layer0)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        b = []
        b.append(self.classifier.parameters())
        # b.append(self.final2.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': 1 * learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]


class FCDiscriminator_model(nn.Module):

    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator_model, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x





if __name__ == '__main__':
    model = Encoder()
    x = torch.randn(1, 3, 400, 400)
    print("input:", x.shape)
    _, _, _, y = model(x)
    print(model)
    # pred = pred.data.max(1)[1].cpu().numpy()


