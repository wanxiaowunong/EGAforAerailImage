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
from utils.util import gen_adj, gen_Node

'''利用图卷积网络的消息传播机制聚合上下文特征，增强特征表示能力,sum'''

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features  # input dimension
        self.out_features = out_features  # output dimension
        self.weight = Parameter(torch.Tensor(in_features, out_features))  # define weight metrix
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  # self-defineation intilization

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # XW(N,D'); X(N,D);W(D,D')
        output = torch.mm(adj, support)  # P(N,D')
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nin, nhid, nout, dropout):  #64,512,1024
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nin, nhid)  # 256,512
        self.gc2 = GraphConvolution(nhid, nout)  # 512,1024
        self.dropout = dropout

        self.fc_2 = nn.Conv2d(1024, 2048, kernel_size=1, padding=0, stride=(1, 1),
                              groups=1, bias=False)
        self.blocker = nn.BatchNorm2d(2048)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  #the first layer
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)  #512   #the second layer
        gcn_out = x.transpose(1, 0)  # c* hw
        gcn_out = gcn_out.reshape([gcn_out.shape[0], int(gcn_out.shape[1] ** 0.5), int(gcn_out.shape[1] ** 0.5)])
        gcn_out = gcn_out.unsqueeze(0)
        x=self.blocker(self.fc_2(gcn_out))
        return x


class GCN_Resnet(nn.Module):
    def __init__(self, num_classes):
        super(GCN_Resnet, self).__init__()
        self.n_classes = num_classes

        Seg_Model = Res_Deeplab(num_classes=self.n_classes)

        self.layer0 = nn.Sequential(Seg_Model.conv1, Seg_Model.bn1, Seg_Model.relu, Seg_Model.maxpool)
        self.layer1 = Seg_Model.layer1  # 3
        self.layer2 = Seg_Model.layer2  # 4
        self.layer3 = Seg_Model.layer3  # 23
        self.layer4 = Seg_Model.layer4  # 3

        self.conv51 = nn.Sequential(nn.Conv2d(2048, 512, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(512, 64, 1, padding=0, bias=False),
                                    nn.ReLU())   #reduce dimension 512-->64
        self.gcn = GCN(64, 512, 1024, 0.5)
        self.conv53 = nn.Sequential(nn.Conv2d(2048, 2048, 1, padding=0, bias=False),  # 512,512
                                    nn.BatchNorm2d(2048),
                                    nn.ReLU())
        self.classifier = Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], self.n_classes)

    def forward(self, x):
        inp_shape = x.shape[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_map = self.layer4(x)   # extract feature map
        # print(x.shape)  # feature_map
        feature_map1=self.conv51(feature_map)  # n*1024*h*w  # reduce dimension

        gcn_in=self.conv52(feature_map1) #n*512*h*w   # reduce dimension
        '''constract the graph structure'''
        Node = gen_Node(gcn_in)
        adj = gen_adj(gcn_in)
        adj=adj.detach()

        gcn_out = self.gcn(Node, adj)  # hw *c   256,512,1024
        # gcn_out = gcn_out.transpose(1, 0)  # c* hw
        # gcn_out = gcn_out.reshape([gcn_out.shape[0], int(gcn_out.shape[1] ** 0.5), int(gcn_out.shape[1] ** 0.5)])
        # gcn_out = gcn_out.unsqueeze(0)
        #x = torch.cat([feature_map, gcn_out], dim=1)
        # print(gcn_out.shape)
        '''sum to fusion feature'''
        concat_f = feature_map + gcn_out  #1024
        out=self.conv53(concat_f)

        classifi = self.classifier(out)

        return classifi   #classifi,feature_map,out

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
        b.append(self.conv51.parameters())
        b.append(self.conv52.parameters())
        b.append(self.gcn.parameters())
        b.append(self.conv53.parameters())
        b.append(self.classifier.parameters())
        # b.append(self.final2.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': 1 * learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]


class FCDiscriminator_model(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator_model, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)

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


