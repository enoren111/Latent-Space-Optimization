import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from collections import OrderedDict
import torch
import math


class VGG_Network(nn.Module):
    def __init__(self, hp):
        super(VGG_Network, self).__init__()
        self.backbone = backbone_.vgg16(pretrained=True).features
        self.pool_method = nn.AdaptiveMaxPool2d(1)

    def forward(self, input, bb_box=None):
        x = self.backbone(input)
        x = self.pool_method(x).view(-1, 512)
        return F.normalize(x)


class InceptionV3_Network(nn.Module):
    def __init__(self, hp):
        super(InceptionV3_Network, self).__init__()
        backbone = backbone_.inception_v3(pretrained=True)

        ## Extract Inception Layers ##
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e

        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c
        self.pool_method = nn.AdaptiveMaxPool2d(1)  # as default

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        backbone_tensor = self.Mixed_7c(x)
        feature = self.pool_method(backbone_tensor).view(-1, 2048)
        return F.normalize(feature)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        # self.se = SELayer(planes, 16)
        
        # self.bn3 = nn.Sequential(self.bn3, self.se)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


# class ResNet50_Stride_Network(nn.Module):
#     in_planes = 2048
#     num_classes = 1000
#
#     def __init__(self):
#         super(ResNet50_Stride_Network, self).__init__()
#         self.base = ResNet()
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         # self.gap = nn.AdaptiveMaxPool2d(1)
#         # self.num_classes = num_classes
#         # self.neck_feat = neck_feat
#         self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck.bias.requires_grad_(False)  # no shift
#         self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
#
#         self.bottleneck.apply(weights_init_kaiming)
#         self.classifier.apply(weights_init_classifier)
#
#     def forward(self, x):
#
#         global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
#         global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
#
#         feat = self.bottleneck(global_feat)  # normalize for angular softmax
#
#         if self.training:
#             cls_score = self.classifier(feat)
#             return cls_score, global_feat  # global feature for triplet loss
#         else:
#             return feat


class ResNet50_Stride_Network(nn.Module):
    def __init__(self, hp):
        self.block = Bottleneck
        self.layers = [3, 4, 6, 3]
        self.inplanes = 64
        super(ResNet50_Stride_Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)      
        self.layer1 = self._make_layer(self.block, 64, self.layers[0])
        self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=2)
        # self.load_param(r'resnet50-19c8e357.pth')
        self.pool_method = nn.AdaptiveMaxPool2d(1)

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'last_linear' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # add missed relu
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)
        x = self.pool_method(x)
        x = torch.flatten(x, 1)

        return F.normalize(x)


class ResNet50_Network(nn.Module):
    def __init__(self, hp):
        super(ResNet50_Network, self).__init__()
        backbone = backbone_.resnet50(pretrained=True)  # resnet50, resnet18, resnet34
        # print(backbone)
        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)
        self.pool_method = nn.AdaptiveMaxPool2d(1)

    def forward(self, input, bb_box=None):
        x = self.features(input)
        # print(x.shape)
        x = self.pool_method(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        return F.normalize(x)

class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck_SE(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEResNetBottleneck(Bottleneck_SE):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet_Network(nn.Module):

    def __init__(self,hp):
        super(SENet_Network, self).__init__()
        self.block = SEResNetBottleneck
        self.layers = [3, 4, 6, 3]
        self.inplanes = 64
        self.groups = 1
        self.reduction = 16
        self.dropout_p = None,
        self.inplanes = 64,
        self.input_3x3 = False,
        self.downsample_kernel_size = 1,
        self.downsample_padding = 0,
        self.last_stride = 2
        self.inplanes = 64

        layer0_modules = [('conv1', nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                                              padding=3, bias=False)), ('bn1', nn.BatchNorm2d(self.inplanes)),
                          ('relu1', nn.ReLU(inplace=True)), ('pool', nn.MaxPool2d(3, stride=2,
                                                                                  ceil_mode=True))]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            self.block,
            planes=64,
            blocks=self.layers[0],
            groups=self.groups,
            reduction=self.reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            self.block,
            planes=128,
            blocks=self.layers[1],
            stride=2,
            groups=self.groups,
            reduction=self.reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer3 = self._make_layer(
            self.block,
            planes=256,
            blocks=self.layers[2],
            stride=2,
            groups=self.groups,
            reduction=self.reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer4 = self._make_layer(
            self.block,
            planes=512,
            blocks=self.layers[3],
            stride=self.last_stride,
            groups=self.groups,
            reduction=self.reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = None
        self.load_param(r'se_resnet50-ce0d4300.pth')
        self.pool_method = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'last_linear' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool_method(x)
        x = torch.flatten(x, 1)

        return F.normalize(x)


# if __name__ == "__main__":
    # input = torch.rand(3, 3, 256, 128)
    # resnet1 = ResNet50_Stride_Network()
    # resnet2 = ResNet50_Network()
    # print(resnet1)
    # print(resnet2)
    # output1 = resnet1(input)
    # output2 = resnet2(input)
    # print('-----this is output1-----')
    # print(output1.shape)
    # print('-----this is output2-----')
    # print(output2.shape)

    # resnet = backbone_.resnet50(pretrained=False)
    # state_dict = torch.load(r"resnet50-19c8e357.pth")
    # resnet.load_state_dict(state_dict)
    # new_state_dict = resnet.state_dict()
    #
    # # 获取自己创建的resnet50无训练的空权重
    # net = ResNet50_Stride_Network()
    # op = net.state_dict()
    #
    # print(len(new_state_dict.keys()))  # 输出torch官方网络模型字典长度
    # print(len(op.keys()))  # 输出自己网络模型字典长度
    #
    # for i in new_state_dict.keys():   # 查看网络结构的名称 并且得出一共有320个key
    #     print(i)
    # for j in op.keys():   # 查看网络结构的名称 并且得出一共有384个key
    #     print(j)
