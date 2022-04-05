"""
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
"""
from __future__ import print_function,  division,  absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['XceptionCertainty', 'Xception']

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block,  self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class XceptionCertainty(nn.Module):
    def __init__(self,  num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(XceptionCertainty,  self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.entry_flow = nn.Sequential(
            Block(32, 64, 2, 2, start_with_relu=False, grow_first=True), 
            Block(64, 128, 2, 2, start_with_relu=True, grow_first=True), 
            Block(128, 256, 2, 2, start_with_relu=True, grow_first=True),
        )

        self.middle_flow = nn.Sequential(
            Block(256, 256, 3, 1, start_with_relu=True, grow_first=True),
        )

        self.exit_flow = nn.Sequential(
            Block(256, 512, 2, 2, start_with_relu=True, grow_first=False),
            SeparableConv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.fc = nn.Linear(512, num_classes)
        self.confidence = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        pred = self.fc(x)
        confidence = self.confidence(x)
        return pred, confidence


class Xception(nn.Module):
    def __init__(self,  num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception,  self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.entry_flow = nn.Sequential(
            Block(32, 64, 2, 2, start_with_relu=False, grow_first=True), 
            Block(64, 128, 2, 2, start_with_relu=True, grow_first=True), 
            Block(128, 256, 2, 2, start_with_relu=True, grow_first=True),
        )

        self.middle_flow = nn.Sequential(
            Block(256, 256, 3, 1, start_with_relu=True, grow_first=True)
        )

        self.exit_flow = nn.Sequential(
            Block(256, 512, 2, 2, start_with_relu=True, grow_first=False),
            SeparableConv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=0.5, inplace=True)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        pred = self.fc(x)
        return pred


if __name__ == '__main__':
    from torchsummary import summary
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = Xception(num_classes=2)
    x = torch.randn(2, 3, 224, 224)
    pred, confidence = model(x)
    model.cuda()
    summary(model, (3, 224, 224))
    # print(model)
    from torchvision.models import resnet18
    model = resnet18(pretrained=False)
    model.cuda()
    # print(model)
    summary(model, (3, 224, 224))

