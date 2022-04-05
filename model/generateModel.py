from torch import mode
from torchvision import models
import torch.nn as nn
from torchvision.models.resnet import resnet34

def resnet18(pretrained, num_classes):
    model = models.resnet18(pretrained=pretrained)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_features=in_feat, out_features=num_classes)
    nn.init.normal_(model.fc.weight, 0, 0.01)
    nn.init.constant_(model.fc.bias, 0)
    return model


def resnet34(pretrained, num_classes):
    model = models.resnet34(pretrained=pretrained)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_features=in_feat, out_features=num_classes)
    nn.init.normal_(model.fc.weight, 0, 0.01)
    nn.init.constant_(model.fc.bias, 0)
    return model


def resnet50(pretrained, num_classes):
    model = models.resnet50(pretrained=pretrained)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_features=in_feat, out_features=num_classes)
    nn.init.normal_(model.fc.weight, 0, 0.01)
    nn.init.constant_(model.fc.bias, 0)
    return model


def resnet101(pretrained, num_classes):
    model = models.resnet101(pretrained=pretrained)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_features=in_feat, out_features=num_classes)
    nn.init.normal_(model.fc.weight, 0, 0.01)
    nn.init.constant_(model.fc.bias, 0)
    return model


def densenet121(pretrained, num_classes):
    model = models.densenet121(pretrained=pretrained)
    in_feat = model.classifier.in_features
    model.classifier = nn.Linear(in_feat, num_classes)
    nn.init.normal_(model.classifier.weight, 0, 0.01)
    nn.init.constant_(model.classifier.bias, 0)
    return model


def inceptionv3(pretrained, num_classes):
    model = models.inception_v3(pretrained=pretrained, transform_input=False)
    in_feat_aux = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(in_feat_aux, num_classes)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    nn.init.normal_(model.AuxLogits.fc.weight, 0, 0.01)
    nn.init.normal_(model.fc.weight, 0, 0.01)
    nn.init.constant_(model.AuxLogits.fc.bias, 0)
    nn.init.constant_(model.fc.bias, 0)
    return model

# from torchsummary import summary
# model = resnet18(pretrained=False, num_classes=2)
# model.cuda()
# summary(model, (3,224,224))