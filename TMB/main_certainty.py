# @author Jiefeng Gan
import os
import time
import copy
import torch
import logging
import shutil
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

import sys
sys.path.append(os.getcwd())
from dataset import CertaintyCrossEntropy, Cutout
from model.xception import XceptionCertainty
from dataset import tcgaSubtypePseudo, HEJitter

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES']= '0, 1'


def test(loader, model, criterion, threshold=206, biomarker='TMB'):
    model.eval()
    running_loss, neg, pos = 0., 0., 0.
    out_lst, conf_lst, biomarker_lst = [], [], []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(loader)):
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs, confidence = model(inputs)
            loss = criterion(outputs, confidence, labels)
            running_loss += loss.item() * inputs.size(0)
            outputs = F.softmax(outputs, dim=1)
            confidence = torch.sigmoid(confidence)
            neg += torch.sum(labels[(confidence >= 0.5).squeeze()] == 0).item()
            pos += torch.sum(labels[(confidence >= 0.5).squeeze()] == 1).item()
            out_lst.append(outputs)
            conf_lst.append(confidence)
            biomarker_lst.append(labels)

    out_lst = [x.cpu().numpy() for x in out_lst]
    out_array =  np.concatenate(out_lst, axis=0)
    conf_lst = [x.cpu().numpy() for x in conf_lst]
    conf_array = np.concatenate(conf_lst, axis=0)
    biomarker_lst = [x.cpu().numpy() for x in biomarker_lst]
    biomarker_array = np.concatenate(biomarker_lst, axis=0)

    pid_result = {}
    score_lst, pred_lst, gt_lst = [], [], []
    for index in range(out_array.shape[0]):
        anno = dataset_valid.data_df.iloc[index]
        data_biomarker = dataset_valid.data_biomarker
        pid_name = anno['name'].split('_')[0][0:12]
        if not pid_name in pid_result:
            biomarker_numerical = float(data_biomarker[data_biomarker['patient'] == pid_name][biomarker])
            biomarker = 1 if biomarker_numerical >= threshold else 0
            pid_result[pid_name] = {'pred': [], 'conf': [], 'label':biomarker}
        pid_result[pid_name]['pred'].append(out_array[index, 1])
        pid_result[pid_name]['conf'].append(conf_array[index, 0])
        if conf_array[index, 0] >= 0.5:
            score_lst.append(out_array[index, 1])
            pred_lst.append(1) if out_array[index, 1] >= 0.5 else pred_lst.append(0)
            gt_lst.append(biomarker_array[index].item())
    # For tile level.
    if len(score_lst) != 0:
        acc = accuracy_score(np.array(gt_lst), np.array(pred_lst))
        auc = roc_auc_score(np.array(gt_lst), np.array(score_lst))
        logging.info('tile level, acc:{:.4f}, auc:{:.4f}, num:{} / {}, neg:{}, pos:{}'.format(acc, auc, len(gt_lst), out_array.shape[0], neg, pos))
    # For patient level.
    pred_lst, score_lst, biomarker_lst = [], [], []
    confidence_30, confidence_40, confidence_50, total = 0, 0, 0, 0
    for _, value in pid_result.items():
        value['pred'], value['conf'] = np.array(value['pred']), np.array(value['conf'])
        index = value['conf'] >= 0.5
        pred_temp = value['pred'][index]
        confidence_30 += np.sum((value['conf'] >= 0.3))
        confidence_40 += np.sum((value['conf'] >= 0.4))
        confidence_50 += np.sum((value['conf'] >= 0.5))
        if pred_temp.shape[0] == 0:
            index = np.argsort(value['conf'])[-1:]
            pred_temp = value['pred'][index]
        pred = np.median(pred_temp)
        pred_lst.append(1) if pred >= 0.5 else pred_lst.append(0)
        score_lst.append(pred)
        biomarker = value['label']
        biomarker_lst.append(biomarker)
    score, pred, gt = np.array(score_lst), np.array(pred_lst), np.array(biomarker_lst)
    acc = accuracy_score(gt, pred)
    auc = roc_auc_score(gt, score)
    ap = average_precision_score(gt, score)
    logging.info('patient level, acc:{}, auc:{}, ap:{}'.format(acc, auc, ap))
    logging.info('confidence score >= 0.3: {} / {}'.format(confidence_30, total))
    logging.info('confidence score >= 0.4: {} / {}'.format(confidence_40, total))
    logging.info('confidence score >= 0.5: {} / {}'.format(confidence_50, total))


def train(epoch, loader, model, criterion, optimizer, show_interval=50):
    model.train()
    running_loss, running_corrects, neg, pos = 0., 0., 0., 0.
    for i, (input, target) in enumerate(tqdm(loader)):
        norm_max = 0
        input = input.cuda()
        target = target.cuda()
        optimizer.zero_grad()
        output, confidence = model(input)
        loss = criterion(output, confidence, target)
        loss.backward()

        l2_reg = torch.tensor(0.)
        for _, param in model.named_parameters():
            if param.requires_grad is True:
                norm = torch.norm(param.grad)
                norm_max = max(norm_max, norm)
                l2_reg += torch.norm(param)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
        optimizer.step()
        _, pred = torch.max(output, dim=1)
        running_loss += loss.item() * input.size(0)
        running_corrects += torch.sum(pred == target.data).item()
        neg += torch.sum(target == 0).item()
        pos += torch.sum(target == 1).item()

        if i % show_interval == 0:
            logging.info('epoch: {} / {}, iter: {} / {}, lr: {}, loss: {:.4f}, norm: {}, reg_loss:{:.4f}'.format(
                    epoch, num_epochs, i, len(loader), optimizer.param_groups[0]['lr'], loss.item(), norm_max, l2_reg.item()))
    mean_loss = running_loss / len(loader.dataset)
    acc = running_corrects / len(loader.dataset)
    logging.info('Train, epoch:{}/{}, mean Loss: {:.4f}, acc:{:.4f}, neg:{}, pos:{}'.format(epoch, num_epochs, mean_loss, acc, neg, pos))
    return model


if __name__ == '__main__':
    save_pth = '/mnt/data_2/gjf/detector_TMB_cascadercnn_206_vote_3'
    shutil.copyfile('./TMB/main_certainty.py', os.path.join(save_pth, 'main_certainty.py'))
    shutil.copyfile('./model/xception.py', os.path.join(save_pth, 'xception.py'))
    logging.basicConfig(filename='./log/log_pseudo_7.log', level=logging.INFO)
    logging.info(save_pth)
    
    train_transform = A.Compose([
        A.Resize(299, 299),
        HEJitter(hematoxylin_limit=(-0.1, 0.1), eosin_limit=(-0.1, 0.1), dab_limit=(-0.1, 0.1), p=0.5),
        A.RandomRotate90(p=0.5),
        Cutout(16, n_holes=8, p=0.5),
        A.Normalize(mean=[0.7218, 0.5980, 0.7143],
                    std=[0.1853, 0.2281, 0.1877]),
        ToTensorV2()
    ])

    valid_transform = A.Compose([
        A.Resize(299, 299),
        A.Normalize(mean=[0.7218, 0.5980, 0.7143],
                    std=[0.1853, 0.2281, 0.1877]),
        ToTensorV2()
    ])

    dataDIR = '/mnt/data_2/gjf/detection_data/TCGA_RoI_grid'
    trainDataCSV = './data/TCGA_train_grid.csv'
    validDataCSV = './data/TCGA_test_grid.csv'
    biomarkerDataTXT = './data/TMB_TCGA_LUAD.txt'
    threshold = 206
    biomarker = 'TMB'

    dataset_train = tcgaSubtypePseudo(
        dataDIR,
        trainDataCSV,
        biomarker_txt=biomarkerDataTXT,
        biomarker_threshold=threshold,
        transform=train_transform,
        biomarker=biomarker)
    dataset_valid = tcgaSubtypePseudo(
        dataDIR,
        validDataCSV,
        biomarker_txt=biomarkerDataTXT,
        biomarker_threshold=threshold,
        transform=valid_transform,
        biomarker=biomarker)

    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset_valid, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    model = XceptionCertainty(num_classes=2)
    model.to(device)

    weight = torch.tensor([1., 2.]).float().cuda()
    criterion = CertaintyCrossEntropy(weight=weight)
    optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.95, weight_decay=5e-4)
    model = torch.nn.DataParallel(model)

    iter_per_epoch = len(train_loader)
    num_epochs = 100
    test_interval = 1
    since = time.time()

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model = train(epoch, train_loader, model, criterion, optimizer, show_interval=100)
        torch.save(model.module.state_dict(), os.path.join(save_pth, 'epoch_' + str(epoch) + '.pth'))

        if epoch % test_interval == 0:
            model_copy = copy.deepcopy(model)
            test(val_loader, model_copy, criterion, threshold=threshold, biomarker=biomarker)

        eta = (time.time() - start_time) * (num_epochs - epoch)
        logging.info('eta: {:.0f}m {:.0f}s'.format(eta//60, eta%60))

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

