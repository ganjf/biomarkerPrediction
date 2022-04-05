# @author Jiefeng Gan
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from sklearn.metrics import confusion_matrix

import sys
sys.path.append(os.getcwd())
from model import Xception
from model import resnet50, densenet121, inceptionv3
from model import MobileNetV3
from dataset import tcgaSubtypePseudo


os.environ['CUDA_VISIBLE_DEVICES']= '0, 1'


def test(loader, model, pred_threshold=0.5, threshold=206, biomarker='TMB'):
    model.eval()
    neg, pos = 0., 0.
    out_lst, biomarker_lst = [], []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(loader)):
            inputs = inputs.cuda()
            outputs = model(inputs)
            neg += torch.sum(labels == 0).item()
            pos += torch.sum(labels == 1).item()
            outputs = torch.softmax(outputs, dim=1)
            out_lst.append(outputs)
            biomarker_lst.append(labels)

    out_lst = [x.cpu().numpy() for x in out_lst]
    out_array =  np.concatenate(out_lst, axis=0)
    biomarker_lst = [x.cpu().numpy() for x in biomarker_lst]
    biomarker_array = np.concatenate(biomarker_lst, axis=0)

    pid_result = {}
    score_lst, pred_lst, gt_lst = [], [], []
    for index in range(out_array.shape[0]):
        name = dataset.data_df.iloc[index]['name']
        data_biomarker = dataset.data_biomarker
        pid_name = name.split('_')[0][0:12] # TCGA
        # pid_name = name.split('.')[0][:-3] # CPTAC
        if not pid_name in pid_result:
            biomarker_numerical = float(data_biomarker[data_biomarker['patient'] == pid_name][biomarker])
            biomarker = 1 if biomarker_numerical >= threshold else 0
            pid_result[pid_name] = {'pred': [], 'label':biomarker}
        pid_result[pid_name]['pred'].append(out_array[index, 1])
        score_lst.append(out_array[index, 1])
        pred_lst.append(1) if out_array[index, 1] >= 0.5 else pred_lst.append(0)
        gt_lst.append(biomarker_array[index].item())
    # For tile level.
    acc = accuracy_score(np.array(gt_lst), np.array(pred_lst))
    auc = roc_auc_score(np.array(gt_lst), np.array(score_lst))
    print('tile level, acc:{:.4f}, auc:{:.4f}, num:{} / {}'.format(acc, auc, len(gt_lst), out_array.shape[0]))
    # For patient level.
    pred_lst, score_lst, biomarker_lst = [], [], []
    for _, value in pid_result.items():
        value['pred'] = np.array(value['pred'])
        pred = np.median(value['pred'])
        biomarker = value['label']
        pred_lst.append(1) if pred >= pred_threshold else pred_lst.append(0)
        score_lst.append(pred)
        biomarker = value['label']
        biomarker_lst.append(biomarker)
    score, pred, gt = np.array(score_lst), np.array(pred_lst), np.array(biomarker_lst)

    acc = accuracy_score(gt, pred)
    auc = roc_auc_score(gt, score)
    ap = average_precision_score(gt, score)
    cm = confusion_matrix(gt, pred)
    tn, fp, fn, tp = cm.ravel()
    sp = tn / (tn + fp)
    se = tp / (tp + fn)
    print('patient level, acc:{}, auc:{}, ap:{}'.format(acc, auc, ap))
    # print('patient level, acc:{}, auc:{}'.format(acc, auc))
    print(cm)
    print('sensitivity: {}, specificity: {}'.format(se, sp))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--biomarker_txt', type=str, default=None)
    parser.add_argument('--biomarker', type=str, default=None)
    parser.add_argument('--biomarker_threshold', type=float, default=None)
    args = parser.parse_args()

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.7218, 0.5980, 0.7143],
                    std=[0.1853, 0.2281, 0.1877]),
        ToTensorV2()
    ])

    checkpoint = args.checkpoint
    data_dir = args.data_dir
    input = args.input
    biomarker_txt = args.biomarker_txt
    biomarker = args.biomarker
    biomarker_threshold = args.biomarker_threshold
    pred_threshold = 0.38

    print('checkpoint: ', checkpoint)
    print('input: ', input)
    print('biomarker: ', biomarker)
    print('bimarker threshold: ', biomarker_threshold)
    print('binary prediction threshold: ', pred_threshold)

    dataset = tcgaSubtypePseudo(
        data_dir,
        input,
        biomarker_txt=biomarker_txt,
        biomarker_threshold=biomarker_threshold,
        transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=12, pin_memory=True)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # model = densenet121(pretrained=False, num_classes=2)
    # model = resnet50(pretrained=False, num_classes=2)
    # model = inceptionv3(pretrained=False, num_classes=2)
    model = Xception(num_classes=2)
    # model = MobileNetV3(num_classes = 2, model_mode='LARGE', multiplier=0.75, dropout_rate=0.5)    

    model.load_state_dict(torch.load(checkpoint))
    model.to(device)
    model = torch.nn.DataParallel(model)

    test(loader,
        model,
        pred_threshold=pred_threshold,
        threshold=biomarker_threshold,
        biomarker=biomarker)

