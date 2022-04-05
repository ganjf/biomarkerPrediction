# @author Jiefeng Gan
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, confusion_matrix

import sys
sys.path.append(os.getcwd())
from model.xception import XceptionCertainty
from dataset import tcgaSubtypePseudo

os.environ['CUDA_VISIBLE_DEVICES']= '0, 1'
torch.multiprocessing.set_sharing_strategy('file_system')

def test(loader, model, pred_thr=0.5, confidence_thr=0.5, biomarker_thr=206, biomarker='TMB'):
    model.eval()
    neg, pos = 0., 0.
    out_lst, conf_lst, biomarker_lst = [], [], []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(loader)):
            inputs = inputs.cuda()
            outputs, confidence = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            confidence = torch.sigmoid(confidence)
            neg += torch.sum(labels[(confidence >= confidence_thr).squeeze()] == 0).item()
            pos += torch.sum(labels[(confidence >= confidence_thr).squeeze()] == 1).item()
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
        anno = dataset.data_df.iloc[index]
        data_biomarker = dataset.data_biomarker
        pid_name = anno['name'].split('_')[0][0:12] # TCGA
        # pid_name = anno['name'].split('.')[0][:-3] # CPTAC
        if not pid_name in pid_result:
            biomarker_numerical = float(data_biomarker[data_biomarker['patient'] == pid_name][biomarker])
            biomarker_status = 1 if biomarker_numerical >= biomarker_thr else 0
            pid_result[pid_name] = {'pred': [], 'conf': [], 'label':biomarker_status}
        pid_result[pid_name]['pred'].append(out_array[index, 1])
        pid_result[pid_name]['conf'].append(conf_array[index, 0])
        if conf_array[index, 0] >= confidence_thr:
            score_lst.append(out_array[index, 1])
            pred_lst.append(1) if out_array[index, 1] >= 0.5 else pred_lst.append(0)
            gt_lst.append(biomarker_array[index].item())
    # For tile level.
    acc = accuracy_score(np.array(gt_lst), np.array(pred_lst))
    auc = roc_auc_score(np.array(gt_lst), np.array(score_lst))
    print('tile level, acc:{:.4f}, auc:{:.4f}, num:{} / {}'.format(acc, auc, len(gt_lst), out_array.shape[0]))
    # For patient level.
    pred_lst, score_lst, biomarker_lst = [], [], []
    num_valid, num_total = 0, 0
    for _, value in pid_result.items():
        value['pred'], value['conf'] = np.array(value['pred']), np.array(value['conf'])
        index = value['conf'] >= confidence_thr
        pred_temp = value['pred'][index]
        if pred_temp.shape[0] == 0:
            index = np.argsort(value['conf'])[-1:]
            pred_temp = value['pred'][index]
        num_valid += pred_temp.shape[0]
        num_total += value['conf'].shape[0]

        pred = np.median(pred_temp)
        pred_lst.append(1) if pred >= pred_thr else pred_lst.append(0)
        score_lst.append(pred)
        biomarker_status = value['label']
        biomarker_lst.append(biomarker_status)
    score, pred, gt = np.array(score_lst), np.array(pred_lst), np.array(biomarker_lst)
    acc = accuracy_score(gt, pred)
    auc = roc_auc_score(gt, score)
    ap = average_precision_score(gt, score)
    cm = confusion_matrix(gt, pred)
    print('confidence score >= {}: {} / {}'.format(confidence_thr, num_valid, num_total))
    # print('patient level, acc:{}, auc:{}'.format(acc, auc))
    print('patient level, acc:{}, auc:{}, ap:{}'.format(acc, auc, ap))
    print(cm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--biomarker_txt', type=str, default=None)
    parser.add_argument('--biomarker', type=str, default=None)
    parser.add_argument('--confidence_threshold', type=float, default=None, help='format of percentage.')
    parser.add_argument('--biomarker_threshold', type=float, default=None)
    args = parser.parse_args()

    transform = A.Compose([
        A.Resize(299, 299),
        A.Normalize(mean=[0.7218, 0.5980, 0.7143],
                    std=[0.1853, 0.2281, 0.1877]),
        ToTensorV2()
    ])

    checkpoint = args.checkpoint
    data_dir = args.data_dir
    input = args.input
    biomarker_txt = args.biomarker_txt
    biomarker = args.biomarker
    
    confidence_threshold = args.confidence_threshold / 100
    biomarker_threshold = args.biomarker_threshold
    pred_threshold = 0.5

    print('checkpoint: ', checkpoint)
    print('input: ', input)
    print('biomarker: ', biomarker)
    print('threshold: ', biomarker_threshold)
    print('binary prediction threshold: ', pred_threshold)
    print('confidence threshold: ', confidence_threshold)

    dataset = tcgaSubtypePseudo(
        data_dir,
        input,
        biomarker_txt=biomarker_txt,
        biomarker_threshold=biomarker_threshold,
        transform=transform,
        biomarker=biomarker)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=12)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    model = XceptionCertainty(num_classes=2)
    model.load_state_dict(torch.load(checkpoint))
    model.to(device)
    model = torch.nn.DataParallel(model)


    test(loader,
        model,
        pred_thr=pred_threshold,
        biomarker_thr=biomarker_threshold,
        confidence_thr=confidence_threshold,
        biomarker=biomarker)

