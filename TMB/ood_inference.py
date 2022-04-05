# @author Jiefeng Gan
import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
sys.path.append(os.getcwd())
from model import *
from dataset import tcgaSubtypePseudo
torch.multiprocessing.set_sharing_strategy('file_system')


def OOD_inference(DataDIR, DataCSV, biomarkerDataTXT, outputCSV, model, biomarker, conf_label='confidence', pred_label='prediction'):
    transform = A.Compose([
        A.Resize(299, 299),
        A.Normalize(mean=[0.7218, 0.5980, 0.7143],
                    std=[0.1853, 0.2281, 0.1877]),
        ToTensorV2()
    ])
    dataset = tcgaSubtypePseudo(
        DataDIR,
        DataCSV,
        biomarker_txt=biomarkerDataTXT,
        biomarker_threshold=threshold,
        transform=transform,
        biomarker=biomarker)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=12)
    
    # pred_lst, confidence_lst, label_lst = [], [], []
    confidence_lst, pred_lst = [], []
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(tqdm(loader)):
            inputs = inputs.cuda()
            pred, confidence = model(inputs)
            pred = torch.softmax(pred, dim=1)
            pred_lst.append(pred)
            # label_lst.append(labels)
            confidence = torch.sigmoid(confidence)
            confidence_lst.append(confidence)
    pred_lst = [x.cpu().numpy() for x in pred_lst]
    pred_array = np.concatenate(pred_lst, axis=0)
    # label_lst = [x.numpy() for x in label_lst]
    # label_array = np.concatenate(label_lst, axis=0)
    confidence_lst = [x.cpu().numpy() for x in confidence_lst]
    confidence_array = np.concatenate(confidence_lst, axis=0).squeeze()

    data = pd.read_csv(DataCSV)
    data[pred_label] = list(pred_array[:, 1])
    data[conf_label] = list(confidence_array)
    data.to_csv(outputCSV, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--input', type=str, nargs='+')
    parser.add_argument('--output', type=str, nargs='+')
    parser.add_argument('--biomarker_txt', type=str, default=None)
    parser.add_argument('--biomarker', type=str, default=None)
    parser.add_argument('--biomarker_threshold', type=float, default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    data_dir = args.data_dir
    biomarker_txt = args.biomarker_txt
    biomarker = args.biomarker
    input_output = zip(args.input, args.output)
    threshold = args.biomarker_threshold

    model = XceptionCertainty(num_classes=2)
    model.load_state_dict(torch.load(checkpoint))
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()
    print(args.input)
    print(args.output)
    # exit()

    for in_csv, out_csv in input_output:
        print('input: {}'.format(in_csv))
        print('Save to: {}'.format(out_csv))
        OOD_inference(
            DataDIR=data_dir,
            DataCSV=in_csv,
            biomarkerDataTXT=biomarker_txt,
            outputCSV=out_csv,
            model=model,
            biomarker=biomarker
        )