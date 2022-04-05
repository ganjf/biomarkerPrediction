# @author Jiefeng Gan
import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(os.getcwd())
from inference.processRoI import _nms


def rm_overlap(patch_dir_path, threshold=0.1, out_csv='delete.csv'):
    tile_lst = os.listdir(patch_dir_path)
    wsi_lst = list(set([x.split('_')[0] for x in tile_lst]))
    delete_lst = []

    for wsi_name in tqdm(wsi_lst):
        sample_tile_lst = [x for x in tile_lst if x.startswith(wsi_name)]
        boxes = []
        for tile in sample_tile_lst:
            data = tile.split('_')
            x1, y1, w, h, score = int(data[1]), int(data[2]), int(data[3]), int(data[4]), float(data[5][0:4])
            x2, y2 = x1 + w, y1 + h
            boxes.append(np.array([x1, y1, x2, y2, score]))
        boxes = np.stack(boxes, axis=0)
        keep = _nms(boxes, threshold)
        for i in range(len(sample_tile_lst)):
            if not i in keep:
                delete_lst.append(sample_tile_lst[i])
    print('overlap: ', len(delete_lst))
    data_df = pd.DataFrame({'name': delete_lst})
    data_df.to_csv(out_csv, index=False)


def make_dataset(roi_dir, delete_csv, biomarker_txt, data_csv, pid_dis=None):
    data_biomarker = pd.read_csv(biomarker_txt, sep='\t')
    data_biomarker = data_biomarker.fillna(0)
    tile_lst = os.listdir(roi_dir)
    delete_tile = pd.read_csv(delete_csv)['name']
    tile_lst = list(set(tile_lst) - set(delete_tile))
    data_biomarker_name = list(data_biomarker['patient'])
    
    # tile_lst = [x for x in tile_lst if x[0:12] in data_biomarker_name] # TCGA
    tile_lst = [x for x in tile_lst if x.split('.')[0][:-3] in data_biomarker_name] # CPTAC

    if pid_dis is None:
        biomarker_lst, subtype_lst = [], []
        for name in tqdm(tile_lst):
            pid_name = name[0:12] # TCGA
            # pid_name = name.split('.')[0][:-3] # CPTAC
            biomarker = (data_biomarker[data_biomarker['patient'] == pid_name])['TMB'].values[0] # CPTAC
            biomarker_lst.append(biomarker)
            subtype_lst.append(name[-5])
        data = pd.DataFrame({'name':tile_lst, 'TMB':biomarker_lst, 'subtype':subtype_lst})
        data.to_csv(data_csv[0], index=False)
    else:
        pid_data = pd.read_csv(pid_dis)
        pid_train, pid_val = pid_data[pid_data['kind'] == 'train'], pid_data[pid_data['kind'] == 'val']
        pid_train, pid_val = list(pid_train['name']), list(pid_val['name'])
        tile_train_lst = [x for x in tile_lst if x.split('_')[0] in pid_train]
        tile_val_lst = [x for x in tile_lst if x.split('_')[0] in pid_val]

        biomarker_lst, subtype_lst = [], []
        tile_train_lst = random.sample(tile_train_lst, len(tile_train_lst))
        for name in tqdm(tile_train_lst):
            pid_name = name[0:12]
            biomarker = (data_biomarker[data_biomarker['patient'] == pid_name])['TMB'].values[0]
            biomarker_lst.append(biomarker)
            subtype_lst.append(name[-5])
        data = pd.DataFrame({'name':tile_train_lst, 'TMB':biomarker_lst, 'subtype':subtype_lst})
        data.to_csv(data_csv[0], index=False)

        biomarker_lst, subtype_lst = [], []
        for name in tqdm(tile_val_lst):
            pid_name = name[0:12]
            biomarker = (data_biomarker[data_biomarker['patient'] == pid_name])['TMB'].values[0]
            biomarker_lst.append(biomarker)
            subtype_lst.append(name[-5])
        data = pd.DataFrame({'name':tile_val_lst, 'TMB':biomarker_lst, 'subtype':subtype_lst})
        data.to_csv(data_csv[1], index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi_dir', type=str, default=None)
    parser.add_argument('--biomarker_txt', type=str, default=None)
    parser.add_argument('--data_csv', type=str, nargs='+')
    parser.add_argument('--pid_dis', type=str, default=None)
    args = parser.parse_args()

    roi_dir = args.roi_dir
    biomarker_txt = args.biomarker_txt
    data_csv =  args.data_csv
    pid_dis = args.pid_dis

    rm_overlap(patch_dir_path=roi_dir,
        threshold=0.1,
        out_csv=os.path.join('delete.csv'))

    make_dataset(roi_dir=roi_dir,
        delete_csv='delete.csv',
        biomarker_txt=biomarker_txt,
        data_csv=data_csv,
        pid_dis=pid_dis)
    
    os.remove('delete.csv')