# @author Jiefeng Gan
import os
import json
import argparse
import numpy as np


def generate_infer_data(patch_path, infer_path):
    data = {
        'images':[], 'annotations':[],
        'categories':[{'id': 1, 'name': 'R'},
                        {'id': 2, 'name': 'N'},
                        {'id': 3, 'name': 'S'},
                        {'id': 4, 'name': 'T'},
                        {'id': 5, 'name': 'X'}]
    }
    # R: Papillary, N:Mucinous, S:solid, T:Lepidic, X:Acinar,
    for idx, name in enumerate(patch_path):
        temp = {'id': idx, 'file_name': name, 'height': 1000, 'width': 1000}
        data['images'].append(temp)
    with open(infer_path, 'w') as f:
        json.dump(data, f)


def not_white_black(patch,
                white_thr = 220,
                white_ratio = 0.5,
                black_thr = 20,
                black_ratio = 0.5):
    patch = np.array(patch, dtype=np.uint8)
    # 剔除空白过多的图片
    white_area = np.logical_and(np.logical_and(patch[:, :, 0]>white_thr, patch[:, :, 1]>white_thr),
                                patch[:, :, 2]>white_thr)
    white_ratio_real = np.sum(white_area) / (patch.shape[0] * patch.shape[1])
    # 剔除黑色过多的区域
    black_area = np.logical_and(np.logical_and(patch[:, :, 0]<black_thr, patch[:, :, 1]<black_thr),
                                patch[:, :, 2]<black_thr)
    black_ratio_real = np.sum(black_area) / (patch.shape[0] * patch.shape[1])
    return (white_ratio_real < white_ratio) and (black_ratio_real < black_ratio)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', type=str, default=None,)
    parser.add_argument('--patch_dir', type=str, default=None)
    parser.add_argument('--inference_coco', type=str, default=None)
    args = parser.parse_args()

    wsi_dir = args.wsi_dir
    patch_dir = args.patch_dir
    inference_coco = args.inference_coco

    patch_path = os.listdir(patch_dir)
    generate_infer_data(patch_path, inference_coco)