# @author Jiefeng Gan
import os
import argparse
import openslide
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool


def not_white_black(patch, valid_size,
                white_thr = 220,
                white_ratio = 0.5,
                black_thr = 20,
                black_ratio = 0.5):
    patch = np.array(patch, dtype=np.uint8)
    patch = patch[:valid_size[1], :valid_size[0], :]
    # 剔除空白过多的图片
    white_area = np.logical_and(np.logical_and(patch[:, :, 0]>white_thr, patch[:, :, 1]>white_thr),
                                patch[:, :, 2]>white_thr)
    white_ratio_real = np.sum(white_area) / (patch.shape[0] * patch.shape[1])
    # 剔除黑色过多的区域
    black_area = np.logical_and(np.logical_and(patch[:, :, 0]<black_thr, patch[:, :, 1]<black_thr),
                                patch[:, :, 2]<black_thr)
    black_ratio_real = np.sum(black_area) / (patch.shape[0] * patch.shape[1])

    return (white_ratio_real < white_ratio) and (black_ratio_real < black_ratio)


def _single_wsi_clip(wsi_info):
    wsi_path, patch_dir = wsi_info

    print('reading...', wsi_path)
    wsi = openslide.open_slide(os.path.join(wsi_dir, wsi_path))
    # slide_thumbnail = wsi.get_thumbnail((500, 500))
    # slide_thumbnail.save('./thumbnail.jpg')
    orig_w = int(wsi.properties.get('openslide.level[0].width'))
    orig_h = int(wsi.properties.get('openslide.level[0].height'))
    mag = wsi.properties['aperio.AppMag']
    if mag[0:2] == '20':
        patch_size=(4000, 4000)
        stride_size=(3000, 3000)
    elif mag[0:2] == '40':
        patch_size=(8000, 8000)
        stride_size=(6000, 6000)
    resize_patch = 1000
    downsample = patch_size[0] // resize_patch

    save = False
    while (not save and stride_size[0] >= 1000):
        for y in range(0, orig_h, stride_size[1]):
            for x in range(0, orig_w, stride_size[0]):
                # 根据坐标，截取图片
                save_path = os.path.join(patch_dir, os.path.basename(wsi_path) + '_' + str(x)+ '_' + str(y) + '.jpg')
                if not os.path.exists(save_path):
                    patch = wsi.read_region((x, y), 0, patch_size).convert('RGB') 
                    patch = patch.resize((resize_patch, resize_patch), Image.BICUBIC)
                    valid_size = (min(x + patch_size[0], orig_w) - x, min(y + patch_size[1], orig_h) - y)
                    valid_size = (valid_size[0] // downsample, valid_size[1] // downsample)
                    if not_white_black(patch, valid_size):
                        patch.save(save_path)
                        print('save_path:', save_path)
                        if not save:
                            save = True
                else:
                    print('save_path:', save_path)
                    if not save:
                        save = True
        if not save:
            stride_size = (stride_size[0]-1000, stride_size[1]-1000)
    wsi.close()


def wsi_clip(wsi_lst, patch_dir):
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)
    param_lst = [[wsi_path, patch_dir] for wsi_path in wsi_lst]

    pool = ThreadPool()
    pool.map(_single_wsi_clip, param_lst)
    pool.close()
    pool.join()
    print('finish!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', type=str, default=None,)
    parser.add_argument('--patch_dir', type=str, default=None)
    parser.add_argument('--pid_dis', type=str, default=None)
    parser.add_argument('--biomarker_txt', type=str, default=None)
    args = parser.parse_args()
    
    wsi_dir = args.wsi_dir
    patch_dir = args.patch_dir
    pid_dis = args.pid_dis
    biomarker_txt = args.biomarker_txt

    # Set WSIs which need to be processed.
    # TCGA
    wsi_lst = list(pd.read_csv(pid_dis)['name'])

    # CPTAC
    # biomarker_data = pd.read_csv(biomarker_txt, delimiter='\t')
    # biomarker_dict = {k: v for k, v in zip(biomarker_data['patient'], biomarker_data['TMB'])}
    # wsi_lst = os.listdir(wsi_dir)
    # pid_lst = [x[:-7] for x in wsi_lst]
    # pid_lst = [x for x in pid_lst if x in biomarker_dict]
    # pid_lst = list(set(pid_lst))
    # pid_lst = [x[6:] for x in pid_lst]
    # wsi_lst = [x for x in wsi_lst if x[:-7] in pid_lst]
    # wsi_lst = sorted(wsi_lst)

    wsi_clip(
        wsi_lst = wsi_lst,
        patch_dir = patch_dir)