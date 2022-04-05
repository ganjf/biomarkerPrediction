# @author Jiefeng Gan
import os
import json
import math
import mmcv
import argparse
import openslide
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool


def save_wsi_thumbnail(wsi_dir, wsi_lst, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for wsi_name in tqdm(wsi_lst):
        slide = openslide.open_slide(os.path.join(wsi_dir, wsi_name))
        # size = slide.(level_count[-1])
        size = slide.level_dimensions[-1]
        thumbnail = slide.get_thumbnail(size)
        path = os.path.join(save_path, wsi_name.replace('svs', 'png'))
        if not os.path.exists(path):
            thumbnail.save(path)


def _nms(bboxes, threshold=0.1):
    x1, y1 = bboxes[:, 0], bboxes[:, 1]
    x2, y2 = bboxes[:, 2], bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1) + 1e-12
    scores = bboxes[:, 4]
    order = scores.argsort()[::-1]
    keep = []
    while order.shape[0] > 0:
        if order.shape[0] == 1:
            keep.append(order[0])
            break
        else:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[order[1:]], x1[i])
            yy1 = np.maximum(y1[order[1:]], y1[i])
            xx2 = np.minimum(x2[order[1:]], x2[i])
            yy2 = np.minimum(y2[order[1:]], y2[i])
            inter = np.maximum((xx2 - xx1), 0) * np.maximum((yy2 - yy1), 0)
            ious = inter / (areas[order[1:]] + areas[i] - inter)
            idx = np.where(ious <= threshold)[0]
            if idx.shape[0] == 0:
                break
            order = order[idx+1]
    return np.array(keep)


def gen_roi_json(wsi_dir, inference_coco, result_pkl, roi_json):
    with open(inference_coco, 'r') as f:
        data = json.load(f)
    patch_path = data['images']
    results = mmcv.load(result_pkl)
    # statistic of RoI info at wsi-level.
    wsi_detect_results = {}
    for anno, result in tqdm(zip(patch_path, results)):
        path = anno['file_name']
        wsi_name = path.split('_')[0]
        pos_x, pos_y = float(path.split('.')[-2].split('_')[1]), float(path.split('.')[-2].split('_')[2])

        if not wsi_name in wsi_detect_results:
            wsi = openslide.open_slide(os.path.join(wsi_dir, wsi_name))
            mag = wsi.properties['aperio.AppMag']
            wsi_detect_results[wsi_name] = {'roi':[[],[],[],[],[]], 'mag':mag[0:2]}

        if wsi_detect_results[wsi_name]['mag']=='20':
            downsample = 4
        elif wsi_detect_results[wsi_name]['mag']=='40':
            downsample = 8
        for idx, rois in enumerate(result):
            if rois.shape[0] != 0:
                rois[:, 0:4] = rois[:, 0:4] * downsample + np.array([pos_x, pos_y, pos_x, pos_y]).reshape(-1, 4)
                wsi_detect_results[wsi_name]['roi'][idx].append(rois)
    print('Detection result of RoIs ready.')

    # NMS for WSI-level (iou_threshold=0.1), slide windows with overlapping may casue overlap RoI Box.
    total, total_nms = 0, 0
    for key, value in tqdm(wsi_detect_results.items()):
        roi_five_lst, roi_subtype_lst = [], []
        for i in range(5): # For five subtypes.
            roi_lst = value['roi'][i]
            if len(roi_lst) != 0:
                roi_lst_array = np.concatenate(roi_lst, axis=0)
                roi_five_lst.append(roi_lst_array)
                roi_subtype_lst.append(np.full((roi_lst_array.shape[0], 1), fill_value=i))

        roi_five_array = np.concatenate(roi_five_lst, axis=0)
        total += roi_five_array.shape[0]
        roi_subtype_array = np.concatenate(roi_subtype_lst, axis=0)
        keep = _nms(roi_five_array, threshold=0.1)
        roi_five_array = roi_five_array[keep]
        roi_subtype_array = roi_subtype_array[keep]
        roi_array = np.concatenate([roi_five_array, roi_subtype_array], axis=1)
        wsi_detect_results[key]['roi'] = roi_array
        total_nms += roi_array.shape[0]

    mmcv.dump(wsi_detect_results, roi_json, file_format='json', indent=2)
    print('nms=0.1 on RoIs ready.')
    print('total roi for classification: {} / {}'.format(total_nms, total))


def _not_white_black(patch,
        white_ratio=0.5, #空白部分比例阈值设定(超过该阈值不会被保存)
        white_gray=220, # 空白灰度值阈值设定
        black_ratio=0.5, # 黑色部分比例阈值设定
        black_gray=20 # 黑色灰度值阈值设定
    ):
    patch = np.array(patch, dtype=np.uint8)
    # 剔除空白过多的图片
    white_area = np.logical_and(np.logical_and(patch[:, :, 0]>white_gray, patch[:, :, 1]>white_gray), patch[:, :, 2]>white_gray)
    white_ratio_real = np.sum(white_area) / (patch.shape[0] * patch.shape[1])
    # 剔除黑色过多的区域
    black_area = np.logical_and(np.logical_and(patch[:, :, 0]<black_gray, patch[:, :, 1]<black_gray), patch[:, :, 2]<black_gray)
    black_ratio_real = np.sum(black_area) / (patch.shape[0] * patch.shape[1])
    return (white_ratio_real < white_ratio) and (black_ratio_real < black_ratio)


def _single_wsi_clip_grid(param):
    wsi_name, roi_lst = param
    subtype_lst = ['R', 'N', 'S', 'T', 'X']
    print('starting', wsi_name)
    wsi = openslide.open_slide(os.path.join(wsi_dir, wsi_name))
    mag = roi_lst['mag']
    if mag == '20':
        stride = 512
    elif mag == '40':
        stride = 1024
    for roi in roi_lst['roi']:
        x1, y1, x2, y2, score, subtype = roi
        subtype = subtype_lst[int(subtype)]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        w, h = math.ceil(x2 - x1), math.ceil(y2 - y1)
        w_length = stride if w <= stride else (w // stride + 1) * stride
        h_length = stride if h <= stride else (h // stride + 1) * stride
        x1, y1 = int(center_x - w_length // 2), int(center_y - h_length // 2)
        img = wsi.read_region((x1, y1), 0, (w_length, h_length)).convert('RGB')
        for i in range(w_length // stride):
            for j in range(h_length // stride):
                tile = img.crop((i * stride, j * stride, (i + 1) * stride, (j + 1) * stride))
                name = wsi_name + '_' + str(x1 + i * stride) + '_' + str(y1 + j * stride) + '_' + str(stride) + '_' + str(stride) +\
                        '_' + mag +  '_' +  str(score)[0:4] + '_' + subtype + '.png'
                path = os.path.join(roi_dir, name)
                if not os.path.exists(path) and _not_white_black(tile):
                    tile = tile.resize((512, 512), Image.ANTIALIAS)
                    tile.save(path)
    print('finish', wsi_name)


def _single_wsi_clip_padding(param):
    wsi_name, roi_lst = param
    subtype_lst = ['R', 'N', 'S', 'T', 'X']
    print('starting', wsi_name)
    wsi = openslide.open_slide(os.path.join(wsi_dir, wsi_name))
    mag = roi_lst['mag']
    for roi in roi_lst['roi']:
        x1, y1, x2, y2, score, subtype = roi
        subtype = subtype_lst[int(subtype)]
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        w, h = math.ceil(x2 - x1), math.ceil(y2 - y1)
        side_length = max(w, h)
        x1, y1 = int(center_x - side_length // 2), int(center_y - side_length // 2)
        name = wsi_name + '_' + str(x1) + '_' + str(y1) + '_' + str(side_length)+\
                '_' + mag +  '_' +  str(score) + '_' + subtype + '.png'
        path = os.path.join(roi_dir, name)
        if not os.path.exists(path):
            img = wsi.read_region((x1, y1), 0, (side_length, side_length)).convert('RGB')
            if _not_white_black(img):
                img.save(path)
    print('finish', wsi_name)


def clip_rois_patch(detected_json, func):
    with open(detected_json, 'r') as f:
        wsi_detect_results = json.load(f)
    param_lst = [[k, v] for k, v in wsi_detect_results.items()]
    pool = ThreadPool()
    pool.map(func, param_lst)
    pool.close()
    pool.join()
    print('Cropping of RoIs ready.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wsi_dir', type=str, default=None,)
    parser.add_argument('--inference_coco', type=str, default=None)
    parser.add_argument('--result_pkl', type=str, default=None)
    parser.add_argument('--roi_json', type=str, default=None)
    parser.add_argument('--roi_dir', type=str, default=None)
    args = parser.parse_args()

    wsi_dir = args.wsi_dir
    inference_coco = args.inference_coco
    result_pkl = args.result_pkl
    roi_json = args.roi_json
    roi_dir = args.roi_dir

    gen_roi_json(wsi_dir = wsi_dir,
                inference_coco = inference_coco,
                roi_json = roi_json,
                result_pkl = result_pkl)

    # clip_rois_patch(detected_json=roi_json,
    #             func=_single_wsi_clip_padding)

    clip_rois_patch(detected_json=roi_json,
                func=_single_wsi_clip_grid)