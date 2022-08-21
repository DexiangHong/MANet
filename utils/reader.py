from PIL import Image
import csv
import numpy as np
import h5py
import torch
from torch.nn import functional as F
import cv2
import os


def image_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def parse_annotations_file(anno_txt_file):
    with open(anno_txt_file, 'r') as f:
        reader = csv.reader(f)
        return list(reader)[1:]


def generate_split_dict(video_set_file):
    split_dict = {}
    with open(video_set_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            vid, split = row[0], int(row[-1])
            split_dict[vid] = split
    return split_dict


def get_masks(mask_path, instance_id, return_others=False):
    f = h5py.File(mask_path, 'r')
    # print(mask_path)
    instance_ids = f['instance'][:]
    if instance_ids.shape[0] == 1:
        mask = f['reMask'][:].T
        bbox = f['reBBox'][:]
    else:
        index = np.argwhere(instance_ids == instance_id)
        index = np.squeeze(index)
        mask = f['reMask'][index].T
        mask = np.squeeze(mask)
        # print(mask.shape)
        if index.size != 1:
            mask = np.sum(mask, axis=2)
        bbox = f['reBBox'][:, index]

    # print(bbox.shape)
    if len(bbox.shape) == 2:
        bbox = bbox.T
    elif len(bbox.shape) == 1:
        bbox = bbox[None, :]
    # print(bbox.shape)
    # print(f['reBBox'][:].shape)
    # print(instance_id)
    # print(instance_ids)

    mask = mask.astype(np.int32)

    mask_other = []
    bbox_other = []
    if return_others:
        for instance_id_tmp in instance_ids:
            if instance_id_tmp != instance_id:
                index = np.argwhere(instance_ids == instance_id_tmp)
                index = np.squeeze(index)
                mask_tmp = f['reMask'][index].T
                mask_tmp = np.squeeze(mask_tmp)
                # print(mask.shape)
                if index.size != 1:
                    mask_tmp = np.sum(mask_tmp, axis=2)
                bbox_tmp = f['reBBox'][:, index]
                if len(bbox_tmp.shape) == 2:
                    bbox_tmp = bbox_tmp.T
                elif len(bbox_tmp.shape) == 1:
                    bbox_tmp = bbox_tmp[None, :]
                mask_other.append(mask_tmp)
                bbox_other.append(bbox_tmp)

        if len(mask_other) > 0:
            mask_other = np.stack(mask_other, axis=0)
            bbox_other = np.concatenate(bbox_other, axis=0)
        return mask, bbox, mask_other, bbox_other
    return mask, bbox


def preprocess_motion_vector(mv, target_shape=(320, 320)):
    # size = 20
    # mv = (mv * (127.5 / size)).astype(np.int32)
    # mv += 128
    # mv = (np.minimum(np.maximum(mv, 0), 255)).astype(np.uint8)
    mv = torch.from_numpy(mv).permute(2, 0, 1).to(torch.float32)[None]
    mv = F.interpolate(mv, size=target_shape, mode='bilinear', align_corners=False)[0]
    mv = mv / 255.0 - 0.5
    # print(mv.shape)
    # mv = (mv - 0.5) / 0.226
    return mv


def preprocess_residual(res, target_shape=(320, 320)):
    # res += 128
    # res = (np.minimum(np.maximum(res, 0), 255)).astype(np.uint8)
    res = cv2.resize(res, target_shape, interpolation=cv2.INTER_LINEAR)
    res = (res.astype(np.float32) / 255.0 - 0.5) / np.array([0.229, 0.224, 0.225])
    res = np.transpose(res, (2, 0, 1)).astype(np.float32)
    return torch.from_numpy(res)


def load_side_data(compressed_root, video_name, frame_idx, mode):
    assert mode in ['mv', 'res']
    frame_path = os.path.join(compressed_root, mode, video_name, '{:0>4d}.npy'.format(frame_idx))
    if os.path.isfile(frame_path):
        data = np.load(frame_path)
        preprocess_func = {'mv': preprocess_motion_vector, 'res': preprocess_residual}[mode]
        return preprocess_func(data)
    else:
        return torch.zeros([3, 320, 320], dtype=torch.float32) if mode == 'res' else torch.zeros([2, 320, 320], dtype=torch.float32)

