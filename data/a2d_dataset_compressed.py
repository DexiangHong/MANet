import pickle
import time

import cv2
from torch.utils.data import Dataset
import os
import csv
from torchvision import transforms
import h5py
import numpy as np
from PIL import Image
from transformers import BertTokenizer
import torch
from torch.nn import functional as F
import mmcv
from utils.distribute import is_main_process, synchronize
import random
from tqdm import tqdm
from utils.reader import *

GOP_SIZE = 12


class A2DDatasetCompressed(Dataset):
    def __init__(self, cfg, mode='train'):
        assert mode in ['train', 'test']

        clip_len = cfg.INPUT.CLIP_LEN_COMPRESSED
        root = cfg.DATASETS.ROOT

        self.target_shape = (320, 320)

        self.failed_info_dict_path = os.path.join(root, 'failed_video_instance.pkl')

        with open(self.failed_info_dict_path, 'rb') as f:
            self.failed_info_dict = pickle.load(f)

        self.data_root = cfg.DATASETS.ROOT
        self.mode = mode
        self.frame_root = os.path.join(root, 'Release', 'Frames')
        self.compressed_root = os.path.join(root, 'Release', 'compressed_gop_12')
        self.clip_len = clip_len

        anno_txt_path = os.path.join(root, 'a2d_annotation.txt')
        self.mask_root = os.path.join(root, 'a2d_annotation_with_instances')
        self.split_info = {'train': [0], 'test': [1]}

        self.instance_list = parse_annotations_file(anno_txt_path)
        self.split_dict = generate_split_dict(os.path.join(self.data_root, 'Release', 'videoset.csv'))
        self.item_list = self.get_item_list()

        self.img_transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.tokenizer = BertTokenizer.from_pretrained('./bert_pretrain/bert_base')

    def get_item_list(self):
        item_list = []
        for row in self.instance_list:
            video_name, instance_id, sentence = row[0], int(row[1]), row[2]
            if self.split_dict[video_name] in self.split_info[self.mode]:
                mask_dir = os.path.join(self.mask_root, video_name)
                mask_frame_ids = [int(name[:-3]) for name in sorted(os.listdir(mask_dir))]
                for frame_id in mask_frame_ids:
                    if frame_id in self.failed_info_dict[video_name]:
                        # print('in')
                        continue
                    item_list.append((row, frame_id))

        return item_list

    def __getitem__(self, item):
        row, frame_idx = self.item_list[item][0], self.item_list[item][1]
        video_name, instance_id, sentence = row[0], int(row[1]), row[2]
        mask_path = os.path.join(self.mask_root, video_name, '{:05d}.h5'.format(frame_idx))
        if (frame_idx-1) % GOP_SIZE == 0:
            # frame_idx_range = np.array(
            #     [i for i in range(frame_idx - self.clip_len*GOP_SIZE // 2, frame_idx + self.clip_len*GOP_SIZE // 2, GOP_SIZE)])
            frame_idx_range = [frame_idx - (i+1)*GOP_SIZE for i in range((self.clip_len-1)//2)] + [frame_idx] + [frame_idx + (i+1)*GOP_SIZE for i in range((self.clip_len-1)//2)]
            frame_idx_range = np.array(frame_idx_range)

        else:
            res_minus = - ((frame_idx-1) % GOP_SIZE)
            res_add = GOP_SIZE - abs(res_minus)
            if abs(res_minus) > abs(res_add):
                frame_idx_reassign = frame_idx + res_add
            else:
                frame_idx_reassign = frame_idx + res_minus
            # print(frame_idx_reassign)
            frame_idx_range = [frame_idx_reassign - (i+1)*GOP_SIZE for i in range((self.clip_len-1)//2)] + [frame_idx_reassign] + [frame_idx_reassign + (i+1)*GOP_SIZE for i in range((self.clip_len-1)//2)]
            frame_idx_range = np.array(frame_idx_range)
        # print(frame_idx, frame_idx_range)
        # print(frame_idx_range)
        # print(frame_idx)
        frame_dir = os.path.join(self.frame_root, video_name)
        compressed_res_dir = os.path.join(self.compressed_root, 'res', video_name)
        compressed_mv_dir = os.path.join(self.compressed_root, 'mv', video_name)
        max_length = min(int(sorted(os.listdir(compressed_mv_dir))[-1][:-4]), int(sorted(os.listdir(compressed_res_dir))[-1][:-4]))

        # max_length = len(os.listdir(frame_dir))
        frame_idx_range[frame_idx_range >= max_length] = max_length
        frame_idx_range[frame_idx_range <= 1] = 1
        imgs = [self.img_transform(image_loader(os.path.join(frame_dir, '{:05d}.jpg'.format(i)))) for i in
                frame_idx_range]
        key_frame = self.img_transform(image_loader(os.path.join(frame_dir, '{:05d}.jpg'.format(frame_idx))))
        mask, bbox, mask_other, bbox_other = get_masks(mask_path, instance_id, return_others=True)
        if (mask > 0).any():
            valid = 1
        else:
            valid = 0
        motion_frame_indices = []
        for i in frame_idx_range:
            motion_frame_indices.extend([i+j for j in range(0, GOP_SIZE-1)])

        motions = torch.stack([load_side_data(self.compressed_root, video_name, i, 'mv') for i in motion_frame_indices], dim=0)
        residuals = torch.stack([load_side_data(self.compressed_root, video_name, i, 'res') for i in motion_frame_indices], dim=0)

        encoder_dict = self.tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=20,
                                                  padding='max_length', return_attention_mask=True,
                                                  return_tensors='pt', truncation=True)
        word_ids = encoder_dict['input_ids'].squeeze()
        attention_mask = encoder_dict['attention_mask'].squeeze()
        imgs = torch.stack(imgs, dim=0)
        mask_resize = F.interpolate(torch.tensor(mask).float()[None, None, ...], (320, 320), mode='bilinear', align_corners=True).squeeze()

        logits = torch.tensor(0, dtype=torch.long)
        valid = torch.tensor(valid)
        valid_indices = np.argwhere(np.arange(frame_idx_range[0], frame_idx_range[-1])==frame_idx).squeeze()
        ori_size = mask.shape
        # print(len(frame_idx_range))
        # print(valid_indices)
        # print(imgs.shape)
        # print(motions.shape)
        # print(frame_idx_range)
        sample = {
            'img_seq': imgs,
            'key_frame': key_frame,
            'word_ids': word_ids,
            'attention_mask': attention_mask,
            'motions': motions,
            'residuals': residuals,
            'label': mask_resize,
            'video_name': video_name,
            'frame_idx': frame_idx,
            'sentence': sentence,
            'instance_id': instance_id,
            'logits': logits,
            'valid': valid,
            'valid_indices': valid_indices,
            'ori_size': ori_size
        }
        return sample

    def __len__(self):
        return len(self.item_list)


if __name__ == '__main__':
    from modeling import cfg
    from torch.utils.data import DataLoader
    cfg.DATASETS.ROOT = '../Dataset/A2D'
    cfg.INPUT.CLIP_LEN_COMPRESSED = 3
    train_data = A2DDatasetCompressed(cfg, mode='train')

    train_loader = DataLoader(train_data, batch_size=8, num_workers=0)
    try:
        for i, sample in enumerate(tqdm(train_data)):
            pass
    except Exception as e:
        print(e)
        print(i)






