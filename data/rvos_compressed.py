import sys
import os

import torch
from torch.utils.data import Dataset
# from utils.misc import nested_tensor_from_videos_list

import json
from tqdm import tqdm

from torchvision import transforms
from transformers import BertTokenizer
from utils.reader import *
from glob import glob

GOP_SIZE = 5
class ReferYouTubeVOSDatasetCompressed(Dataset):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the full
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.
    """
    def __init__(self, cfg, mode='train'):
        super(ReferYouTubeVOSDatasetCompressed, self).__init__()
        clip_len = cfg.INPUT.CLIP_LEN_COMPRESSED
        root = cfg.DATASETS.ROOT
        self.end_to_end = cfg.MODEL.END_TO_END
        self.target_shape = (320, 320)

        self.data_root = cfg.DATASETS.ROOT
        self.mode = mode
        self.frame_root = os.path.join(root, mode, 'JPEGImages')
        self.all_frame_root = os.path.join(root, f'{mode}_all_frames', 'JPEGImages')
        self.compressed_root = os.path.join(root, f'compressed_{mode}')
        self.clip_len = clip_len
        # if mode != 'train':
        #     self.clip_len = 36

        anno_txt_path = os.path.join(root, 'meta_expressions', mode, 'meta_expressions.json')
        with open(anno_txt_path, 'r') as f:
            self.anno_txt_express = json.load(f)
        if mode == 'train':
            self.mask_root = os.path.join(root, mode, 'Annotations')
        else:
            self.mask_root = None

        self.split_info = {'train': 0, 'test': 1}

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
        sample_list = []
        # print(self.anno_txt_express.keys())
        video_names = os.listdir(self.frame_root)
        # if self.end_to_end:
        for video_name in tqdm(video_names):
            exp_dict = self.anno_txt_express['videos'][video_name]['expressions']
            all_frames = self.anno_txt_express['videos'][video_name]['frames']

            for exp_index in exp_dict:
                if self.mode == 'train':
                    exp, obj_id = exp_dict[exp_index]['exp'], int(exp_dict[exp_index]['obj_id'])
                else:
                    exp = exp_dict[exp_index]['exp']
                    obj_id = int(exp_index)
                # if self.mode != 'train':
                #     sample_list.append((video_name, 0, all_frames, exp, obj_id))
                # else:
                for cur_frame_idx in range(0, len(all_frames), self.clip_len):
                    sample_list.append((video_name, cur_frame_idx, all_frames, exp, obj_id))
                    # for cur_frame_idx in range(len(all_frames)):
                    #     sample_list.append((video_name, cur_frame_idx, all_frames, exp, obj_id))

        return sample_list

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, item):
        video_name, cur_frame_idx, all_frames, exp, obj_id = self.item_list[item]
        # obj_id = int(obj_id)

        frame_dir = os.path.join(self.frame_root, video_name)
        all_frame_dir = os.path.join(self.all_frame_root, video_name)

        num_frames_key = len(all_frames)
        num_all_frames = len(glob(os.path.join(all_frame_dir, '*.jpg')))
        # if self.end_to_end:
        img_seq_indices = [min(max(index, 0), num_frames_key - 1) for index in
                           range(cur_frame_idx, cur_frame_idx + self.clip_len)]

        imgs = [self.img_transform(image_loader(os.path.join(frame_dir, '{}.jpg'.format(all_frames[i])))) for i in
                img_seq_indices]

        frame_idx_range = [int(all_frames[i]) for i in img_seq_indices]

        # compressed_res_dir = os.path.join(self.compressed_root, 'res', video_name)
        # compressed_mv_dir = os.path.join(self.compressed_root, 'mv', video_name)
        motion_frame_indices = []
        for i in frame_idx_range:
            motion_frame_indices.extend([i+j for j in range(0, GOP_SIZE-1)])

        motions = torch.stack([load_side_data(self.compressed_root, video_name, i, 'mv') for i in motion_frame_indices],
                              dim=0)
        residuals = torch.stack(
            [load_side_data(self.compressed_root, video_name, i, 'res') for i in motion_frame_indices], dim=0)

        key_frame = image_loader(os.path.join(frame_dir, '{}.jpg'.format(all_frames[cur_frame_idx])))
        ori_size = torch.tensor([key_frame.height, key_frame.width])
        # key_frame = self.img_transform(key_frame)
        encoder_dict = self.tokenizer.encode_plus(exp, add_special_tokens=True, max_length=20,
                                                  padding='max_length', return_attention_mask=True,
                                                  return_tensors='pt', truncation=True)
        word_ids = encoder_dict['input_ids'].squeeze()
        attention_mask = encoder_dict['attention_mask'].squeeze()
        imgs = torch.stack(imgs, dim=0)
        # logits = torch.tensor(0, dtype=torch.long)
        logits = torch.zeros([self.clip_len], dtype=torch.long)

        sample = {
            'img_seq': imgs,
            'word_ids': word_ids,
            'attention_mask': attention_mask,
            'motions': motions,
            'residuals': residuals,
            'logits': logits
        }
        # if self.end_to_end:
        if self.mode == 'train':
            mask_paths = [os.path.join(self.mask_root, video_name, all_frames[i]+'.png') for i in img_seq_indices]
            masks = [torch.tensor(np.array(Image.open(mask_path))) == obj_id for mask_path in mask_paths]
            masks_resize = torch.stack([F.interpolate(mask[None, None, ...].float(), size=self.target_shape, mode='bilinear',
                                        align_corners=False).squeeze() for mask in masks], dim=0)
            valid = []
            for mask in masks_resize:
                if (mask > 0).any():
                    valid.append(1)
                else:
                    valid.append(0)
            valid = torch.tensor(valid)

            sample.update({'label': masks_resize, 'valid': valid})

        valid_indices = torch.arange(0, self.clip_len, dtype=torch.int64)
        img_mask = torch.zeros([self.clip_len, *self.target_shape])
        resize_frame_size = torch.tensor(self.target_shape, dtype=torch.int64)
        # ori_frame_size = ori_size
        cur_frame_indices = torch.tensor([i for i in img_seq_indices]).int()

        sample.update({'resized_frame_size': resize_frame_size,
                       'ori_size': ori_size, 'valid_indices':valid_indices, 'img_mask': img_mask,
                       'cur_frame': cur_frame_indices, 'video_name': video_name, 'exp_id': obj_id})

        return sample


if __name__ == '__main__':

    from modeling import cfg

    cfg.DATASETS.ROOT = '/home/dexiang/Dataset/YoutubeVos/rvos/'
    cfg.INPUT.CLIP_LEN = 3
    cfg.MODEL.END_TO_END = True

    dataset = ReferYouTubeVOSDatasetCompressed(cfg, mode='valid')

    item_list = dataset.item_list
    len_item_list = len(item_list)
    # for i in range(len(dataset)):
    #     sample = dataset[i]

    sample_single = dataset[0]

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    #
    for sample in tqdm(loader):
        print(sample['label'].shape)
        # print(sample[''])
        # print(sample['video_name'])
        # print(sample['ori_size'].shape)


    # for i in range(len(dataset)):
    #     sample = dataset[i]
    #     print(sample['video_name'])
#



















