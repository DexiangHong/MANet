import os

import cv2
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import pandas
from os import path
from glob import glob
from tqdm import tqdm
import random
import scipy.io
from PIL import Image
import json
from transformers import BertTokenizer
from torchvision import transforms
import numpy as np
from data.a2d_dataset_compressed import image_loader
from utils.reader import load_side_data
from torch.nn import functional as F
# from misc import nested_tensor_from_videos_list
# from datasets.a2d_sentences.a2d_sentences_dataset import A2dSentencesTransforms
# from datasets.jhmdb_sentences.create_gt_in_coco_format import create_jhmdb_sentences_ground_truth_annotations, get_image_id

GOP_SIZE = 12

class JHMDBSentencesDatasetCompressed(Dataset):
    """
    A Torch dataset for JHMDB-Sentences.
    For more information check out: https://kgavrilyuk.github.io/publication/actor_action/ or the original paper at:
    https://arxiv.org/abs/1803.07485
    """
    def __init__(self, cfg):
        super(JHMDBSentencesDatasetCompressed, self).__init__()
        clip_len = cfg.INPUT.CLIP_LEN_COMPRESSED
        root = cfg.DATASETS.ROOT
        self.target_shape = (320, 320)
        self.root = root
        self.samples_metadata = self.get_samples_metadata(root)

        self.clip_len = clip_len

        self.frame_root = os.path.join(root, 'Frames')
        self.compressed_root = os.path.join(root, 'compressed')
        self.mask_root = os.path.join(root, 'puppet_mask')
        self.img_transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    @staticmethod
    def get_samples_metadata(root_path):
        samples_metadata_file_path = os.path.join(root_path, 'jhmdb_sentences_samples_metadata.json')
        with open(samples_metadata_file_path, 'r') as f:
            samples_metadata = [tuple(a) for a in json.load(f)]
            return samples_metadata

    def __getitem__(self, idx):
        video_id, chosen_frame_path, video_masks_path, video_total_frames, text_query = self.samples_metadata[idx]
        text_query = " ".join(text_query.lower().split())  # clean up the text query

        # read the source window frames:
        chosen_frame_idx = int(chosen_frame_path.split('/')[-1].split('.')[0])

        if (chosen_frame_idx-1) % GOP_SIZE == 0:
            # frame_idx_range = np.array(
            #     [i for i in range(frame_idx - self.clip_len*GOP_SIZE // 2, frame_idx + self.clip_len*GOP_SIZE // 2, GOP_SIZE)])
            frame_idx_range = [chosen_frame_idx - (i+1)*GOP_SIZE for i in range((self.clip_len-1)//2)] + [chosen_frame_idx] + [chosen_frame_idx + (i+1)*GOP_SIZE for i in range((self.clip_len-1)//2)]
            frame_idx_range = np.array(frame_idx_range)

        else:
            res_minus = - ((chosen_frame_idx-1) % GOP_SIZE)
            res_add = GOP_SIZE - abs(res_minus)
            if abs(res_minus) > abs(res_add):
                frame_idx_reassign = chosen_frame_idx + res_add
            else:
                frame_idx_reassign = chosen_frame_idx + res_minus
            # print(frame_idx_reassign)
            frame_idx_range = [frame_idx_reassign - (i+1)*GOP_SIZE for i in range((self.clip_len-1)//2)] + [frame_idx_reassign] + [frame_idx_reassign + (i+1)*GOP_SIZE for i in range((self.clip_len-1)//2)]
            frame_idx_range = np.array(frame_idx_range)
        video_class, video_name = chosen_frame_path.split('/')[-3], chosen_frame_path.split('/')[-2]
        frame_dir = os.path.join(self.frame_root, video_class, video_name)
        compressed_res_dir = os.path.join(self.compressed_root, 'res', video_class, video_name)
        compressed_mv_dir = os.path.join(self.compressed_root, 'mv', video_class, video_name)
        max_length = min(min(int(sorted(os.listdir(compressed_mv_dir))[-1][:-4]), int(sorted(os.listdir(compressed_res_dir))[-1][:-4])), int(sorted(os.listdir(frame_dir))[-1][:-4]))
        frame_idx_range[frame_idx_range >= max_length] = max_length
        frame_idx_range[frame_idx_range <= 1] = 1
        imgs = [self.img_transform(image_loader(os.path.join(frame_dir, '{:05d}.png'.format(i)))) for i in
                frame_idx_range]
        imgs = torch.stack(imgs, dim=0)
        motion_frame_indices = []
        for i in frame_idx_range:
            motion_frame_indices.extend([i + j for j in range(0, GOP_SIZE - 1)])

        motions = torch.stack([load_side_data(self.compressed_root, video_name, i, 'mv') for i in motion_frame_indices],
                              dim=0)
        residuals = torch.stack(
            [load_side_data(self.compressed_root, video_name, i, 'res') for i in motion_frame_indices], dim=0)

        encoder_dict = self.tokenizer.encode_plus(text_query, add_special_tokens=True, max_length=20,
                                                  padding='max_length', return_attention_mask=True,
                                                  return_tensors='pt', truncation=True)
        word_ids = encoder_dict['input_ids'].squeeze()
        attention_mask = encoder_dict['attention_mask'].squeeze()

        video_masks_path = os.path.join(self.mask_root, video_class, video_name, 'puppet_mask.mat')
        all_video_masks = scipy.io.loadmat(video_masks_path)['part_mask'].transpose(2, 0, 1)
        # note that to take the center-frame corresponding mask we switch to 0-indexing:
        instance_mask = torch.tensor(all_video_masks[chosen_frame_idx - 1]).unsqueeze(0)
        mask_resize = F.interpolate(instance_mask.float()[None, ...], (320, 320), mode='bilinear', align_corners=True).squeeze()
        # logits = torch.tensor(0, dtype=torch.long)
        if (instance_mask > 0).any():
            valid = 1
        else:
            valid = 0
        valid = torch.tensor(valid)
        valid_indices = np.argwhere(np.arange(frame_idx_range[0], frame_idx_range[-1]+1) == chosen_frame_idx).squeeze()
        ori_size = instance_mask.shape
        logits = torch.tensor(0, dtype=torch.long)

        # print(len(frame_idx_range))
        # print(valid_indices)
        # print(imgs.shape)
        # print(motions.shape)
        # print(frame_idx_range)
        sample = {
            'img_seq': imgs,
            'word_ids': word_ids,
            'attention_mask': attention_mask,
            'motions': motions,
            'residuals': residuals,
            'label': mask_resize,
            'video_name': video_name,
            'frame_idx': chosen_frame_idx,
            'sentence': text_query,
            'logits': logits,
            'valid': valid,
            'valid_indices': valid_indices,
            'ori_size': ori_size
        }
        return sample


    def __len__(self):
        return len(self.samples_metadata)


def reDecodeVideo(video_root, target_root):
    video_classes = os.listdir(video_root)
    for video_class in tqdm(video_classes):
        video_names = os.listdir(os.path.join(video_root, video_class))
        for video_name in video_names:
            video_prefix = video_name.split('.')[0]
            save_dir = os.path.join(target_root, video_class, video_prefix)
            os.makedirs(save_dir, exist_ok=True)
            cap = cv2.VideoCapture(os.path.join(video_root, video_class, video_name))
            ret, image = cap.read()
            count = 1
            while ret:
                cv2.imwrite(os.path.join(save_dir, '{:05d}.png'.format(count)), image)
                ret, image = cap.read()
                count += 1


if __name__ == '__main__':
    from modeling import cfg
    from torch.utils.data import DataLoader
    cfg.DATASETS.ROOT = '../Dataset/JHMDB'
    cfg.INPUT.CLIP_LEN_COMPRESSED = 3
    train_data = JHMDBSentencesDatasetCompressed(cfg)

    train_loader = DataLoader(train_data, batch_size=8, num_workers=0)
    sample0 = train_data[0]
    try:
        for i, sample in enumerate(tqdm(train_data)):
            # print(sample['img_seq'].shape)
            for key in sample:
                if hasattr(sample[key], 'shape'):
                    # print(key)
                    assert sample[key].shape == sample0[key].shape
            # pass
    except Exception as e:
        print(e)
        print(i)
    # try:
    #     for i, sample in enumerate(train_loader):
    #         pass
    # except Exception as e:
    #     print(e)
    #     print(i)
    #
    # sample = train_data[120]

    sample = train_data[131]


    # reDecodeVideo('../Dataset/JHMDB/ReCompress_Videos', '../Dataset/JHMDB/Frames')

