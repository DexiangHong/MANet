import os

import einops
import torch
import torch.nn as nn

import mmaction
from mmaction.models import build_model
from mmcv import Config


class CSN(nn.Module):
    def __init__(self, pretrain=True, downsample=True):
        super().__init__()
        mmaction_root = os.path.dirname(os.path.abspath(mmaction.__file__))
        if downsample:
            config_file = os.path.join(mmaction_root, os.pardir,
                                       'configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb.py')
        else:
            config_file = os.path.join(mmaction_root, os.pardir, 'configs/recognition/csn/ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_no_downsample.py')


        cfg = Config.fromfile(config_file)

        model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        if pretrain:

            state_dict = torch.load(os.path.join(mmaction_root, os.pardir, 'ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth'))
            print('load from ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth', flush=True)
            model.load_state_dict(state_dict['state_dict'])

            del model.cls_head
        self.model = model

    def forward(self, x):
        x = einops.rearrange(x, 'b t c h w -> b c t h w')
        """csn backbone need (B, C, T, H, W) format"""
        x = self.model.extract_feat(x)
        x = einops.rearrange(x, 'b c t h w -> b t c h w')
        return x


class video_swin_transformer(nn.Module):
    def __init__(self, pretrain=True, downsample=True):
        super().__init__()
        mmaction_root = os.path.dirname(os.path.abspath(mmaction.__file__))
        if downsample:
            config_file = os.path.join(mmaction_root, os.pardir,
                                       'configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py')
        else:
            config_file = os.path.join(mmaction_root, os.pardir, 'configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k_no_down.py')


        cfg = Config.fromfile(config_file)

        model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        if pretrain:
            state_dict = torch.load(os.path.join(mmaction_root, os.pardir, 'swin_tiny_patch244_window877_kinetics400_1k.pth'))
            print('load from ircsn_ig65m_pretrained_bnfrozen_swin_tiny_patch244_window877_kinetics400_1k.pth', flush=True)
            model.load_state_dict(state_dict['state_dict'])

            del model.cls_head
        self.model = model

    def forward(self, x):
        x = einops.rearrange(x, 'b t c h w -> b c t h w')
        """csn backbone need (B, C, T, H, W) format"""
        x = self.model.extract_feat(x)
        x = einops.rearrange(x, 'b c t h w -> b t c h w')
        return x

if __name__ == '__main__':
    model = CSN(pretrain=False, downsample=True)
    x = torch.randn([2,32, 3,320,320])
    out = model(x)
    print(out.shape)