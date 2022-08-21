from torch import nn
import torch
from torchvision import models
import einops
from torch.nn import functional as F
from modeling.backbone import init_backbone


class Predictor(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Predictor, self).__init__()

        def Conv2D(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
                nn.LeakyReLU(0.1)
            )

        self.conv0 = Conv2D(ch_in, 8, kernel_size=3, stride=1)
        dd = 8
        self.conv1 = Conv2D(ch_in + dd, 8, kernel_size=3, stride=1)
        dd += 8
        self.conv2 = Conv2D(ch_in + dd, 6, kernel_size=3, stride=1)
        dd += 6
        self.conv3 = Conv2D(ch_in + dd, 4, kernel_size=3, stride=1)
        dd += 4
        self.conv4 = Conv2D(ch_in + dd, 2, kernel_size=3, stride=1)
        dd += 2
        self.predict_flow = nn.Conv2d(ch_in + dd, ch_out, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x = torch.cat((self.conv0(x), x), 1)
        x = torch.cat((self.conv1(x), x), 1)
        x = torch.cat((self.conv2(x), x), 1)
        x = torch.cat((self.conv3(x), x), 1)
        x = torch.cat((self.conv4(x), x), 1)
        return self.predict_flow(x)


class DPDA(nn.Module):
    def __init__(self, cfg, project_dims, mode='mv', gop_size=12):
        super().__init__()
        self.mode = mode
        # self._use_gan = cfg.MODEL.USE_GAN
        in_dim = {
            'mv': 2, 'res': 3
        }[mode]
        self.backbone = init_backbone(backbone_name='resnet18', in_channel=in_dim)

        # i_feature_dims = [96, 192, 384, 768]
        # p_feature_dims = [64, 128, 256, 512]
        self.channel_weight_predictors = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        ) for dim in project_dims])
        self.gop_size = gop_size

        self.spatial_modules = nn.ModuleList([Predictor(in_dim + project_dims[i] * 2, 1) for i in range(len(project_dims))])
        self.channel_modules = nn.ModuleList([Predictor(in_dim + project_dims[i] * 2, project_dims[i]) for i in range(len(project_dims))])

    def forward(self, i_features_list, p_motions, batch_size, num_gop):
        """
        Args:
            imgs: (4, 100, 3, 224, 224)
            i_features: (100, 256, 7, 7)
            p_motions: (100, 3, 2, 224, 224)
        Returns:
        """
        # B = imgs.shape[0]
        # num_gop = imgs.shape[1] // GOP
        B = batch_size
        # i_features_o = i_features = i_features.unsqueeze(1).expand(-1, GOP - 1, -1, -1, -1).reshape(-1, *i_features.shape[-3:])  # (bn gop) c h w

        p_motions = einops.rearrange(p_motions, 'bn gop c h w -> (bn gop) c h w')
        p_features_list = self.backbone(p_motions)[1:]
        p_features_list_out = []

        for i, (i_features, p_features) in enumerate(zip(i_features_list, p_features_list)):
            # print(i_features.shape)
            # print(i, i_features.shape,p_features.shape)
            i_features = i_features.unsqueeze(1).expand(-1, self.gop_size-1, -1, -1, -1).reshape(-1, *i_features.shape[-3:])

            p_motions_resized = F.interpolate(p_motions, size=p_features.shape[-2:], mode='bilinear', align_corners=False)

            channel_weight = self.channel_modules[i](torch.cat([p_motions_resized, p_features, i_features], dim=1))
            weight = self.channel_weight_predictors[i](F.adaptive_avg_pool2d(channel_weight, 1).flatten(1))  # (300, 256)

            i_features = i_features * weight.unsqueeze(-1).unsqueeze(-1)  # (bn gop) c h w

            spatial_weight = self.spatial_modules[i](torch.cat([p_motions_resized, p_features, i_features], dim=1))
            spatial_weight = F.softmax(spatial_weight.view(*spatial_weight.shape[:2], -1), dim=-1).view_as(spatial_weight)

            i_features = i_features * spatial_weight
            p_features_out = i_features + p_features
            p_features_out = einops.rearrange(p_features_out, '(b n t) c h w -> b n t c h w', b=B, n=num_gop)
            p_features_list_out.append(p_features_out)

        return p_features_list_out
