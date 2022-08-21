import einops
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel
from modeling.loss import overall_dice_loss, mean_dice_loss, BinaryFocalLoss
from modeling.multimodal_transformer import MultimodalTransformer
from modeling.baseline import MLP, LSTM
from modeling.layers.I3D import Unit3D
from modeling.matcher import HungarianMatcher
from modeling.swin_transformer import VideoSwinTransformerBackbone
from modeling.layers.subnet import DPDA


class FPNSpatialDecoder(nn.Module):
    """
    An FPN-like spatial decoder. Generates high-res, semantically rich features which serve as the base for creating
    instance segmentation masks.
    """
    def __init__(self, in_channel, context_dim, fpn_dims):
        super().__init__()

        inter_dims = [context_dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16]
        self.proj = nn.Conv2d(in_channel, context_dim, kernel_size=1)
        self.lay1 = torch.nn.Conv2d(context_dim, inter_dims[0], 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, inter_dims[0])
        self.lay2 = torch.nn.Conv2d(inter_dims[0], inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.context_dim = context_dim

        self.add_extra_layer = len(fpn_dims) == 3
        if self.add_extra_layer:
            self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)
            self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
            self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
            self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)
        else:
            self.out_lay = torch.nn.Conv2d(inter_dims[3], 1, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, layer_features):
        # print(x.shape)
        x = self.proj(x)
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(layer_features[0])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(layer_features[1])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        if self.add_extra_layer:
            cur_fpn = self.adapter3(layer_features[2])
            x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
            x = self.lay5(x)
            x = self.gn5(x)
            x = F.relu(x)

        x = self.out_lay(x)
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CompressedModelMS(nn.Module):
    def __init__(self, cfg):
        super(CompressedModelMS, self).__init__()
        self.cfg = cfg
        self.d_model = 1024

        # self.video_encoder = CSN(pretrain=True, downsample=False)
        self.video_encoder = VideoSwinTransformerBackbone(True, './swin_tiny_patch244_window877_kinetics400_1k.pth', True, 'train')
        self.video_backbone_dims = [192, 384, 768]
        self.project_features_dims = [128, 256, 512]
        self.txt_encoder = BertModel.from_pretrained('bert-base-uncased')

        self.use_heat_map = cfg.MODEL.HEAD.HEATMAP
        self.nq = cfg.MODEL.NQ
        # print(self.use_heat_map)
        self.attention_use_decoder = cfg.MODEL.ATTENTION.USE_DECODER
        self.use_abs_pos = cfg.MODEL.USE_ABS_POS

        if self.use_abs_pos:
            self.pixel_decoder = FPNSpatialDecoder(self.project_features_dims[-1]+3, self.project_features_dims[-1], [512*2, 256*2, 128*2])
        else:
            self.pixel_decoder = FPNSpatialDecoder(self.project_features_dims[-1]+1, self.project_features_dims[-1], [512*2, 256*2, 128*2])

        if self.attention_use_decoder:
            self.obj_queries = nn.Embedding(self.nq, self.project_features_dims[-1])

        self.matcher = HungarianMatcher()

        self.fusion = MultimodalTransformer(in_channels=self.project_features_dims[-1]*2, nheads=8, dropout=0.1, d_model=self.project_features_dims[-1], dim_feedforward=self.project_features_dims[-1],
                                            use_decoder=self.attention_use_decoder)

        if self.use_abs_pos:
            self.project_kernel = MLP(self.project_features_dims[-1], self.project_features_dims[-1], self.project_features_dims[-1]+2, 2)
        else:
            self.project_kernel = MLP(self.project_features_dims[-1], self.project_features_dims[-1], self.project_features_dims[-1], 2)
        self.class_embed = MLP(self.project_features_dims[-1], self.project_features_dims[-1], 1, 2)

        self.fusion_type = cfg.MODEL.FUSION_TYPE

        if cfg.MODEL.BACKBONE.TEXT.FREEZE:
            print('Freeze Bert weight')
            for param in self.txt_encoder.parameters():
                param.requires_grad = False

        if cfg.MODEL.LOSS.FOCAL:
            self.focal_loss = BinaryFocalLoss()

        self.gop_size = 12

        self.mv_module = DPDA(cfg, project_dims=self.project_features_dims, mode='mv')
        self.res_module = DPDA(cfg, project_dims=self.project_features_dims, mode='res')
        self.embeddings = nn.ModuleList([nn.Conv2d(self.video_backbone_dims[i], self.project_features_dims[i], 3, 1, 1) for i in range(len(self.video_backbone_dims))])

        self.temporal_modules = nn.ModuleList([nn.Sequential(
            Unit3D(d_model, d_model, kernel_shape=(3, 1, 1), padding=1),
            nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1)),
            Unit3D(d_model, d_model, kernel_shape=(3, 1, 1), padding=1),
            nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(3, 1, 1)),
            Unit3D(d_model, d_model, kernel_shape=(3, 1, 1), padding=1),
            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            Unit3D(d_model, d_model, kernel_shape=(2, 1, 1), padding=0),
            # nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),
            # Unit3D(self.d_model, self.d_model, kernel_shape=(3, 1, 1), padding=1),
        ) for d_model in self.project_features_dims])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def generate_multi_scale_temporal_features(self, i_features_list, p_mv_feature_list, p_res_feature_list, valid_indices, num_gop):
        num_scales = len(i_features_list)
        B = i_features_list[0].shape[0] // num_gop
        temporal_feature_spatial_list = []
        # temporal_feature_action_list = []
        for i in range(num_scales):
            i_features = i_features_list[i]
            p_features = p_mv_feature_list[i] + p_res_feature_list[i]
            temporal_embeddings = torch.zeros([B, num_gop, self.gop_size, *i_features.shape[-3:]],  dtype=i_features.dtype, device=i_features.device)
            i_features = einops.rearrange(i_features, '(b gop) c h w -> b gop c h w', gop=num_gop)
            temporal_embeddings[:, :, 0] = i_features
            temporal_embeddings[:, :, 1:] = p_features
            temporal_embeddings = einops.rearrange(temporal_embeddings, 'b n gop c h w -> b c (n gop) h w')
            temporal_embeddings_action = self.temporal_modules[i](temporal_embeddings)[:, :, 0, :, :]
            temporal_embeddings_select = []
            for j, valid_index in enumerate(valid_indices):
                temporal_embeddings_select.append(temporal_embeddings[j, :, valid_index, :, :])
            temporal_embeddings_select = torch.stack(temporal_embeddings_select, dim=0)
            temporal_embeddings = torch.cat([temporal_embeddings_action, temporal_embeddings_select], dim=1) # b 2c h w
            temporal_feature_spatial_list.append(temporal_embeddings) # b 2c h w
            # temporal_feature_action_list.append(temporal_embeddings_action) # bchw

        return temporal_feature_spatial_list

    def forward(self, sample):
        heat_maps = None
        img_seq, word_ids, attention_mask, mv, res, valid_indices = \
            sample['img_seq'], sample['word_ids'], sample['attention_mask'], sample['motions'], \
            sample['residuals'], sample['valid_indices']

        temporal_embedding_list = self.video_encoder(img_seq)[1:] # B num_gop c h w

        # print(temporal_embedding.shape)
        temporal_embedding_list = [einops.rearrange(temporal_embedding, 'b t c h w -> (b t) c h w')
                                   for temporal_embedding in temporal_embedding_list]

        temporal_embedding_list = [self.embeddings[i](temporal_embedding) for i, temporal_embedding in enumerate(temporal_embedding_list)]

        # i_features = self.embedding(temporal_embedding)
        word_embedding = self.txt_encoder(word_ids, attention_mask)[0]
        # print(word_embedding.shape)

        B, num_gop, _, _, _ = img_seq.shape
        p_motions = einops.rearrange(mv,  'b (n gop) c h w -> (b n) gop c h w', gop=self.gop_size-1)

        p_features_mv_list = self.mv_module(temporal_embedding_list, p_motions, B, num_gop)
        p_res = einops.rearrange(res, 'b (n gop) c h w -> (b n) gop c h w', gop=self.gop_size-1)
        p_features_res_list = self.res_module(temporal_embedding_list, p_res, B, num_gop)
        # temporal_embedding_list = [einops.rearrange(temporal_embedding, '(b t) c h w -> b t c h w', t=num_gop) for temporal_embedding in temporal_embedding_list]
        multi_scale_temporal_feature_list_high_res = self.generate_multi_scale_temporal_features(temporal_embedding_list, p_features_mv_list, p_features_res_list, valid_indices, num_gop)

        vid_embeds = multi_scale_temporal_feature_list_high_res[-1][:, None, ...]
        vid_pad_mask = torch.zeros(
            [vid_embeds.shape[0], vid_embeds.shape[1], vid_embeds.shape[3], vid_embeds.shape[4]], dtype=torch.bool,
            device=vid_embeds.device)
        fusion_feat, text_memory, hs = self.fusion(vid_embeds, vid_pad_mask, word_embedding, attention_mask,
                                                   self.obj_queries.weight)
        # print(fusion_feat.shape)
        # fusion_feat, text_memory, hs = self.fusion(img_embedding, temporal_embeddings, word_embedding,
        #                                            attention_mask, self.obj_queries.weight)
        fusion_feat = fusion_feat[:, 0, ...]
        hs = einops.rearrange(hs[-1][0], 'b n c -> (b n) c')
        pred_kernel = self.project_kernel(hs)[..., None, None]
        pred_kernel = einops.rearrange(pred_kernel, '(b n) c h w -> n b c h w', n=self.nq)
        pred_logits = self.class_embed(hs)
        if self.nq > 1:
            pred_logits = einops.rearrange(pred_logits, '(b n) nc -> b n nc', n=self.nq)[:, None,:, :]
        else:
            pred_logits = pred_logits[:, None, None, :]
        # else:
        #     pred_logits = None
        #     fusion_feat, text_memory = self.fusion(img_embedding, temporal_embedding, word_embedding, attention_mask)
        #     pred_kernel = self.project_kernel(torch.mean(text_memory, dim=1))[..., None, None]

        if self.use_heat_map:
            # heat_maps = []
            # print(fusion_feat.shape)
            if self.use_abs_pos:
                imh, imw = fusion_feat.shape[-2:]
                x_range = torch.linspace(-1, 1, imw, device=fusion_feat.device)
                y_range = torch.linspace(-1, 1, imh, device=fusion_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([fusion_feat.shape[0], 1, -1, -1])
                x = x.expand([fusion_feat.shape[0], 1, -1, -1])
                fusion_feat = torch.cat([fusion_feat, x, y], dim=1)
            heat_maps = []
            for j in range(self.nq):
                heat_maps_per_query = []
                for i in range(fusion_feat.shape[0]):
                    heat_maps_per_query.append(F.conv2d(fusion_feat[i:i + 1], pred_kernel[j][i:i + 1]))
                heat_maps_per_query = torch.cat(heat_maps_per_query, dim=0) # b 1 h w
                heat_maps.append(heat_maps_per_query)
            heat_maps = torch.stack(heat_maps, dim=2) # b 1 nq h w
            # print(heat_maps.shape)

            # if self.training:
            with torch.no_grad():
                indices = self.matcher({'pred_masks': heat_maps, 'pred_logits': pred_logits},
                                       {'mask': sample['label'][:, None, ...], 'logits': sample['logits'], 'valid': sample['valid']})
            # else:
            # _, indices = torch.max(pred_logits.sigmoid()[:, 0, ...], dim=1)
                # print(indices.shape)

            heat_maps_selected = []

            for i in range(heat_maps.shape[0]):
                heat_maps_selected.append(heat_maps[i].index_select(1, indices[i]))
            heat_maps_selected = torch.cat(heat_maps_selected, dim=0)
            # print(heat_maps_selected.shape)
            heat_maps = heat_maps_selected
            # print(fusion_feat.shape)

            fusion_feat = torch.cat([fusion_feat, heat_maps], dim=1)
        # fusion_feat = self.aspp(fusion_feat)
        output = self.pixel_decoder(fusion_feat, multi_scale_temporal_feature_list_high_res[::-1])

        if self.training:
            loss_dict = {}
            mask = sample['label']
            output = F.interpolate(output, mask.shape[-2:], mode='bilinear', align_corners=True)
            # print(mask.max())
            if heat_maps is not None:
                mask_downsample = F.interpolate(mask.float()[:, None, ...], heat_maps.shape[-2:], mode='bilinear',
                                                align_corners=True)
                loss_bce_heat = F.binary_cross_entropy_with_logits(heat_maps.view(-1),
                                                                   mask_downsample.view(-1).float()) * 0.1
                loss_dict.update({'loss_bce_heat_map': loss_bce_heat})

            if pred_logits is not None:
                target_logits = torch.zeros(pred_logits.shape)
                # print(pred_logits.shape)
                # print(indices)
                # print(target_logits.shape)
                for i in range(target_logits.shape[0]):
                    target_logits[i][0][indices[i]] = torch.ones_like(target_logits[i][0][indices[i]])
                target_logits = target_logits.to(pred_logits.device).float()
                loss_bce_pred_logits = F.binary_cross_entropy_with_logits(pred_logits, target_logits)
                loss_dict.update({'loss_logits': loss_bce_pred_logits})

            loss_bce = F.binary_cross_entropy_with_logits(output.view(-1), mask.view(-1).float())
            loss_dice_overall = overall_dice_loss(output, mask.float())
            loss_dice_mean = mean_dice_loss(output, mask.float())
            if self.cfg.MODEL.LOSS.FOCAL:
                loss_focal = self.focal_loss(output, mask)
                loss_dict.update({'loss_focal': loss_focal})
            loss_dict.update(
                {'loss_bce': loss_bce, 'loss_dice_overall': loss_dice_overall, 'loss_dice_mean': loss_dice_mean})

            return loss_dict

        output = torch.sigmoid(output)
        output = F.interpolate(output, size=(320, 320), mode='bilinear', align_corners=True)
        return output