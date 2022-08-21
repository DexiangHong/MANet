import einops
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel
from modeling.video_backbone import CSN, video_swin_transformer
from modeling.deeplab.backbone import build_backbone
from modeling.layers.normalization import FrozenBatchNorm2d
from modeling.deeplab.aspp import build_aspp
from modeling.deeplab.decoder import build_decoder
from modeling.loss import overall_dice_loss, mean_dice_loss, BinaryFocalLoss
from modeling.layers.attention import AttentionFusion
from modeling.multimodal_transformer import MultimodalTransformer
from modeling.matcher import HungarianMatcher


class LSTM(nn.Module):
    def __init__(self, dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, batch_first=True)
        # nn.LSTM
    def forward(self, x):
        output, (_, _) = self.lstm(x)
        return output


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiModalFusion(nn.Module):
    def __init__(self, temporal_dim, bert_dim, image_dim, output_dim=512):
        super(MultiModalFusion, self).__init__()
        self.trans_image_embedding = nn.Sequential(nn.Conv2d(image_dim, output_dim//2, kernel_size=3, padding=1),
                                                   nn.GroupNorm(32, output_dim//2),
                                                   nn.ReLU(True),
                                                   )
        self.trans_temporal_embedding = nn.Sequential(nn.Conv2d(temporal_dim, temporal_dim//2, kernel_size=3, padding=1),
                                                   nn.GroupNorm(32, temporal_dim//2),
                                                   nn.ReLU(True),
                                                   nn.Conv2d(temporal_dim//2, output_dim//2, kernel_size=3, padding=1),
                                                   nn.GroupNorm(32, output_dim//2),
                                                   nn.ReLU(True))

        self.output_dim = output_dim

    def forward(self, image_embedding, temporal_embedding, word_embedding):
        out_image_feat = self.trans_image_embedding(image_embedding)
        imh, imw = out_image_feat.shape[-2:]

        out_temporal_feat = self.trans_temporal_embedding(temporal_embedding)
        # out_bert_feat = self.trans_bert_embedding(word_embedding)
        # print(out_bert_feat)
        out_bert_feat = word_embedding[..., None, None]
        out_bert_feat = out_bert_feat.repeat((1, 1, imh, imw))

        x_range = torch.linspace(-1, 1, imw, device=out_image_feat.device)
        y_range = torch.linspace(-1, 1, imh, device=out_image_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([out_image_feat.shape[0], 1, -1, -1])
        x = x.expand([out_image_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        fusion_feat = torch.cat([out_image_feat, out_temporal_feat, out_bert_feat, coord_feat], dim=1)

        return fusion_feat


class ChannelAttentionFusion(nn.Module):
    def __init__(self, temporal_dim, spatial_dim, bert_dim):
        super(ChannelAttentionFusion, self).__init__()
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.bert_dim = bert_dim

        self.video_reduce = nn.Sequential(nn.Conv2d(temporal_dim+2, temporal_dim//2, kernel_size=3, padding=1),
                                          nn.GroupNorm(32, temporal_dim//2),
                                          nn.ReLU(True),
                                          nn.Conv2d(temporal_dim//2, spatial_dim, kernel_size=3, padding=1),
                                          nn.GroupNorm(32, spatial_dim),
                                          nn.ReLU(True))
        self.spatial_reduce = nn.Sequential(nn.Conv2d(spatial_dim+2, spatial_dim, kernel_size=3, padding=1),
                                            nn.GroupNorm(32, spatial_dim),
                                            nn.ReLU(True))
        self.sentence_reduce = nn.Sequential(nn.Linear(bert_dim, spatial_dim * 2),
                                             nn.Sigmoid())

    def forward(self, spatial_feature, video_feature, bert_feature):
        b, _, imh, imw = video_feature.shape
        x_range = torch.linspace(-1, 1, imw, device=video_feature.device)
        y_range = torch.linspace(-1, 1, imh, device=video_feature.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)

        video_feature = torch.cat([video_feature, coord_feat], dim=1)
        spatial_feature = torch.cat([spatial_feature, coord_feat], dim=1)

        video_feature_reduce = self.video_reduce(video_feature)
        spatial_feature_reduce = self.spatial_reduce(spatial_feature)
        sentence_feature_reduce = self.sentence_reduce(bert_feature)

        spatial_temporal_feature = torch.cat([video_feature_reduce, spatial_feature_reduce], dim=1)
        text_attention_spatial_temporal_feature = spatial_temporal_feature * sentence_feature_reduce[..., None, None] + spatial_temporal_feature

        return text_attention_spatial_temporal_feature


class BaselineModel(nn.Module):
    def __init__(self, cfg, backbone='resnet', output_stride=16, freeze_bn=True, output_dim=512, training=True):
        super(BaselineModel, self).__init__()
        self.cfg = cfg
        # self.video_encoder = CSN(pretrain=True)
        self.video_encoder = video_swin_transformer(pretrain=True)
        self.txt_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.nq = cfg.MODEL.NQ
        self.use_abs_pos = cfg.MODEL.USE_ABS_POS
        self.use_heat_map = cfg.MODEL.HEAD.HEATMAP
        self.attention_use_decoder = cfg.MODEL.ATTENTION.USE_DECODER
        if self.attention_use_decoder:
            self.obj_queries = nn.Embedding(self.nq, 256)
        self.matcher = HungarianMatcher()

        in_dim = 2048 + 768

        if freeze_bn:
            print("Use frozen BN in DeepLab!")
            BatchNorm = FrozenBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        self.image_encoder = build_backbone(backbone, output_stride, BatchNorm, cfg.MODEL.BACKBONE.IMAGE.PRETRAINED_PATH)
        # self.image_encoder.load_state_dict(torch.load('./resnet.pth'))
        self.aspp = build_aspp('custom', output_stride, nn.BatchNorm2d, in_feature=512 + 2 + 768)
        if cfg.MODEL.FUSION_TYPE == 'concat':
            self.fusion = MultiModalFusion(2048, 768, 2048, output_dim)
            self.decoder = build_decoder('resnet', nn.BatchNorm2d, output_dim + 2 + 768)
        elif cfg.MODEL.FUSION_TYPE == 'attention':
            self.fusion = AttentionFusion(2048, 2048, 512, 768)
            self.decoder = build_decoder('resnet', nn.BatchNorm2d, 256)
        elif cfg.MODEL.FUSION_TYPE == 'channel_attention':
            self.fusion = ChannelAttentionFusion(2048, 256, 768)
            self.decoder = build_decoder('resnet', nn.BatchNorm2d, 512)
        elif cfg.MODEL.FUSION_TYPE == 'multimodaltransformer':
            self.fusion = MultimodalTransformer(in_channels=in_dim, nheads=8, dropout=0.1, d_model=256, dim_feedforward=2048, use_decoder=self.attention_use_decoder)
            # if self.use_heat_map:
            self.decoder = build_decoder('resnet', nn.BatchNorm2d, 256)
            # else:
            #     self.decoder = build_decoder('resnet', nn.BatchNorm2d, 256)
            if self.use_heat_map:
                # self.project_kernel = nn.Linear(256, 256)
                # self.project_kernel = MLP(256, 256, 256, 2)
                # self.aspp = build_aspp('custom', output_stride, nn.BatchNorm2d, in_feature=257)
                if self.use_abs_pos:
                    self.project_kernel = MLP(256, 256, 256 + 2, 2)
                    self.aspp = build_aspp('custom', 16, nn.BatchNorm2d, in_feature=259)
                else:
                    self.project_kernel = MLP(256, 256, 256, 2)
                    self.aspp = build_aspp('custom', 16, nn.BatchNorm2d, in_feature=257)
                self.class_embed = MLP(256, 256, 1, 2)

            else:
                self.aspp = build_aspp('custom', output_stride, nn.BatchNorm2d, in_feature=256)

        else:
            raise NotImplementedError

        self.fusion_type = cfg.MODEL.FUSION_TYPE

        self.training = training

        if cfg.MODEL.BACKBONE.TEXT.FREEZE:
            print('Freeze Bert weight')
            for param in self.txt_encoder.parameters():
                param.requires_grad = False

        if cfg.MODEL.BACKBONE.VIDEO.FREEZE:
            print('Freeze Bert weight')
            for param in self.video_encoder.parameters():
                param.requires_grad = False

        if cfg.MODEL.LOSS.FOCAL:
            self.focal_loss = BinaryFocalLoss()

    def forward(self, sample):
        heat_maps = None

        img_seq, key_frame, word_ids, attention_mask = sample['img_seq'], sample['key_frame'], sample['word_ids'], sample['attention_mask']

        img_embedding, low_level_feat = self.image_encoder(key_frame)
        # img_embedding = self.aspp(img_embedding)2

        b, c, h, w = img_embedding.shape

        temporal_embedding = self.video_encoder(img_seq)
        temporal_embedding = torch.mean(temporal_embedding, dim=1)
        temporal_embedding = F.interpolate(temporal_embedding, size=(h, w), mode='bilinear', align_corners=True)

        if self.fusion_type == 'attention':
            word_embedding = self.txt_encoder(word_ids, attention_mask)[0]
            word_embedding = einops.rearrange(word_embedding, 'b t c -> b c t')
        elif self.fusion_type == 'concat':
            word_embedding = self.txt_encoder(word_ids, attention_mask)[1]
        elif self.fusion_type == 'channel_attention':
            word_embedding = self.txt_encoder(word_ids, attention_mask)[1]
        elif self.fusion_type =='multimodaltransformer':
            word_embedding = self.txt_encoder(word_ids, attention_mask)[0]
        else:
            raise NotImplementedError
        pred_logits = None
        if self.fusion_type =='multimodaltransformer':
            if self.attention_use_decoder:
                vid_embeds = torch.cat([img_embedding, temporal_embedding], dim=1)
                vid_embeds = vid_embeds[:, None, ...]
                # print(vid_embeds.shape)
                vid_pad_mask = torch.zeros([vid_embeds.shape[0], vid_embeds.shape[1], vid_embeds.shape[3], vid_embeds.shape[4]], dtype=torch.bool, device=vid_embeds.device)
                fusion_feat, text_memory, hs = self.fusion(vid_embeds, vid_pad_mask, word_embedding, attention_mask, self.obj_queries.weight)

                # print(fusion_feat.shape)
                fusion_feat = fusion_feat[:, 0, ...]
                hs = einops.rearrange(hs[-1][0], 'b n c -> (b n) c')
                pred_kernel = self.project_kernel(hs)[..., None, None]
                pred_kernel = einops.rearrange(pred_kernel, '(b n) c h w -> n b c h w', n=self.nq)
                pred_logits = self.class_embed(hs)
                if self.nq > 1:
                    pred_logits = einops.rearrange(pred_logits, '(b n) nc -> b n nc', n=self.nq)[:, None, :, :]
                else:
                    pred_logits = pred_logits[:, None, None, :]
            else:

                vid_embeds = torch.cat([img_embedding, temporal_embedding], dim=1)
                vid_embeds = vid_embeds[:, None, ...]
                vid_pad_mask = torch.zeros([vid_embeds.shape[0], vid_embeds.shape[1], vid_embeds.shape[3], vid_embeds.shape[4]], dtype=torch.bool, device=vid_embeds.device)
                fusion_feat, text_memory = self.fusion(vid_embeds, vid_pad_mask, word_embedding, attention_mask)
                pred_kernel = self.project_kernel(torch.mean(text_memory, dim=1))[..., None, None]

            if self.use_heat_map:
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
                    heat_maps_per_query = torch.cat(heat_maps_per_query, dim=0)  # b 1 h w
                    heat_maps.append(heat_maps_per_query)
                heat_maps = torch.stack(heat_maps, dim=2)  # b 1 nq h w
                with torch.no_grad():
                    indices = self.matcher({'pred_masks': heat_maps, 'pred_logits': pred_logits},
                                           {'mask': sample['label'][:, None, ...], 'logits': sample['logits'],
                                            'valid': sample['valid']})

                heat_maps_selected = []

                for i in range(heat_maps.shape[0]):
                    heat_maps_selected.append(heat_maps[i].index_select(1, indices[i]))
                heat_maps_selected = torch.cat(heat_maps_selected, dim=0)
                # print(heat_maps_selected.shape)
                heat_maps = heat_maps_selected
                # print(fusion_feat.shape)

                fusion_feat = torch.cat([fusion_feat, heat_maps], dim=1)

        else:
            fusion_feat = self.fusion(img_embedding, temporal_embedding, word_embedding)
        fusion_feat = self.aspp(fusion_feat)
        output = self.decoder(fusion_feat, low_level_feat)

        if self.training:
            loss_dict = {}
            mask = sample['label']
            output = F.interpolate(output, mask.shape[-2:], mode='bilinear', align_corners=True)
            # print(mask.max())
            if heat_maps is not None:
                mask_downsample = F.interpolate(mask.float()[:, None, ...], heat_maps.shape[-2:], mode='bilinear',
                                                align_corners=True)
                loss_bce_heat = F.binary_cross_entropy_with_logits(heat_maps.view(-1), mask_downsample.view(-1).float()) * 0.1
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
            loss_dict.update({'loss_bce': loss_bce, 'loss_dice_overall': loss_dice_overall, 'loss_dice_mean': loss_dice_mean})

            return loss_dict

        output = torch.sigmoid(output)

        return output


if __name__ == '__main__':
    from transformers import BertTokenizer
    from modeling import cfg

    cfg.MODEL.BACKBONE.IMAGE.PRETRAINED_PATH = './resnet.pth'
    cfg.MODEL.ATTENTION.USE_DECODER = True
    cfg.MODEL.HEAD.HEATMAP = True
    cfg.MODEL.FUSION_TYPE = 'multimodaltransformer'
    model = BaselineModel(cfg, training=False)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    img_seq = torch.randn([2, 32, 3, 320, 320])
    key_frame = torch.randn([2, 3, 320, 320])
    sentence = 'The words'
    encoder_dict = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=20,
                                              pad_to_max_length=True, return_attention_mask=True,
                                              return_tensors='pt')
    word_ids = encoder_dict['input_ids']
    attention_mask = encoder_dict['attention_mask']
    word_ids = torch.cat([word_ids, word_ids], dim=0)
    attention_maks = torch.cat([attention_mask, attention_mask], dim=0)

    output = model({'img_seq':img_seq, 'key_frame': key_frame, 'word_ids': word_ids, 'attention_mask': attention_maks})
    print(output.shape)




