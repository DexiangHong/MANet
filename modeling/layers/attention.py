import torch
import torch.nn.functional as F
import math
from torch import nn
import numpy as np
import einops
from modeling.layers.coord import generate_coord_feature


class IA_gate(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(IA_gate, self).__init__()
        self.IA = nn.Linear(in_dim, out_dim)

    def forward(self, x, IA_head):
        a = self.IA(IA_head)
        a = 1. + torch.tanh(a)
        a = a.unsqueeze(-1).unsqueeze(-1)
        x = a * x
        return x

def calculate_attention_head(ref_embedding, ref_label, prev_embedding, prev_label, epsilon=1e-5):

    ref_head = ref_embedding * ref_label
    ref_head_pos = torch.sum(ref_head, dim=(2,3))
    ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
    ref_pos_num = torch.sum(ref_label, dim=(2,3))
    ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
    ref_head_pos = ref_head_pos / (ref_pos_num + epsilon)
    ref_head_neg = ref_head_neg / (ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)

    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)

    return total_head


def calculate_attention_head_for_eval(ref_embeddings, ref_labels, prev_embedding, prev_label, epsilon=1e-5):
    total_ref_head_pos = 0.
    total_ref_head_neg = 0.
    total_ref_pos_num = 0.
    total_ref_neg_num = 0.

    for idx in range(len(ref_embeddings)):
        ref_embedding = ref_embeddings[idx]
        ref_label = ref_labels[idx]
        ref_head = ref_embedding * ref_label
        ref_head_pos = torch.sum(ref_head, dim=(2,3))
        ref_head_neg = torch.sum(ref_embedding, dim=(2,3)) - ref_head_pos
        ref_pos_num = torch.sum(ref_label, dim=(2,3))
        ref_neg_num = torch.sum(1. - ref_label, dim=(2,3))
        total_ref_head_pos = total_ref_head_pos + ref_head_pos
        total_ref_head_neg = total_ref_head_neg + ref_head_neg
        total_ref_pos_num = total_ref_pos_num + ref_pos_num
        total_ref_neg_num = total_ref_neg_num + ref_neg_num
    ref_head_pos = total_ref_head_pos / (total_ref_pos_num + epsilon)
    ref_head_neg = total_ref_head_neg / (total_ref_neg_num + epsilon)

    prev_head = prev_embedding * prev_label
    prev_head_pos = torch.sum(prev_head, dim=(2,3))
    prev_head_neg = torch.sum(prev_embedding, dim=(2,3)) - prev_head_pos
    prev_pos_num = torch.sum(prev_label, dim=(2,3))
    prev_neg_num = torch.sum(1. - prev_label, dim=(2,3))
    prev_head_pos = prev_head_pos / (prev_pos_num + epsilon)
    prev_head_neg = prev_head_neg / (prev_neg_num + epsilon)

    total_head = torch.cat([ref_head_pos, ref_head_neg, prev_head_pos, prev_head_neg], dim=1)
    return total_head


class AttentionFusion(nn.Module):
    def __init__(self, video_feat_dim, spatial_feat_dim, reduced_video_feat_dim, dim_semantic, use_coord_feat=True):
        super(AttentionFusion, self).__init__()
        # self.opt = opt
        visual_feat_dim = reduced_video_feat_dim
        if use_coord_feat:
            visual_feat_dim += 2

        self.use_coord_feat = use_coord_feat

        self.reduce_video_feat = nn.Sequential(
            nn.Conv2d(video_feat_dim+spatial_feat_dim, video_feat_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(32, video_feat_dim // 2),
            nn.ReLU(True),
            nn.Conv2d(video_feat_dim // 2, reduced_video_feat_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, reduced_video_feat_dim),
            nn.ReLU(True))

        self.video_reduce = nn.Linear(in_features=visual_feat_dim, out_features=dim_semantic)
        self.video_linearK = nn.Linear(in_features=visual_feat_dim, out_features=visual_feat_dim)
        self.video_linearQ = nn.Linear(in_features=visual_feat_dim, out_features=visual_feat_dim)
        self.video_linearV = nn.Linear(in_features=visual_feat_dim, out_features=visual_feat_dim)

        # self.txt_pool = nn.MaxPool1d(kernel_size=self.opt.sentence_length)
        self.txt_increase = nn.Linear(in_features=dim_semantic, out_features=visual_feat_dim)
        self.txt_pool = nn.MaxPool1d(kernel_size=20)

        self.visual_feat_dim = visual_feat_dim
        self.dim_semantic = dim_semantic

    def forward(self, spatial_feat,video_feat, txt):
        # video (N, 832, 32, 32), spatial (N, 8, 32, 32), txt (N, 300, 20)
        # (N, 832+8, 32, 32) -> (N, 32, 32, 832+8)
        # print(video_feat.shape)
        # print(spatial_feat.shape)
        # print(txt.shape)

        video_spatial_org = torch.cat([video_feat, spatial_feat], dim=1)
        video_spatial_org = self.reduce_video_feat(video_spatial_org)

        b, c, h, w = video_spatial_org.shape
        if self.use_coord_feat:
            x_range = torch.linspace(-1, 1, w, device=video_spatial_org.device)
            y_range = torch.linspace(-1, 1, h, device=video_spatial_org.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([b, 1, -1, -1])
            x = x.expand([b, 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            video_spatial_org = torch.cat([video_spatial_org, coord_feat], dim=1)

        video_spatial_org = video_spatial_org.permute(0, 2, 3, 1).contiguous()

        # (N, 32, 32, 832+8) -> (N*32*32, 832+8) -> (N*32*32, 300)
        video_spatial_reduce = self.video_reduce(video_spatial_org.view(-1, self.visual_feat_dim))
        # (N * 32 * 32, 300) -> (N, 32*32, 300)
        video_spatial_reduce = video_spatial_reduce.view(-1, h*w, self.dim_semantic)
        # (N, 32*32, 300) * (N, 300, 20) -> (N, 32*32, 20)
        txt_score = F.softmax(torch.bmm(video_spatial_reduce, txt) / np.power(self.dim_semantic, 0.5), dim=-1)
        # (N, 32*32, 20) * (N, 20, 300) -> (N, 32*32, 300) -> (N, 32, 32, 300)
        weighted_txt = torch.bmm(txt_score, txt.permute(0, 2, 1).contiguous()).view(-1, h, w, self.dim_semantic)
        # (N, 300, 32, 32)
        weighted_txt = weighted_txt.permute(0, 3, 1, 2).contiguous()

        # (N, 300, 20) -> (N, 300) -> (N, 832+8) -> (N, 32*32, 832+8)
        txt_repeat = self.txt_increase(self.txt_pool(txt).squeeze(-1)).unsqueeze(1).repeat(1, h*w, 1)
        # (N, 32*32, 832+8) -> (N, 32*32, 832+8)
        video_key = self.video_linearK(video_spatial_org.view(-1,self.visual_feat_dim)).view(-1, h*w, self.visual_feat_dim) * txt_repeat
        video_query = self.video_linearQ(video_spatial_org.view(-1, self.visual_feat_dim)).view(-1, h*w, self.visual_feat_dim) * txt_repeat
        video_value = self.video_linearV(video_spatial_org.view(-1, self.visual_feat_dim)).view(-1, h*w, self.visual_feat_dim)
        # (N, 32*32, 832+8) * (N, 832+8, 32*32) -> (N, 32*32, 32*32)
        video_score = F.softmax(torch.bmm(video_key, video_query.permute(0, 2, 1).contiguous()) / np.power(self.visual_feat_dim, 0.5), dim=-1)
        # (N, 32*32, 32*32) * (N, 32*32, 832+8) -> (N, 32*32, 832+8)
        weighted_video = torch.bmm(video_score, video_value).view(-1, h, w, self.visual_feat_dim)
        # (N, 832+8, 32, 32)
        weighted_video = weighted_video.permute(0, 3, 1, 2).contiguous()

        multi_modality_feat = torch.cat([weighted_video, weighted_txt], dim=1)

        return multi_modality_feat


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, visual_feature_dim, sentence_feature_dim, sentence_length, hidden_dim):
        super(CrossModalAttentionFusion, self).__init__()
        self.visual_feature_dim = visual_feature_dim
        self.sentence_feature_dim = sentence_feature_dim
        self.sentence_length = sentence_length

        self.visual_reduce = nn.Sequential(nn.Conv2d(self.visual_feature_dim+2, hidden_dim, kernel_size=3, padding=1),
                                           nn.GroupNorm(32, hidden_dim),
                                           nn.ReLU())
        self.sentence_reduce = nn.LSTM(input_size=sentence_feature_dim, hidden_size=hidden_dim//2, num_layers=1,
                                       batch_first=True, bidirectional=True)

        self.sentence_attn_reduce = nn.Sequential(nn.Linear(sentence_feature_dim, visual_feature_dim),
                                                  nn.Sigmoid())

        self.softmax = nn.Softmax(dim=1)

    def forward(self, visual_feature_org, txt_feature):
        # visual : b c h w, txt: b t c
        # print(visual_feature.shape)
        coord_feat = generate_coord_feature(visual_feature_org.shape[0], visual_feature_org.shape[2], visual_feature_org.shape[3], visual_feature_org.device)
        # print(coord_feat.shape, visual_feature.shape)
        visual_feature = torch.cat([visual_feature_org, coord_feat], dim=1)
        visual_feature_reduce = self.visual_reduce(visual_feature)
        visual_feature_reduce = F.normalize(visual_feature_reduce, p=2)
        visual_feature_reduce = einops.rearrange(visual_feature_reduce, 'b c h w -> b c (h w)')
        sentence_feature_reduce, (_, _) = self.sentence_reduce(txt_feature)
        sentence_feature_reduce = F.normalize(sentence_feature_reduce, p=2, dim=2)

        # print(sentence_feature_reduce)
        A = torch.bmm(sentence_feature_reduce, visual_feature_reduce) # b t hw
        w = self.softmax(torch.sum(A, dim=2))[..., None] # b t 1
        txt_feature_attn = torch.sum(w * txt_feature, dim=1)
        txt_feature_attn_reduce = self.sentence_attn_reduce(txt_feature_attn)

        visual_feature_attn = visual_feature_org + visual_feature_org * txt_feature_attn_reduce[..., None, None]
        return visual_feature_attn


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
            q: Queries张量，形状为[B, L_q, D_q]
            k: Keys张量，形状为[B, L_k, D_k]
            v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
            scale: 缩放因子，一个浮点标量
            attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
            上下文张量和attetention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        assert torch.isnan(attention).sum() == 0, print('before attention nan')
        if scale:
            attention = attention * scale
        assert torch.isnan(torch.tensor(scale)).sum() == 0, print('scale nan')
        if attn_mask:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)
        assert torch.isnan(attention).sum() == 0, print('attention softmax nan')
        # 添加dropout
        attention = self.dropout(attention)
        # assert torch.isnan(attention).sum() == 0, print('attention nan')
        assert torch.isnan(v).sum() == 0, print('v nan')
        # 和V做点积
        context = torch.bmm(attention, v)
        assert torch.isnan(context).sum() == 0, print('context nan')

        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        # if attn_mask:
        #     attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale, attn_mask)
        # assert torch.isnan(context).sum == 0, print('context cause nan')
        assert torch.isnan(context).sum() == 0, print('context come nan')


        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)
        assert torch.isnan(output).sum() == 0, print('output before layer norm')
        # add residual and norm layer
        output = self.layer_norm(residual + output)
        assert torch.isnan(output).sum() == 0, print('output after layer norm')

        return output, attention



if __name__ == '__main__':
    model = AttentionFusion(2048, 256, 256, 768)
    spatial_feat = torch.randn([2, 256, 10, 10])
    video_feat = torch.randn([2, 2048, 10, 10])
    word_feat = torch.randn([2, 768, 20])

    output = model(video_feat, spatial_feat, word_feat)
    print(output.shape)