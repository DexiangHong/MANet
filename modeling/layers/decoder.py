import torch
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):
    def __init__(self, num_stages, spatial_feature_dims, video_feature_dims, sentence_feat_dim):
        super(Decoder, self).__init__()
        self.num_stages = num_stages
        self.hidden_dims = [1024//2**i for i in range(num_stages)]
        print(self.hidden_dims)
        self.linear_spatial_modules = nn.ModuleList([nn.Sequential(nn.Linear(sentence_feat_dim, spatial_feature_dims[i]), nn.Softmax(dim=1)) for i in range(num_stages)])
        self.linear_video_modules = nn.ModuleList([nn.Sequential(nn.Linear(sentence_feat_dim, video_feature_dims[i]), nn.Softmax(dim=1)) for i in range(num_stages)])

        self.trans_spatial_modules = nn.ModuleList([nn.Sequential(
            nn.Conv2d(spatial_feature_dims[i], self.hidden_dims[i], kernel_size=3, padding=1),
            nn.GroupNorm(32, self.hidden_dims[i]), nn.ReLU(True)) for i in range(num_stages)])
        self.trans_video_modules = nn.ModuleList([nn.Sequential(
            nn.Conv2d(video_feature_dims[i], self.hidden_dims[i], kernel_size=3, padding=1),
            nn.GroupNorm(32, self.hidden_dims[i]), nn.ReLU(True)) for i in range(num_stages)])

        self.upsample_modules = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i + 1], kernel_size=3, padding=1),
            nn.GroupNorm(32, self.hidden_dims[i + 1]), nn.ReLU(True)) for i in range(num_stages-1)])

        self.predict = nn.Conv2d(self.hidden_dims[-1], 1, kernel_size=1)

    def forward(self, spatial_features, video_features, sentence_feature):
        decode_feature_stage = None
        for i in range(self.num_stages):
            sentence_spatial_attn = self.linear_spatial_modules[i](sentence_feature)[..., None, None]
            sentence_video_attn = self.linear_video_modules[i](sentence_feature)[..., None, None]
            # print(sentence_video_attn.shape, video_features[i].shape, sentence_spatial_attn.shape, spatial_features[i].shape)
            spatial_video_feature_fusion = self.trans_video_modules[i](sentence_video_attn * video_features[i]) + \
                                           self.trans_spatial_modules[i](sentence_spatial_attn * spatial_features[i])

            if decode_feature_stage is None:
                decode_feature_stage = spatial_video_feature_fusion
            else:
                decode_feature_stage = self.upsample_modules[i-1](decode_feature_stage)
                decode_feature_stage = F.interpolate(decode_feature_stage, scale_factor=2, mode='bilinear', align_corners=True)
                decode_feature_stage = decode_feature_stage + spatial_video_feature_fusion

        predict = self.predict(decode_feature_stage)

        return predict


def build_decoder(num_stages, spatial_feature_dims, video_feature_dims, sentence_feat_dim):
    return Decoder(num_stages, spatial_feature_dims, video_feature_dims, sentence_feat_dim)


# if __name__ == '__main__':
#     model = Decoder(5, [2048, 1024, 512, 256, 64], [1024, 832, 480, 192, 64], 768)
#     spatial_features = list(reversed([torch.randn([2, 64, 160, 160]), torch.randn([2, 256, 80, 80]), torch.randn([2, 512, 40, 40]), torch.randn([2, 1024, 20, 20]), torch.randn([2, 2048, 10, 10])]))
#     video_features = list(reversed([torch.randn([2, 64, 160, 160]), torch.randn([2, 192, 80, 80]), torch.randn([2, 480, 40, 40]),
#                         torch.randn([2, 832, 20, 20]), torch.randn([2, 1024, 10, 10])]))
#
#     sentence_feature = torch.randn([2, 768])
#     out = model(spatial_features, video_features, sentence_feature)
#     print(out.shape)





