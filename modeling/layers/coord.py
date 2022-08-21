import torch


def generate_coord_feature(batch, imh, imw, device):
    x_range = torch.linspace(-1, 1, imw, device=device)
    y_range = torch.linspace(-1, 1, imh, device=device)
    y, x = torch.meshgrid(y_range, x_range)
    y = y.expand([batch, 1, -1, -1])
    x = x.expand([batch, 1, -1, -1])
    coord_feat = torch.cat([x, y], 1)
    return coord_feat
