from torch.nn import functional as F
import torch
from torch import nn


def overall_dice_loss(inputs, targets, smooth=1.):
    inputs = F.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs_flatten = inputs.view(-1)
    targets_flatten = targets.view(-1)

    intersection = (inputs_flatten * targets_flatten).sum()
    overall_dice = (2. * intersection + smooth) / (inputs_flatten.sum() + targets_flatten.sum() + smooth)
    return 1 - overall_dice


def mean_dice_loss(inputs, targets, smooth=1.):
    b = inputs.shape[0]
    inputs = F.sigmoid(inputs)
    inputs_flatten = inputs.view(b, -1)
    targets_flatten = targets.view(b, -1)

    # dices = []
    # for i in range(b):
    #     intersection = (inputs_flatten[i] * targets_flatten[i]).sum()
    #     overall_dice = (2. * intersection + smooth) / (inputs_flatten[i].sum() + targets_flatten[i].sum() + smooth)
    #     dices.append(overall_dice)
    # mean_dice = sum(dices) / b
    intersection = torch.sum((inputs_flatten * targets_flatten), dim=1)
    union = torch.sum(inputs_flatten, dim=1) + torch.sum(targets_flatten, dim=1)
    mean_dice = torch.mean((2 * intersection + smooth) / (union + smooth))

    return 1 - mean_dice


class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=3, gamma=2, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        # if self.alpha is None:
        #     self.alpha = torch.ones(2)
        # elif isinstance(self.alpha, (list, np.ndarray)):
        #     self.alpha = np.asarray(self.alpha)
        #     self.alpha = np.reshape(self.alpha, (2))
        #     assert self.alpha.shape[0] == 2, \
        #         'the `alpha` shape is not match the number of class'
        # elif isinstance(self.alpha, (float, int)):
        #     self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        # else:
        #     raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss
