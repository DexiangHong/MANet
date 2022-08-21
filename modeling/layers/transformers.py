import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear, Dropout, LayerNorm
from torch import Tensor
from typing import Optional
import einops
import copy
import math
from modeling.layers.attention import MultiHeadAttention

# from torch.nn import TransformerEncoderLayer


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=True):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.batch_first = batch_first

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, query: Tensor, key: Tensor, value: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # if self.batch_first:
        #     src = einops.rearrange(src, 'b t c -> t b c')
        #     query = einops.rearrange(query, 'b t c -> t b c')
        #     key = einops.rearrange(key, 'b t c -> t b c')
        #     value = einops.rearrange(value, 'b t c -> t b c')
        assert torch.isnan(src).sum() == 0, print('src come nan')
        assert torch.isnan(query).sum() == 0, print('query come nan')
        assert torch.isnan(key).sum() == 0, print('key come nan')
        assert torch.isnan(value).sum() == 0, print('value come nan')

        src2 = self.self_attn(query, key, value)[0]

        if torch.isnan(src2).sum() != 0:
            print('caught error')
            torch.save(src, './src.pth', )
            torch.save(query, './query.pth')
            torch.save(key, './key.pth')
            torch.save(value, './value.pth')
            torch.save(src2, './scr2.pth')
            torch.save(self.self_attn.state_dict(), 'attn.pth')
        assert torch.isnan(src2).sum() == 0, print('src2 come nan')
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        # if self.batch_first:
        #     src = einops.rearrange(src, 't b c -> b t c')
        return src