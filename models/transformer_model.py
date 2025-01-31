import math
from functools import partial

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from models.layers import Xtoy, Etoy, masked_softmax

from models.general_models import NodeBlock, EdgeBlock

from diffusers.models.cross_attention import CrossAttention


def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly., {}'.format((variable * (1 - node_mask.long())).abs().max().item())


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, t_dim: int):
        super().__init__()
        self.gelu = nn.GELU()
        self.linear = nn.Linear(t_dim, dim*2)
        self.layernorm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x: Tensor, t_emb: Tensor):
        emb: Tensor = self.linear(self.gelu(t_emb)).unsqueeze(1)
        while emb.dim() < x.dim():
            emb = emb.unsqueeze(1)

        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.layernorm(x) * (1. + scale) + shift

        return x


class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dt: int = 2048,
                 use_y_feat: bool = False,
                 use_source_condition: bool = False, source_dim: int = 128,
                 use_source_condition_edge: bool = False, source_edge_dim: int = 128,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.0,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None,
                 cross_attn=False, cross_attn_dim: int = 768) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, use_y_feat, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

        self.use_cross_attn = cross_attn
        if self.use_cross_attn:
            self.norm_cross_x = AdaLayerNorm(dx, dt)
            self.cross_attn = CrossAttention(
                                query_dim=dx,
                                cross_attention_dim=cross_attn_dim,
                                heads=n_head,
                                dim_head=dx//n_head, 
                                dropout=0.0, 
                                bias=False,  # attention_bias
                                upcast_attention=False,
                            )

        # source condition
        self.use_source_condition = use_source_condition
        if self.use_source_condition:
            self.x_source_norm = NodeBlock(dx, dx, time_emb_dim=source_dim)

        self.use_source_condition_edge = use_source_condition_edge
        self.source_edge_dim = source_edge_dim
        if self.use_source_condition_edge:
            self.e_source_norm = EdgeBlock(de, de, time_emb_dim=source_edge_dim)

        # time embedding norm
        self.x_t_norm_1 = NodeBlock(dx, dx, time_emb_dim=dt)
        self.x_t_norm_2 = NodeBlock(dx, dx, time_emb_dim=dt)
        
        self.e_t_norm_1 = EdgeBlock(de, de, time_emb_dim=dt)
        self.e_t_norm_2 = EdgeBlock(de, de, time_emb_dim=dt)


    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor=None, t_embed=None, condition_feat: Tensor = None, source_condition: Tensor = None, source_condition_edge: Tensor = None, res_X: Tensor = None, return_res: bool = False):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            t_embed: time embedding
            obj_num_embed: obj_num embedding
            condition_feat: textual feature for cross attention
            source_condition: source room condition on the node
            res_X: (bs, n, c), optional
            Output: newX, newE, new_y with the same shape.
        """

        if return_res:
            h = []

        # source condition
        if self.use_source_condition:
            X_norm = self.x_source_norm(X, source_condition)
        else:
            X_norm = X

        b, n, _, c = E.shape
        if self.use_source_condition_edge:
            source_condition_edge = source_condition_edge.view(b, -1, self.source_edge_dim)
            E_norm = self.e_source_norm(E, source_condition_edge)  
        else:
            E_norm = E

        # cross attention here
        if self.use_cross_attn:
            X_norm = self.norm_cross_x(X_norm, t_embed)
            attn_output = self.cross_attn(X_norm, condition_feat)
            if node_mask is not None:
                x_mask = node_mask.unsqueeze(-1)
                attn_output = attn_output * x_mask
            X_norm = X_norm + attn_output

        if res_X is not None:
            X_norm = torch.cat((X_norm, res_X[0]), dim=2)
        X_norm = self.x_t_norm_1(X_norm, t_embed)
        E_norm = self.e_t_norm_1(E_norm, t_embed)
        if return_res:
            h.append(X_norm)

        # gnn
        newX, newE, new_y = self.self_attn(X_norm, E_norm, y, node_mask=node_mask)  # (b,n,c), (b,n,n,c), (b,c)
        X = self.normX1(newX) + X
        E = self.normE1(newE) + E

        # FFN
        if res_X is not None:
            X_norm = self.x_t_norm_2(torch.cat((X, res_X[1]), dim=2), t_embed)
        else:
            X_norm = self.x_t_norm_2(X, t_embed)
        if return_res:
            h.append(X_norm)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X_norm))))
        X = self.normX2(ff_outputX) + X

        E_norm = self.e_t_norm_2(E, t_embed)
        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E_norm))))
        E = self.normE2(ff_outputE) + E

        if return_res:
            return X, E, y, h
        else:
            return X, E, y


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, use_y_feat, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

        self.use_y_feat = use_y_feat

        if self.use_y_feat:
            # FiLM y to E
            self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
            self.y_e_add = Linear(dy, dx)

            # FiLM y to X
            self.y_x_mul = Linear(dy, dx)
            self.y_x_add = Linear(dy, dx)

            # Process y
            self.y_y = Linear(dy, dy)
            self.x_y = Xtoy(dx, dy)
            self.e_y = Etoy(de, dy)


    def forward(self, X, E, y, node_mask=None):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        if node_mask is not None:
            x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
            e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
            e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        if node_mask is not None:
            Q = self.q(X) * x_mask           # (bs, n, dx)
            K = self.k(X) * x_mask           # (bs, n, dx)
            assert_correctly_masked(Q, x_mask)
        else:
            Q = self.q(X)      # (bs, n, dx)
            K = self.k(X)      # (bs, n, dx)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))

        if node_mask is not None:
            E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
            E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        else:
            E1 = self.e_mul(E)                   # bs, n, n, dx
            E2 = self.e_add(E)                   # bs, n, n, dx

        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx

        if self.use_y_feat:  # false
            ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
            ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
            newE = ye1 + (ye2 + 1) * newE

        # Output E
        if node_mask is not None:
            newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        else:
            newE = self.e_out(newE)      # bs, n, n, de

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        if node_mask is not None:
            softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
            attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head
        else:
            attn = torch.softmax(Y, dim=2)

        if node_mask is not None:
            V = self.v(X) * x_mask                        # bs, n, dx
        else:
            V = self.v(X)
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx
        
        # Incorporate y to X
        newX = weighted_V
        if self.use_y_feat:  # false
            yx1 = self.y_x_add(y).unsqueeze(1)
            yx2 = self.y_x_mul(y).unsqueeze(1)
            newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        if node_mask is not None:
            newX = self.x_out(newX) * x_mask
        else:
            newX = self.x_out(newX)

        return newX, newE, y
