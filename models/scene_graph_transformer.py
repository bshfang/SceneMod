import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce
from models.transformer_model import XEyTransformerLayer
from models.general_models import ResnetBlock

from utils import utils


def exists(x):
    return x is not None


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
        

def encoder_mlp(input_size, hidden_size):
    mlp_layers = [
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size*2),
            nn.GELU(),
            nn.Linear(hidden_size*2, hidden_size),
        ]
    return nn.Sequential(*mlp_layers)


def decoder_mlp(hidden_size, output_size):
    mlp_layers = [
        nn.Linear(hidden_size, hidden_size*2),
        nn.GELU(),
        nn.Linear(hidden_size*2, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, output_size),
    ]
    return nn.Sequential(*mlp_layers)


class SceneGraphTransformer(nn.Module):
    def __init__(self, cfg):
        super(SceneGraphTransformer, self).__init__()

        self.cfg = cfg

        model_cfg = cfg['model']
        self.model_cfg = model_cfg

        self.separate_encoder_node = model_cfg['separate_encoder_node']
        self.separate_encoder_edge = model_cfg['separate_encoder_edge']
        self.separate_encoder_edge_attr = model_cfg.get('separate_encoder_edge', False)

        self.use_y_feat = model_cfg.get('use_y_feat', False)
        
        self.use_node_mask = model_cfg.get('use_node_mask', True)

        self.n_layers = model_cfg['n_layers']

        input_dims = model_cfg['input_dims']
        output_dims = model_cfg['output_dims']
        hidden_dims = model_cfg['hidden_dims']

        x_dim = model_cfg['hidden_dims']['dx']  # node feature dim
        e_dim = model_cfg['hidden_dims']['de']  # edge feature dim
        y_dim = model_cfg['hidden_dims']['dy']

        self.x_dim = x_dim
        self.e_dim = e_dim
        self.y_dim = y_dim

        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']
        
        # load from the dataset
        feat_dims = model_cfg.get('feat_dims', {})
        self.translation_dim = feat_dims.get("translation_dim", 3)
        self.distance_dim = feat_dims.get("distance_dim", 0)
        self.size_dim = feat_dims.get("size_dim", 3)
        self.rotation_dim = feat_dims.get("rotation_dim", 2)
        self.cat_dim = feat_dims.get("cat_dim", 23)
        self.objfeat_dim = feat_dims.get("objfeat_dim", 32)
        self.bbox_dim = self.translation_dim + self.distance_dim + self.size_dim + self.rotation_dim

        self.mod_mask_dim = feat_dims.get("mod_mask_dim", 0)

        self.edge_cat_dim = feat_dims.get("edge_cat_dim", 11)
        self.edge_attr_dim = feat_dims.get("edge_attr_dim", 12)
        if self.separate_encoder_edge_attr:
            self.edge_attr_dim_dis = feat_dims.get("edge_attr_dim_dis", 3)
            self.edge_attr_dim_angle = feat_dims.get("edge_attr_dim_angle", 2)
            self.edge_attr_dim_angle_class = feat_dims.get("edge_attr_dim_angle_class", 4)
            self.edge_attr_dim_parallel = feat_dims.get("edge_attr_dim_parallel", 3)

        self.edge_dim = self.edge_cat_dim + self.edge_attr_dim

        # time embeddings
        self.use_time_embedding = True 
        if self.use_time_embedding:
            time_dim = x_dim * 4
            sinu_pos_emb = SinusoidalPosEmb(x_dim)
            fourier_dim = x_dim
            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )

        # source condition
        self.instance_condition = self.model_cfg.get("instance_condition", False)
        self.source_room_condition_node = self.model_cfg.get("source_room_condition_node", False)
        self.source_room_condition_edge = self.model_cfg.get("source_room_condition_edge", False)
        self.source_room_condition_node_s2 = self.model_cfg.get("source_room_condition_node_s2", False)
        self.source_room_condition_edge_s2 = self.model_cfg.get("source_room_condition_edge_s2", False)
        
        self.source_emb_dim = self.model_cfg.get("condition_emb_dim", 0)  # source_emb_dim
        self.source_edge_emb_dim = self.model_cfg.get("source_edge_emb_dim", 0)

        self.use_source_condition = self.instance_condition or self.source_room_condition_node or self.source_room_condition_node_s2
        self.use_source_condition_edge = self.source_room_condition_edge or self.source_room_condition_edge_s2

        # position embeddings
        self.use_instance_embedding = self.model_cfg.get("instance_embedding", False)
        self.edge_instance_embedding = self.model_cfg.get("edge_instance_embedding", True)

        self.gnn_residual_feature = self.model_cfg.get("gnn_residual_feature", False)
        
        if self.gnn_residual_feature:
            self.final_res_block_node = ResnetBlock(x_dim*2, x_dim, time_emb_dim=time_dim)

        # encoder and decoder
        # node feature encoder and decoder
        if self.separate_encoder_node:
            # encoder
            if self.translation_dim > 0:
                self.bbox_trans_encoder = encoder_mlp(self.translation_dim+self.distance_dim, x_dim)
                self.bbox_trans_decoder = decoder_mlp(x_dim, self.translation_dim+self.distance_dim)
            if self.size_dim > 0:
                self.bbox_size_encoder = encoder_mlp(self.size_dim, x_dim)
                self.bbox_size_decoder = decoder_mlp(x_dim, self.size_dim)
            if self.rotation_dim > 0:
                self.bbox_rot_encoder = encoder_mlp(self.rotation_dim, x_dim)
                self.bbox_rot_decoder = decoder_mlp(x_dim, self.rotation_dim)
            if self.cat_dim > 0:
                self.class_encoder = encoder_mlp(self.cat_dim, x_dim)
                self.class_decoder = decoder_mlp(x_dim, self.cat_dim)
            if self.objfeat_dim > 0:
                self.objfeat_encoder = encoder_mlp(self.objfeat_dim, x_dim)
                self.objfeat_decoder = decoder_mlp(x_dim, self.objfeat_dim)

            if self.mod_mask_dim > 0:
                self.mod_mask_encoder = encoder_mlp(self.mod_mask_dim, x_dim)
                self.mod_mask_decoder = decoder_mlp(x_dim, self.mod_mask_dim)

        else:
            self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], x_dim*2), nn.GELU(),
                                      nn.Linear(x_dim*2, x_dim), nn.GELU())
            self.mlp_out_X = nn.Sequential(nn.Linear(x_dim, x_dim*2), nn.GELU(),
                                       nn.Linear(x_dim*2, output_dims['X']))
        self.init_node = nn.Linear(x_dim, x_dim)
            
        # edge encoder and decoder
        if self.separate_encoder_edge:
            if self.edge_cat_dim > 0:
                self.edge_cat_encoder = encoder_mlp(self.edge_cat_dim, e_dim)
                self.edge_cat_decoder = decoder_mlp(e_dim, self.edge_cat_dim)

            if self.separate_encoder_edge_attr:
                if self.edge_attr_dim_dis > 0:
                    self.edge_attr_1_encoder = encoder_mlp(self.edge_attr_dim_dis, e_dim)
                    self.edge_attr_1_decoder = decoder_mlp(e_dim, self.edge_attr_dim_dis)
                if self.edge_attr_dim_angle > 0:
                    self.edge_attr_2_encoder = encoder_mlp(self.edge_attr_dim_angle, e_dim)
                    self.edge_attr_2_decoder = decoder_mlp(e_dim, self.edge_attr_dim_angle)
                if self.edge_attr_dim_angle_class > 0:
                    self.edge_attr_3_encoder = encoder_mlp(self.edge_attr_dim_angle_class, e_dim)
                    self.edge_attr_3_decoder = decoder_mlp(e_dim, self.edge_attr_dim_angle_class)
                if self.edge_attr_dim_parallel > 0:
                    self.edge_attr_4_encoder = encoder_mlp(self.edge_attr_dim_parallel, e_dim)
                    self.edge_attr_4_decoder = decoder_mlp(e_dim, self.edge_attr_dim_parallel)
                
            else:
                self.edge_attr_encoder = encoder_mlp(self.edge_attr_dim, e_dim)
                self.edge_attr_decoder = decoder_mlp(e_dim, self.edge_attr_dim)
        else:
            self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], e_dim*2), nn.GELU(),
                                        nn.Linear(e_dim*2, e_dim), nn.GELU())
            self.mlp_out_E = nn.Sequential(nn.Linear(e_dim, e_dim*2), nn.GELU(),
                                        nn.Linear(e_dim*2, input_dims['E']))
        self.init_edge = nn.Linear(e_dim, e_dim)
        
        # y encoder and decoder (not used)
        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], y_dim), nn.GELU(),
                                      nn.Linear(y_dim, y_dim), nn.GELU())
        self.mlp_out_y = nn.Sequential(nn.Linear(y_dim, y_dim), nn.GELU(),
                                       nn.Linear(y_dim, output_dims['y']))
        
        # text encoder config
        self.use_cross_attention = model_cfg.get('use_cross_attention', False)

        # modification in model architecture
        self.node_linear_attention = model_cfg.get("node_linear_attention", False)

        self.gnn_layers = nn.ModuleList([])
        for _ in range(self.n_layers):
            self.gnn_layers.append(
                XEyTransformerLayer(dx=x_dim,
                                    de=e_dim,
                                    dy=y_dim,
                                    dt=time_dim,
                                    use_y_feat=self.use_y_feat,
                                    use_source_condition=self.use_source_condition,
                                    source_dim=self.source_emb_dim,
                                    use_source_condition_edge=self.use_source_condition_edge,
                                    source_edge_dim=self.source_edge_emb_dim,
                                    n_head=hidden_dims['n_head'],
                                    dim_ffX=hidden_dims['dim_ffX'],
                                    dim_ffE=hidden_dims['dim_ffE'],
                                    dim_ffy=hidden_dims['dim_ffy'],
                                    cross_attn=self.use_cross_attention,
                                    cross_attn_dim=model_cfg['cross_attention_dim'] if self.use_cross_attention else 0)
            )

    def forward(self, X, E, y, t, node_mask, cond_feat=None, source_condition=None, source_condition_edge=None):

        bs, n = X.shape[0], X.shape[1]
        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        ## encoder
        if self.separate_encoder_node:
            if self.translation_dim > 0:
                x_bbox_trans = self.bbox_trans_encoder(X[:, :, :self.translation_dim+self.distance_dim])
            else:
                x_bbox_trans = 0

            if self.size_dim > 0:
                x_bbox_size = self.bbox_size_encoder(X[:, :, self.translation_dim+self.distance_dim:self.translation_dim+self.distance_dim+self.size_dim])
            else:
                x_bbox_size = 0

            if self.rotation_dim > 0:
                x_bbox_rot = self.bbox_rot_encoder(X[:, :, self.translation_dim+self.distance_dim+self.size_dim:self.bbox_dim])
            else:
                x_bbox_rot = 0
            
            if self.cat_dim > 0:
                x_class = self.class_encoder(X[:, :, self.bbox_dim:self.bbox_dim+self.cat_dim])
            else:
                x_class = 0

            if self.objfeat_dim > 0:
                x_objfeat = self.objfeat_encoder(X[:, :, self.bbox_dim+self.cat_dim:self.bbox_dim+self.cat_dim+self.objfeat_dim])
            else:
                x_objfeat = 0

            if self.mod_mask_dim > 0:
                x_mod = self.mod_mask_encoder(X[:, :, self.bbox_dim+self.cat_dim+self.objfeat_dim:self.bbox_dim+self.cat_dim+self.objfeat_dim+self.mod_mask_dim])
            else:
                x_mod = 0
            new_X = x_bbox_trans + x_bbox_size + x_bbox_rot + x_class + x_objfeat + x_mod
        else:
            new_X = self.mlp_in_X(X)
        new_X = self.init_node(new_X)

        if self.separate_encoder_edge:
            if self.edge_cat_dim > 0:
                edge_cat_feat = self.edge_cat_encoder(E[:, :, :, :self.edge_cat_dim])
            else:
                edge_cat_feat = 0

            if self.separate_encoder_edge_attr:
                if self.edge_attr_dim_dis > 0:
                    edge_attr_feat_1 = self.edge_attr_1_encoder(E[:, :, :, self.edge_cat_dim : self.edge_cat_dim+self.edge_attr_dim_dis])
                else:
                    edge_attr_feat_1 = 0
                if self.edge_attr_dim_angle > 0:
                    edge_attr_feat_2 = self.edge_attr_2_encoder(E[:, :, :, self.edge_cat_dim+self.edge_attr_dim_dis : self.edge_cat_dim+self.edge_attr_dim_dis+self.edge_attr_dim_angle])
                else:
                    edge_attr_feat_2 = 0
                if self.edge_attr_dim_angle_class > 0:
                    edge_attr_feat_3 = self.edge_attr_3_encoder(E[:, :, :, self.edge_cat_dim+self.edge_attr_dim_dis+self.edge_attr_dim_angle : self.edge_cat_dim+self.edge_attr_dim_dis+self.edge_attr_dim_angle+self.edge_attr_dim_angle_class])
                else:
                    edge_attr_feat_3 = 0
                if self.edge_attr_dim_parallel > 0:
                    edge_attr_feat_4 = self.edge_attr_4_encoder(E[:, :, :, self.edge_cat_dim+self.edge_attr_dim-self.edge_attr_dim_parallel : self.edge_cat_dim+self.edge_attr_dim])
                else:
                    edge_attr_feat_4 = 0
                edge_attr_feat = edge_attr_feat_1 + edge_attr_feat_2 + edge_attr_feat_3 + edge_attr_feat_4
            else:
                edge_attr_feat = self.edge_attr_encoder(E[:, :, :, self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_dim])

            new_E = edge_cat_feat + edge_attr_feat
        else:
            new_E = self.mlp_in_E(E)
        new_E = self.init_edge(new_E)

        new_y = self.mlp_in_y(y)

        if self.use_node_mask:
            after_in = utils.PlaceHolder(X=new_X, E=new_E, y=new_y).mask(node_mask)
        else:
            after_in = utils.PlaceHolder(X=new_X, E=new_E, y=new_y)
        X, E, y = after_in.X, after_in.E, after_in.y

        # get the time embedding
        if t.shape[0] > 1:
            t_embed = self.time_mlp(t.squeeze(-1)) 
        else:
            t_embed = self.time_mlp(t) 

        if self.gnn_residual_feature:  # True
            X_feat_to_out = X[..., :self.x_dim]
            E_feat_to_out = E[..., :self.e_dim]
            y_feat_to_out = y[..., :self.y_dim]

        ## gnn
        for layer in self.gnn_layers:
            X, E, y = layer(X, E, y, 
                            node_mask if self.use_node_mask else None, 
                            t_embed, 
                            condition_feat=cond_feat, 
                            source_condition=source_condition, 
                            source_condition_edge=source_condition_edge)


        if self.gnn_residual_feature:
            # X = X + X_feat_to_out
            E = E + E_feat_to_out
            y = y + y_feat_to_out

            X = torch.cat((X, X_feat_to_out), dim = 2)
            X = X.permute(0, 2, 1)
            X = self.final_res_block_node(X, t_embed)
            X = X.permute(0, 2, 1)


        ## decode
        if self.separate_encoder_node:
            output_X_list = []
            if self.translation_dim > 0:
                out_bbox_trans = self.bbox_trans_decoder(X)
                output_X_list.append(out_bbox_trans)
            if self.size_dim > 0:
                out_bbox_size = self.bbox_size_decoder(X)
                output_X_list.append(out_bbox_size)
            if self.rotation_dim > 0:
                out_bbox_rot = self.bbox_rot_decoder(X)
                output_X_list.append(out_bbox_rot)                    
            if self.cat_dim > 0:
                out_class = self.class_decoder(X)
                output_X_list.append(out_class) 
            if self.objfeat_dim > 0:
                out_objfeat = self.objfeat_decoder(X)
                output_X_list.append(out_objfeat) 
            if self.mod_mask_dim > 0:
                out_mod_mask = self.mod_mask_decoder(X)
                output_X_list.append(out_mod_mask) 
            X = torch.cat(output_X_list, dim=2).contiguous()
        else:
            X = self.mlp_out_X(X)

        if self.separate_encoder_edge:
            output_E_list = []
            if self.edge_cat_dim > 0:
                out_edge_cat = self.edge_cat_decoder(E)
                output_E_list.append(out_edge_cat)
            if self.separate_encoder_edge_attr:
                if self.edge_attr_dim_dis > 0:
                    out_edge_attr_1 = self.edge_attr_1_decoder(E)
                    output_E_list.append(out_edge_attr_1)
                if self.edge_attr_dim_angle > 0:
                    out_edge_attr_2 = self.edge_attr_2_decoder(E)
                    output_E_list.append(out_edge_attr_2)
                if self.edge_attr_dim_angle_class > 0:
                    out_edge_attr_3 = self.edge_attr_3_decoder(E)
                    output_E_list.append(out_edge_attr_3)
                if self.edge_attr_dim_parallel > 0:
                    out_edge_attr_4 = self.edge_attr_4_decoder(E)
                    output_E_list.append(out_edge_attr_4)
            else:
                out_edge_attr = self.edge_attr_decoder(E)
                output_E_list.append(out_edge_attr)
            E = torch.cat(output_E_list, dim=3).contiguous()
        else:
            E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        if self.use_node_mask:
            return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)
        else:
            return utils.PlaceHolder(X=X, E=E, y=y)