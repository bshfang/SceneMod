import math
import torch
import torch.nn as nn

import numpy as np

from utils import utils

from models.diffusion.noise_schedule import PredefinedNoiseSchedule
from models.diffusion import diffusion_utils

from models.scene_graph_transformer import SceneGraphTransformer


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



class DiffusionSG(nn.Module):
    def __init__(self, cfg, inference=False):
        super(DiffusionSG, self).__init__()

        self.cfg = cfg
        self.model_cfg = cfg['model']

        data_cfg = cfg['data']

        self.max_node = cfg['data'].get('max_node', 13)

        self.cfg = cfg
        # self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = self.model_cfg['diffusion_steps']

        self.gamma = PredefinedNoiseSchedule(self.model_cfg['diffusion_noise_schedule'], timesteps=self.model_cfg['diffusion_steps'])
        # print("self.gamma", self.gamma.gamma)
        diffusion_utils.check_issues_norm_values(self.gamma, 1, 1)  # self.norm_values[1], self.norm_values[2]

        self.use_undirected_noise = self.model_cfg.get("use_undirected_graph", False)

        # self.model = None  # define the graph transformer here, with different setting of encoder and decoder; with text condition
        self.model = SceneGraphTransformer(self.cfg)

        self.use_node_mask = self.model_cfg.get('use_node_mask', True)

        # === A new set of diffusion algorithm ===
        betas = diffusion_utils.get_betas(schedule_type='linear', b_start=0.0001, b_end=0.02, time_num=self.T+1)  # NOTE: +1 to align with previous algorithm
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()

        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        # === A new set of diffusion algorithm ===

        # loss
        self.separate_loss = self.model_cfg['separate_loss']

        self.separate_loss_edge_attr = self.model_cfg.get("separate_loss_edge_attr", False)

        self.unable_gnn = self.model_cfg.get("unable_gnn", False)  # if true, edge loss set to zero
        self.only_bbox_loss = self.model_cfg.get("only_bbox_loss", False)
        
        self.loss_use_mask = self.model_cfg.get("loss_use_mask", False)

        self.loss_trans_weight = self.model_cfg.get("loss_trans_weight", 1.0)
        self.loss_size_weight = self.model_cfg.get("loss_size_weight", 1.0)
        self.loss_rot_weight = self.model_cfg.get("loss_rot_weight", 1.0)
        self.loss_class_weight = self.model_cfg.get("loss_class_weight", 1.0)
        self.loss_objfeat_weight = self.model_cfg.get("loss_objfeat_weight", 1.0)
        self.loss_e_class_weight = self.model_cfg.get("loss_e_class_weight", 1.0)
        self.loss_e_attr_weight = self.model_cfg.get("loss_e_attr_weight", 1.0)

        self.mod_mask_weight = self.model_cfg.get("mod_mask_weight", -1.0)
        if self.mod_mask_weight > 0:
            self.use_mod_mask_weight = True
        else:
            self.use_mod_mask_weight = False

        self.translation_dim = data_cfg.get("translation_dim", 3)
        self.distance_dim = data_cfg.get("distance_dim", 0)
        self.size_dim = data_cfg.get("size_dim", 3)
        self.rotation_dim = data_cfg.get("rotation_dim", 2)
        self.cat_dim = data_cfg.get("cat_dim", 23)
        self.objfeat_dim = data_cfg.get("objfeat_dim", 0)
        self.bbox_dim = self.translation_dim + self.distance_dim + self.size_dim + self.rotation_dim
        self.node_dim = self.bbox_dim + self.cat_dim + self.objfeat_dim

        self.mod_mask_dim = data_cfg.get("mod_mask_dim", 0)

        self.edge_cat_dim = data_cfg.get("edge_cat_dim", 15)
        self.edge_attr_dim = data_cfg.get("edge_attr_dim", 0)

        self.edge_attr_dim_dis = data_cfg.get("edge_attr_dim_dis", 0)
        self.edge_attr_dim_angle = data_cfg.get("edge_attr_dim_angle", 0)
        self.edge_attr_dim_angle_class = data_cfg.get("edge_attr_dim_angle_class", 0)
        self.edge_attr_dim_parallel = data_cfg.get("edge_attr_dim_parallel", 0)

        self.edge_dim = self.edge_cat_dim + self.edge_attr_dim

        # the feature dimension for model prediction
        feat_dims = self.model_cfg.get("feat_dims", None)
        if feat_dims is not None:
            self.pred_translation_dim = feat_dims['translation_dim']
            self.pred_distance_dim = feat_dims['distance_dim']
            self.pred_size_dim = feat_dims['size_dim']
            self.pred_rotation_dim = feat_dims['rotation_dim']
            self.pred_cat_dim = feat_dims['cat_dim']
            self.pred_objfeat_dim = feat_dims['objfeat_dim']
            self.pred_mod_mask_dim = feat_dims['mod_mask_dim']
            self.pred_edge_cat_dim = feat_dims['edge_cat_dim']
            self.pred_edge_attr_dim = feat_dims['edge_attr_dim']
            self.pred_edge_attr_dim_dis = feat_dims['edge_attr_dim_dis']
            self.pred_edge_attr_dim_angle = feat_dims['edge_attr_dim_angle']
            self.pred_edge_attr_dim_angle_class = feat_dims['edge_attr_dim_angle_class']
            self.pred_edge_attr_dim_parallel = feat_dims['edge_attr_dim_parallel']
            self.pred_bbox_dim = self.pred_translation_dim + self.pred_distance_dim + self.pred_size_dim + self.pred_rotation_dim
        else:
            self.pred_translation_dim = self.translation_dim
            self.pred_distance_dim = self.distance_dim
            self.pred_size_dim = self.size_dim
            self.pred_rotation_dim = self.rotation_dim
            self.pred_cat_dim = self.cat_dim
            self.pred_objfeat_dim = self.objfeat_dim
            self.pred_mod_mask_dim = self.mod_mask_dim
            self.pred_edge_cat_dim = self.edge_cat_dim
            self.pred_edge_attr_dim = self.edge_attr_dim
            self.pred_edge_attr_dim_dis = self.edge_attr_dim_dis
            self.pred_edge_attr_dim_angle = self.edge_attr_dim_angle
            self.pred_edge_attr_dim_angle_class = self.edge_attr_dim_angle_class
            self.pred_edge_attr_dim_parallel = self.edge_attr_dim_parallel
            self.pred_bbox_dim = self.bbox_dim

        self.use_no_edge_feat_source = self.model_cfg.get('use_no_edge_feat_source', True)
        self.use_no_edge_feat_target = self.model_cfg.get('use_no_edge_feat_target', False)

        # data & model info
        output_dims = self.model_cfg['output_dims']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']

        # instance condition, use the similar setting in diffuscene
        self.instance_condition = self.model_cfg.get("instance_condition", False)
        self.instance_emb_dim = self.model_cfg.get("instance_emb_dim", 128)
        if self.instance_condition:
            self.register_parameter(
                "positional_embedding",
                nn.Parameter(torch.randn(self.max_node, self.instance_emb_dim))
            )

        # floor bound condition
        self.floorplan_condition = self.model_cfg.get("floorplan_condition", False)
        self.floorplan_condition_emb_dim = self.model_cfg.get("floorplan_condition_emb_dim", 128)
        if self.floorplan_condition:
            self.fc_floor_bound_condition = nn.Sequential(
                    nn.Linear(4, self.floorplan_condition_emb_dim, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Linear(self.floorplan_condition_emb_dim, self.floorplan_condition_emb_dim, bias=False),
                )
        
        # source room node & edge condition (stage-two: graph to 3D generation)
        self.source_room_condition_node_s2 = self.model_cfg.get("source_room_condition_node_s2", False)
        self.source_node_emb_dim = self.model_cfg.get("source_node_emb_dim", 256)
        if self.source_room_condition_node_s2:
            self.fc_source_node_condition_cat = nn.Sequential(
                    nn.Linear(self.cat_dim, self.source_node_emb_dim//2, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Linear(self.source_node_emb_dim//2, self.source_node_emb_dim//2, bias=False),
                )
            self.fc_source_node_condition_objfeat = nn.Sequential(
                    nn.Linear(self.objfeat_dim, self.source_node_emb_dim//2, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Linear(self.source_node_emb_dim//2, self.source_node_emb_dim//2, bias=False),
                )
        
        self.source_room_condition_edge_s2 = self.model_cfg.get("source_room_condition_edge_s2", False)
        self.source_edge_emb_dim = self.model_cfg.get("source_edge_emb_dim", 0)
        if self.source_room_condition_edge_s2:
            self.fc_source_edge_condition_spatial = nn.Sequential(
                    nn.Linear(self.edge_cat_dim, self.source_edge_emb_dim//2, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Linear(self.source_edge_emb_dim//2, self.source_edge_emb_dim//2, bias=False),
                )
            self.edge_attr_cat_dim = self.edge_attr_dim_angle_class + self.edge_attr_dim_parallel
            self.fc_source_edge_condition_attr = nn.Sequential(
                    nn.Linear(self.edge_attr_cat_dim, self.source_edge_emb_dim//2, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Linear(self.source_edge_emb_dim//2, self.source_edge_emb_dim//2, bias=False),
                )

        # origin bbox condition
        self.source_room_condition_bbox_s1 = self.model_cfg.get("source_room_condition_bbox_s1", False)
        self.source_node_emb_dim_bbox = self.model_cfg.get("source_node_emb_dim_bbox", 128)
        if self.source_room_condition_bbox_s1:
            self.fc_source_node_origin_bbox_condition = nn.Sequential(
                    nn.Linear(9, self.source_node_emb_dim, bias=False),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Linear(self.source_node_emb_dim, self.source_node_emb_dim_bbox, bias=False),
                )

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def apply_noise_new(self, X, E, y, node_mask, undirected_noise, use_node_mask=True):
        t_int = torch.randint(0, self.T + 1, size=(X.size(0),), device=X.device)

        eps = diffusion_utils.sample_feature_noise(X.size(), E.size(), y.size(), node_mask, undirected_E=undirected_noise, use_node_mask=use_node_mask).type_as(X)

        X_t = self.q_sample(X, t_int, eps.X)
        E_t = self.q_sample(E, t_int, eps.E)
        y_t = y

        noisy_data = {'epsX': eps.X, 'epsE': eps.E, 'epsy': eps.y,
                      'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't_int': t_int}
        return noisy_data

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)   q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
                self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise
        )

    def get_loss(self, data, condition_feat=None):

        # 1) get dense graph
        input_data = data['source_data']
        # get the no_edge_feat
        if self.use_no_edge_feat_source:
            no_edge_feat = torch.zeros((input_data.edge_attr.shape[1]), dtype=input_data.edge_attr.dtype, device=input_data.edge_attr.device)
            no_edge_feat[0] = 1.0
            no_edge_feat[:self.edge_cat_dim] = no_edge_feat[:self.edge_cat_dim] * 2 - 1.0
            no_edge_feat[self.edge_cat_dim+self.edge_attr_dim_angle_class-1] = 1.0
            no_edge_feat[self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_dim_angle_class] = no_edge_feat[self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_dim_angle_class] * 2 - 1.0
            no_edge_feat[self.edge_cat_dim+self.edge_attr_dim_angle_class+self.edge_attr_dim_parallel-1] = 1.0
            no_edge_feat[self.edge_cat_dim+self.edge_attr_dim_angle_class:self.edge_cat_dim+self.edge_attr_dim_angle_class+self.edge_attr_dim_parallel] = \
                no_edge_feat[self.edge_cat_dim+self.edge_attr_dim_angle_class:self.edge_cat_dim+self.edge_attr_dim_angle_class+self.edge_attr_dim_parallel] * 2 - 1.0
        else:
            no_edge_feat = None

        dense_data, node_mask = utils.to_dense(input_data.x, input_data.edge_index, input_data.edge_attr, input_data.batch, max_node=self.max_node, no_edge_feat=no_edge_feat) 

        dense_data = dense_data.mask(node_mask)
        X_source, E_source = dense_data.X, dense_data.E
        y = input_data.y

        source_condition = None
        source_condition_edge = None

        ## get various condition

        # - instance condition
        batch_size, _, _ = X_source.shape
        if self.instance_condition:
            instance_indices = torch.arange(self.max_node).long().to(X_source.device)[None, :].repeat(batch_size, 1)
            instan_condition_f = self.positional_embedding[instance_indices, :]
            source_condition = instan_condition_f

        # - floor bound condition
        if self.floorplan_condition:
            floor_bound_f = self.fc_floor_bound_condition(data['floor_bounds'])
            source_condition = torch.cat([floor_bound_f[:, None, :].repeat(1, self.max_node, 1), source_condition], dim=-1).contiguous()

        if self.source_room_condition_node_s2:
            source_room_condition_cat = self.fc_source_node_condition_cat(X_source[:, :, :self.cat_dim])
            source_room_condition_objfeat = self.fc_source_node_condition_objfeat(X_source[:, :, self.cat_dim:self.cat_dim+self.objfeat_dim])
            source_room_condition = torch.cat([source_room_condition_cat, source_room_condition_objfeat], dim=-1).contiguous()

            source_room_condition = source_room_condition * node_mask.unsqueeze(-1)
            # source_condition = source_room_condition
            source_condition = torch.cat([source_condition, source_room_condition], dim=-1).contiguous()

        if self.source_room_condition_edge_s2:
            source_room_condition_edge_spatial = self.fc_source_edge_condition_spatial(E_source[:, :, :, :self.edge_cat_dim])
            source_room_condition_edge_attr = self.fc_source_edge_condition_attr(E_source[:, :, :, self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_cat_dim])
            source_room_condition_edge = torch.cat([source_room_condition_edge_spatial, source_room_condition_edge_attr], dim=-1).contiguous()

            e_mask1 = node_mask.unsqueeze(-1).unsqueeze(2)
            e_mask2 = node_mask.unsqueeze(-1).unsqueeze(1)
            source_room_condition_edge = source_room_condition_edge * e_mask1 * e_mask2

            source_condition_edge = source_room_condition_edge

        if self.source_room_condition_bbox_s1:
            source_room_condition_bbox = self.fc_source_node_origin_bbox_condition(data['source_node_condition'])
            source_room_condition_bbox = source_room_condition_bbox * node_mask.unsqueeze(-1)
            source_condition = torch.cat([source_condition, source_room_condition_bbox], dim=-1).contiguous()

        if 'target_data' in data:
            target_data = data['target_data']

            if self.use_no_edge_feat_target:
                no_edge_feat = torch.zeros((input_data.edge_attr.shape[1]), dtype=input_data.edge_attr.dtype, device=input_data.edge_attr.device)
                no_edge_feat[0] = 1.0
                no_edge_feat[:self.edge_cat_dim] = no_edge_feat[:self.edge_cat_dim] * 2 - 1.0
                no_edge_feat[self.edge_cat_dim+self.edge_attr_dim_angle_class-1] = 1.0
                no_edge_feat[self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_dim_angle_class] = no_edge_feat[self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_dim_angle_class] * 2 - 1.0
                no_edge_feat[self.edge_cat_dim+self.edge_attr_dim_angle_class+self.edge_attr_dim_parallel-1] = 1.0
                no_edge_feat[self.edge_cat_dim+self.edge_attr_dim_angle_class:self.edge_cat_dim+self.edge_attr_dim_angle_class+self.edge_attr_dim_parallel] = \
                    no_edge_feat[self.edge_cat_dim+self.edge_attr_dim_angle_class:self.edge_cat_dim+self.edge_attr_dim_angle_class+self.edge_attr_dim_parallel] * 2 - 1.0
            else:
                no_edge_feat = None

            dense_data_target, node_mask = utils.to_dense(target_data.x, target_data.edge_index, target_data.edge_attr, target_data.batch, max_node=self.max_node, no_edge_feat=no_edge_feat)
            dense_data_target = dense_data_target.mask(node_mask)
            X, E = dense_data_target.X, dense_data_target.E

        else:  # in uncondition, name the input graph as 'source_graph' in data
            X, E = X_source, E_source

        # 2) add noise to data
        if self.use_node_mask:
            noisy_data = self.apply_noise_new(X, E, y, node_mask, self.use_undirected_noise)
        else:
            noisy_data = self.apply_noise_new(X, E, y, node_mask, self.use_undirected_noise, use_node_mask=False)

        # 3) pred noise
        pred = self.forward(noisy_data, node_mask, condition_feat, source_condition=source_condition, source_condition_edge=source_condition_edge)

        # 4) calculate loss
        loss, loss_dict = self.compute_loss(masked_pred_epsX=pred.X,
                            masked_pred_epsE=pred.E,
                            pred_y=pred.y,
                            true_epsX=noisy_data['epsX'],
                            true_epsE=noisy_data['epsE'],
                            true_y=noisy_data['epsy'],
                            node_mask=node_mask,
                            node_mod_mask=data['mod_masks'] if 'mod_masks' in data else None)

        return loss, loss_dict
    
    def forward(self, noisy_data, node_mask, condition_feat, source_condition=None, source_condition_edge=None):
        return self.model(noisy_data['X_t'], noisy_data['E_t'], noisy_data['y_t'], noisy_data['t_int'], node_mask, condition_feat, source_condition=source_condition, source_condition_edge=source_condition_edge)
    

    def compute_loss(self, masked_pred_epsX, masked_pred_epsE, pred_y, true_epsX, true_epsE, true_y, node_mask, weights=None, node_mod_mask=None):
        """
        compute loss
        """
        if self.loss_use_mask:
            node_num = node_mask.sum(dim=1)
            edge_node_num = torch.square(node_num) - node_num


        loss_dict = {}
        if self.separate_loss:

            # node
            if weights is not None:
                node_weight = weights.unsqueeze(-1).unsqueeze(-1)
                loss_bbox = (node_weight*((masked_pred_epsX[:, :, :self.bbox_dim] - true_epsX[:, :, :self.bbox_dim])**2)).mean()
                loss_trans = (node_weight*((masked_pred_epsX[:, :, :self.translation_dim+self.distance_dim] - true_epsX[:, :, :self.translation_dim+self.distance_dim])**2)).mean()
                loss_size = (node_weight*((masked_pred_epsX[:, :, self.translation_dim+self.distance_dim:self.translation_dim+self.distance_dim+self.size_dim] - true_epsX[:, :, self.translation_dim+self.distance_dim:self.translation_dim+self.distance_dim+self.size_dim])**2)).mean()
                loss_rot = (node_weight*((masked_pred_epsX[:, :, self.translation_dim+self.distance_dim+self.size_dim:self.bbox_dim] - true_epsX[:, :, self.translation_dim+self.distance_dim+self.size_dim:self.bbox_dim])**2)).mean()
                loss_class = (node_weight*((masked_pred_epsX[:, :, self.bbox_dim:self.bbox_dim+self.cat_dim] - true_epsX[:, :, self.bbox_dim:self.bbox_dim+self.cat_dim])**2)).mean()
                if self.objfeat_dim > 0:
                    loss_objfeat = (node_weight*((masked_pred_epsX[:, :, self.bbox_dim+self.cat_dim:self.bbox_dim+self.cat_dim+self.objfeat_dim] - true_epsX[:, :, self.bbox_dim+self.cat_dim:self.bbox_dim+self.cat_dim+self.objfeat_dim])**2)).mean()
                else:
                    loss_objfeat = 0

                batch_size = weights.shape[0]
                loss_bbox = loss_bbox * batch_size / torch.sum(node_weight)
                loss_trans = loss_trans * batch_size / torch.sum(node_weight)
                loss_size = loss_size * batch_size / torch.sum(node_weight)
                loss_rot = loss_rot * batch_size / torch.sum(node_weight)
                loss_class = loss_class * batch_size / torch.sum(node_weight)
                loss_objfeat = loss_objfeat * batch_size / torch.sum(node_weight)
            else:
                if self.loss_use_mask:
                    loss_bbox = ((masked_pred_epsX[:, :, :self.bbox_dim] - true_epsX[:, :, :self.bbox_dim])**2).mean(dim=2).sum(dim=1)
                    loss_bbox = (loss_bbox / (node_num + 1e-8)).mean()
                    loss_trans = ((masked_pred_epsX[:, :, :self.translation_dim+self.distance_dim] - true_epsX[:, :, :self.translation_dim+self.distance_dim])**2).mean(dim=2).sum(dim=1)
                    loss_trans = (loss_trans / (node_num + 1e-8)).mean()
                    loss_size = ((masked_pred_epsX[:, :, self.translation_dim+self.distance_dim:self.translation_dim+self.distance_dim+self.size_dim] - true_epsX[:, :, self.translation_dim+self.distance_dim:self.translation_dim+self.distance_dim+self.size_dim])**2).mean(dim=2).sum(dim=1)
                    loss_size = (loss_size / (node_num + 1e-8)).mean()
                    loss_rot = ((masked_pred_epsX[:, :, self.translation_dim+self.distance_dim+self.size_dim:self.bbox_dim] - true_epsX[:, :, self.translation_dim+self.distance_dim+self.size_dim:self.bbox_dim])**2).mean(dim=2).sum(dim=1)
                    loss_rot = (loss_rot / (node_num + 1e-8)).mean()
                    loss_class = ((masked_pred_epsX[:, :, self.bbox_dim:self.bbox_dim+self.cat_dim] - true_epsX[:, :, self.bbox_dim:self.bbox_dim+self.cat_dim])**2).mean(dim=2).sum(dim=1)
                    loss_class = (loss_class / (node_num + 1e-8)).mean()
                    if self.objfeat_dim > 0:
                        loss_objfeat = ((masked_pred_epsX[:, :, self.bbox_dim+self.cat_dim:self.bbox_dim+self.cat_dim+self.objfeat_dim] - true_epsX[:, :, self.bbox_dim+self.cat_dim:self.bbox_dim+self.cat_dim+self.objfeat_dim])**2).mean(dim=2).sum(dim=1)
                        loss_objfeat = (loss_objfeat / (node_num + 1e-8)).mean()
                    else:
                        loss_objfeat = 0
                else:  ## Here is the code
                    b, n, _ = masked_pred_epsX.shape
                    mod_weights = torch.ones((b,n), dtype=masked_pred_epsX.dtype, device=masked_pred_epsX.device)
                    if self.use_mod_mask_weight:
                        mod_weights[node_mod_mask] = self.mod_mask_weight
                    mod_weights = mod_weights.unsqueeze(-1)
                    if self.pred_bbox_dim > 0:
                        loss_bbox = (mod_weights * ((masked_pred_epsX[:, :, :self.bbox_dim] - true_epsX[:, :, :self.bbox_dim])**2)).mean()  # (b,n,c).mean()
                    else:
                        loss_bbox = 0
                    if self.pred_translation_dim > 0:
                        loss_trans = (mod_weights * ((masked_pred_epsX[:, :, :self.translation_dim+self.distance_dim] - true_epsX[:, :, :self.translation_dim+self.distance_dim])**2)).mean()
                    else:
                        loss_trans = 0
                    if self.pred_size_dim > 0:
                        loss_size = ((masked_pred_epsX[:, :, self.translation_dim+self.distance_dim:self.translation_dim+self.distance_dim+self.size_dim] - true_epsX[:, :, self.translation_dim+self.distance_dim:self.translation_dim+self.distance_dim+self.size_dim])**2).mean()
                    else:
                        loss_size = 0
                    if self.pred_rotation_dim > 0:
                        loss_rot = (mod_weights * ((masked_pred_epsX[:, :, self.translation_dim+self.distance_dim+self.size_dim:self.bbox_dim] - true_epsX[:, :, self.translation_dim+self.distance_dim+self.size_dim:self.bbox_dim])**2)).mean()
                    else:
                        loss_rot = 0
                    if self.pred_cat_dim > 0:
                        loss_class = ((masked_pred_epsX[:, :, self.bbox_dim:self.bbox_dim+self.cat_dim] - true_epsX[:, :, self.bbox_dim:self.bbox_dim+self.cat_dim])**2).mean()
                    else:
                        loss_class = 0
                    if self.pred_objfeat_dim > 0:
                        loss_objfeat = ((masked_pred_epsX[:, :, self.bbox_dim+self.cat_dim:self.bbox_dim+self.cat_dim+self.objfeat_dim] - true_epsX[:, :, self.bbox_dim+self.cat_dim:self.bbox_dim+self.cat_dim+self.objfeat_dim])**2).mean()
                    else:
                        loss_objfeat = 0
                    if self.pred_mod_mask_dim > 0:
                        loss_mod = ((masked_pred_epsX[:, :, self.bbox_dim+self.cat_dim+self.objfeat_dim:self.bbox_dim+self.cat_dim+self.objfeat_dim+self.mod_mask_dim] - true_epsX[:, :, self.bbox_dim+self.cat_dim+self.objfeat_dim:self.bbox_dim+self.cat_dim+self.objfeat_dim+self.mod_mask_dim])**2).mean()
                    else:
                        loss_mod = 0

            # edge
            if self.unable_gnn:
                loss_e_class = 0
                loss_e_attr = 0
            else:
                if weights is not None:
                    edge_weight = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    loss_e_class = (edge_weight*((masked_pred_epsE[:, :, :, :self.edge_cat_dim] - true_epsE[:, :, :, :self.edge_cat_dim])**2)).mean()
                    loss_e_attr = (edge_weight*((masked_pred_epsE[:, :, :, self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_dim] - true_epsE[:, :, :, self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_dim])**2)).mean()
                    batch_size = weights.shape[0]
                    loss_e_class = loss_e_class * batch_size / torch.sum(node_weight)
                    loss_e_attr = loss_e_attr * batch_size / torch.sum(node_weight)
                else:
                    if self.loss_use_mask:
                        loss_e_class = ((masked_pred_epsE[:, :, :, :self.edge_cat_dim] - true_epsE[:, :, :, :self.edge_cat_dim])**2).mean(dim=3).sum(dim=(1,2))
                        loss_e_class = (loss_e_class / (edge_node_num + 1e-8)).mean()
                        if self.separate_loss_edge_attr:
                            loss_e_attr_dis = ((masked_pred_epsE[:, :, :, self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_dim_dis] - \
                                                true_epsE[:, :, :, self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_dim_dis])**2).mean(dim=3).sum(dim=(1,2))
                            loss_e_attr_dis = (loss_e_attr_dis / (edge_node_num + 1e-8)).mean()
                            loss_e_attr_angle = ((masked_pred_epsE[:, :, :, self.edge_cat_dim+self.edge_attr_dim_dis:self.edge_cat_dim+self.edge_attr_dim_dis+self.edge_attr_dim_angle] - \
                                                true_epsE[:, :, :, self.edge_cat_dim+self.edge_attr_dim_dis:self.edge_cat_dim+self.edge_attr_dim_dis+self.edge_attr_dim_angle])**2).mean(dim=3).sum(dim=(1,2))
                            loss_e_attr_angle = (loss_e_attr_angle / (edge_node_num + 1e-8)).mean()
                            loss_e_attr_angle_class = ((masked_pred_epsE[:, :, :, self.edge_cat_dim+self.edge_attr_dim_dis+self.edge_attr_dim_angle:self.edge_cat_dim+self.edge_attr_dim_dis+self.edge_attr_dim_angle+self.edge_attr_dim_angle_class] - \
                                                        true_epsE[:, :, :, self.edge_cat_dim+self.edge_attr_dim_dis+self.edge_attr_dim_angle:self.edge_cat_dim+self.edge_attr_dim_dis+self.edge_attr_dim_angle+self.edge_attr_dim_angle_class])**2).mean(dim=3).sum(dim=(1,2))
                            loss_e_attr_angle_class = (loss_e_attr_angle_class / (edge_node_num + 1e-8)).mean()
                            loss_e_attr_parallel = ((masked_pred_epsE[:, :, :, self.edge_cat_dim+self.edge_attr_dim-self.edge_attr_dim_parallel:self.edge_cat_dim+self.edge_attr_dim] - \
                                                    true_epsE[:, :, :, self.edge_cat_dim+self.edge_attr_dim-self.edge_attr_dim_parallel:self.edge_cat_dim+self.edge_attr_dim])**2).mean(dim=3).sum(dim=(1,2))
                            loss_e_attr_parallel = (loss_e_attr_parallel / (edge_node_num + 1e-8)).mean()

                            loss_e_attr = loss_e_attr_dis + loss_e_attr_angle + loss_e_attr_angle_class + loss_e_attr_parallel
                        else:
                            loss_e_attr = ((masked_pred_epsE[:, :, :, self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_dim] - true_epsE[:, :, :, self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_dim])**2).mean(dim=3).sum(dim=(1,2))
                            loss_e_attr = (loss_e_attr / (edge_node_num + 1e-8)).mean()
                    else:
                        if self.pred_edge_cat_dim > 0:
                            loss_e_class = ((masked_pred_epsE[:, :, :, :self.edge_cat_dim] - true_epsE[:, :, :, :self.edge_cat_dim])**2).mean()
                        else:
                            loss_e_class = 0
                        if self.separate_loss_edge_attr:  # false
                            loss_e_attr_dis = ((masked_pred_epsE[:, :, :, self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_dim_dis] - \
                                                true_epsE[:, :, :, self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_dim_dis])**2).mean()
                            loss_e_attr_angle = ((masked_pred_epsE[:, :, :, self.edge_cat_dim+self.edge_attr_dim_dis:self.edge_cat_dim+self.edge_attr_dim_dis+self.edge_attr_dim_angle] - \
                                                true_epsE[:, :, :, self.edge_cat_dim+self.edge_attr_dim_dis:self.edge_cat_dim+self.edge_attr_dim_dis+self.edge_attr_dim_angle])**2).mean()
                            loss_e_attr_angle_class = ((masked_pred_epsE[:, :, :, self.edge_cat_dim+self.edge_attr_dim_dis+self.edge_attr_dim_angle:self.edge_cat_dim+self.edge_attr_dim_dis+self.edge_attr_dim_angle+self.edge_attr_dim_angle_class] - \
                                                        true_epsE[:, :, :, self.edge_cat_dim+self.edge_attr_dim_dis+self.edge_attr_dim_angle:self.edge_cat_dim+self.edge_attr_dim_dis+self.edge_attr_dim_angle+self.edge_attr_dim_angle_class])**2).mean()
                            loss_e_attr_parallel = ((masked_pred_epsE[:, :, :, self.edge_cat_dim+self.edge_attr_dim-self.edge_attr_dim_parallel:self.edge_cat_dim+self.edge_attr_dim] - \
                                                    true_epsE[:, :, :, self.edge_cat_dim+self.edge_attr_dim-self.edge_attr_dim_parallel:self.edge_cat_dim+self.edge_attr_dim])**2).mean()

                            loss_e_attr = loss_e_attr_dis + loss_e_attr_angle + loss_e_attr_angle_class + loss_e_attr_parallel

                        else:
                            if self.pred_edge_attr_dim:
                                loss_e_attr = ((masked_pred_epsE[:, :, :, self.pred_edge_cat_dim:self.pred_edge_cat_dim+self.pred_edge_attr_dim] - true_epsE[:, :, :, self.pred_edge_cat_dim:self.pred_edge_cat_dim+self.pred_edge_attr_dim])**2).mean()
                            else:
                                loss_e_attr = 0
                    

            if self.only_bbox_loss:
                losses = loss_bbox + \
                         self.loss_class_weight * loss_class + \
                         self.loss_objfeat_weight * loss_objfeat + \
                         self.loss_e_class_weight * loss_e_class + \
                         self.loss_e_attr_weight * loss_e_attr + \
                         loss_mod
            else:
                losses = self.loss_trans_weight * loss_trans + \
                        self.loss_size_weight * loss_size + \
                        self.loss_rot_weight * loss_rot + \
                        self.loss_class_weight * loss_class + \
                        self.loss_objfeat_weight * loss_objfeat + \
                        self.loss_e_class_weight * loss_e_class + \
                        self.loss_e_attr_weight * loss_e_attr + \
                        loss_mod
                # losses = losses.mean()

            loss_dict.update({
                "losses": losses.item(),
                "loss_bbox": loss_bbox.item() if loss_bbox!=0 else loss_bbox,
                "loss_trans": loss_trans.item() if loss_trans!=0 else loss_trans,
                "loss_size": loss_size.item() if loss_size!=0 else loss_size,
                "loss_rot": loss_rot.item() if loss_rot!=0 else loss_rot,
                "loss_class": loss_class.item() if loss_class!=0 else loss_class,
                "loss_objfeat": loss_objfeat.item() if loss_objfeat!=0 else loss_objfeat,
                "loss_e_class": loss_e_class.item() if loss_e_class!=0 else loss_e_class,
                "loss_e_attr": loss_e_attr.item() if loss_e_attr!=0 else loss_e_attr,
                "loss_mod": loss_mod.item() if loss_mod!=0 else loss_mod
            })

            
        else:
            mse_X = ((masked_pred_epsX - true_epsX)**2).mean()
            mse_E = ((masked_pred_epsE - true_epsE)**2).mean()

            losses = mse_X + mse_E

            loss_dict.update({
                "losses": losses.item(),
                "mse_X": mse_X.item(),
                "mse_E": mse_E.item(),
            })


        return losses, loss_dict
    

    def generate_scene_graph_new(self, 
                                 val_data, 
                                 device, 
                                 batch_size=1, 
                                 text=None, 
                                 cond_fn=None, 
                                 cond_fn_2=None, 
                                 condition_feat=None, 
                                 return_input=False, 
                                 return_raw_output=False,
                                 node_source_data=None,
                                 edge_source_data=None,
                                 node_source_bbox_condition=None):
        if node_source_data is not None:
            b, n, _ = node_source_data.shape
            node_mask = torch.ones((b,n), dtype=torch.bool, device=node_source_data.device)
            X = node_source_data
            E = edge_source_data
        else:
            # node mask for all val_data
            input_data = val_data['source_data']

            if self.use_no_edge_feat_source:
                no_edge_feat = torch.zeros((input_data.edge_attr.shape[1]), dtype=input_data.edge_attr.dtype, device=input_data.edge_attr.device)
                no_edge_feat[0] = 1.0
                no_edge_feat[:self.edge_cat_dim] = no_edge_feat[:self.edge_cat_dim] * 2 - 1.0
                no_edge_feat[self.edge_cat_dim+self.edge_attr_dim_angle_class-1] = 1.0
                no_edge_feat[self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_dim_angle_class] = no_edge_feat[self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_dim_angle_class] * 2 - 1.0
                no_edge_feat[self.edge_cat_dim+self.edge_attr_dim_angle_class+self.edge_attr_dim_parallel-1] = 1.0
                no_edge_feat[self.edge_cat_dim+self.edge_attr_dim_angle_class:self.edge_cat_dim+self.edge_attr_dim_angle_class+self.edge_attr_dim_parallel] = \
                    no_edge_feat[self.edge_cat_dim+self.edge_attr_dim_angle_class:self.edge_cat_dim+self.edge_attr_dim_angle_class+self.edge_attr_dim_parallel] * 2 - 1.0
            else:
                no_edge_feat = None

            dense_data, node_mask = utils.to_dense(input_data.x, input_data.edge_index, input_data.edge_attr, input_data.batch, max_node=self.max_node, no_edge_feat=no_edge_feat)  # 13: use a number in the config
            dense_data = dense_data.mask(node_mask)
            
            X, E = dense_data.X, dense_data.E
            y = input_data.y
        
        if self.source_room_condition_bbox_s1:
            if node_source_bbox_condition is not None:
                source_node_bbox_condition = node_source_bbox_condition
            else:
                source_node_bbox_condition = val_data['source_node_condition']

        batch_size = node_mask.shape[0]
        num_nodes = torch.sum(node_mask, dim=1).to(device)
        arange = torch.arange(self.max_node, device=device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < num_nodes.unsqueeze(1)
        node_mask = node_mask.float()
        
        source_condition = None
        source_condition_edge = None
        return_dense_data = False

        # - instance condition
        batch_size, _, _ = X.shape
        if self.instance_condition:
            instance_indices = torch.arange(self.max_node).long().to(X.device)[None, :].repeat(batch_size, 1)
            instan_condition_f = self.positional_embedding[instance_indices, :]
            source_condition = instan_condition_f

        # - floor bound condition
        if self.floorplan_condition:
            floor_bound_f = self.fc_floor_bound_condition(val_data['floor_bounds'])
            source_condition = torch.cat([floor_bound_f[:, None, :].repeat(1, self.max_node, 1), source_condition], dim=-1).contiguous()

        # - source room node condition        
        if self.source_room_condition_node_s2:
            source_room_condition_cat = self.fc_source_node_condition_cat(X[:, :, :self.cat_dim])
            source_room_condition_objfeat = self.fc_source_node_condition_objfeat(X[:, :, self.cat_dim:self.cat_dim+self.objfeat_dim])
            source_room_condition = torch.cat([source_room_condition_cat, source_room_condition_objfeat], dim=-1).contiguous()

            source_room_condition = source_room_condition * node_mask.unsqueeze(-1)
            # source_condition = source_room_condition
            source_condition = torch.cat([source_condition, source_room_condition], dim=-1).contiguous()

        if self.source_room_condition_edge_s2:
            source_room_condition_edge_spatial = self.fc_source_edge_condition_spatial(E[:, :, :, :self.edge_cat_dim])
            source_room_condition_edge_attr = self.fc_source_edge_condition_attr(E[:, :, :, self.edge_cat_dim:self.edge_cat_dim+self.edge_attr_cat_dim])
            source_room_condition_edge = torch.cat([source_room_condition_edge_spatial, source_room_condition_edge_attr], dim=-1).contiguous()

            e_mask1 = node_mask.unsqueeze(-1).unsqueeze(2)
            e_mask2 = node_mask.unsqueeze(-1).unsqueeze(1)
            source_room_condition_edge = source_room_condition_edge * e_mask1 * e_mask2

            source_condition_edge = source_room_condition_edge

        if self.source_room_condition_bbox_s1:
            source_room_condition_bbox = self.fc_source_node_origin_bbox_condition(source_node_bbox_condition)
            source_room_condition_bbox = source_room_condition_bbox * node_mask.unsqueeze(-1)
            source_condition = torch.cat([source_condition, source_room_condition_bbox], dim=-1).contiguous()


        z_T = diffusion_utils.sample_feature_noise(X_size=(batch_size, self.max_node, self.Xdim_output),
                                                   E_size=(batch_size, self.max_node, self.max_node, self.Edim_output),
                                                   y_size=(batch_size, self.ydim_output),
                                                   node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        for t in reversed(range(0, len(self.betas))):

            t_ = torch.empty(batch_size, dtype=torch.int64, device=X.device).fill_(t)
            X_mean, E_mean, model_variance, model_log_variance = self.p_mean_variance(X, E, y, t_, node_mask, condition_feat=condition_feat, cond_fn=cond_fn, cond_fn_2=cond_fn_2, source_condition=source_condition, source_condition_edge=source_condition_edge)

            # noise
            eps = diffusion_utils.sample_feature_noise(X_mean.size(), E_mean.size(), (batch_size, self.ydim_output), node_mask).type_as(X_mean)

            # =========================
            # update the mu by guidance gradient here
            maximum_noise_step = 10
            # print(t)
            if cond_fn is not None and t < maximum_noise_step:

                floor_bound = val_data['floor_bounds']

                gradient_X, _, loss_dict = cond_fn(X_mean, E_mean, node_source_data, node_source_bbox_condition, node_mask, floor_bound)

                if gradient_X is not None:
                    model_variance_X = self._extract(model_variance, t_, X_mean.shape) * torch.ones_like(X_mean)
                    model_log_variance_X = self._extract(model_log_variance, t_, X_mean.shape) * torch.ones_like(X_mean)
                    X_mean = X_mean - model_variance_X * gradient_X.float()
            # =========================

            if t == 0:
                X = X_mean
                E = E_mean
            else:
                model_log_variance_X = self._extract(model_log_variance, t_, X_mean.shape) * torch.ones_like(X_mean)
                X = X_mean + torch.exp(0.5 * model_log_variance_X) * eps.X
                model_log_variance_E = self._extract(model_log_variance, t_, E_mean.shape) * torch.ones_like(E_mean)
                E = E_mean + torch.exp(0.5 * model_log_variance_E) * eps.E
        
        scene_list = []
        for i in range(batch_size):
            n = num_nodes[i]
            node_data = X[i, :n].cpu()
            edge_data = E[i, :n, :n].cpu()
            scene_list.append([node_data, edge_data])

        if return_dense_data or return_input:
            return scene_list, dense_data
        else:
            return scene_list

    def p_mean_variance(self, X_t, E_t, y_t, t, node_mask, condition_feat=None, cond_fn=None, cond_fn_2=None, source_condition=None, source_condition_edge=None, clip_denoised=True):
        batch_size, _, _ = X_t.shape

        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't_int':t}

        ## p_mean_variance
        eps_output = self.forward(noisy_data, node_mask, condition_feat, source_condition, source_condition_edge)

        model_variance, model_log_variance = (self.posterior_variance.to(X_t.device), self.posterior_log_variance_clipped.to(X_t.device))

        x_recon = self._predict_xstart_from_eps(X_t, t=t, eps=eps_output.X)
        e_recon = self._predict_xstart_from_eps(E_t, t=t, eps=eps_output.E)

        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1.0, 1.0) 
            e_recon = torch.clamp(e_recon, -1.0, 1.0) 

        X_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=X_t, t=t)
        E_mean, _, _ = self.q_posterior_mean_variance(x_start=e_recon, x_t=E_t, t=t)

        return X_mean, E_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):

        assert x_t.shape == eps.shape
        return (
                self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def sample_p_zs_given_zt(self, s, t, s_int, t_int, X_t, E_t, y_t, node_mask, condition_feat=None, cond_fn=None, cond_fn_2=None, source_condition=None, source_condition_edge=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = diffusion_utils.sigma_and_alpha_t_given_s(gamma_t,
                                                                                                       gamma_s,
                                                                                                       X_t.size())
        sigma_s = diffusion_utils.sigma(gamma_s, target_shape=X_t.size())
        sigma_t = diffusion_utils.sigma(gamma_t, target_shape=X_t.size())

        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 't_int':t_int}
        eps = self.forward(noisy_data, node_mask, condition_feat, source_condition, source_condition_edge)

        # Compute mu for p(zs | zt).
        mu_X = X_t / alpha_t_given_s - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)) * eps.X
        mu_E = E_t / alpha_t_given_s.unsqueeze(1) - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)).unsqueeze(1) * eps.E
        mu_y = y_t / alpha_t_given_s.squeeze(1) - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)).squeeze(1) * eps.y

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        maximum_noise_step = 50  # 50
        minimum_noise_step = -1
        if (cond_fn is not None) and (s_int < maximum_noise_step) and (s_int > minimum_noise_step):
            variance = torch.square(sigma)
            gradient_X, _, loss_1 = cond_fn(mu_X, mu_E, node_mask)
            new_mu_X = mu_X + variance * gradient_X.float()

            if cond_fn_2 is not None:
                gradient_2_X, _, loss_2 = cond_fn_2(mu_X, mu_E, node_mask)
                new_mu_X = new_mu_X + variance * gradient_2_X.float()

            mu_X = new_mu_X

        # Sample zs given the parameters derived from zt.
        z_s = diffusion_utils.sample_normal(mu_X, mu_E,  mu_y, sigma, node_mask).type_as(mu_X)

        return z_s

    def sample_graph_given_z0(self, X_0, E_0, y_0, node_mask, condition_feat=None, source_condition=None, source_condition_edge=None):
        """ Samples X, E, y ~ p(X, E, y|z0): once the diffusion is done, we need to map the result
        to categorical values.
        """
        zeros = torch.zeros(size=(X_0.size(0), 1), device=X_0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma = diffusion_utils.SNR(-0.5 * gamma_0).unsqueeze(1)
        noisy_data = {'X_t': X_0, 'E_t': E_0, 'y_t': y_0, 't': torch.zeros(y_0.shape[0], 1).type_as(y_0), 't_int': torch.zeros(y_0.shape[0], 1).type_as(y_0)}
        eps0 = self.forward(noisy_data, node_mask, condition_feat, source_condition, source_condition_edge)

        sigma_0 = diffusion_utils.sigma(gamma_0, target_shape=eps0.X.size())
        alpha_0 = diffusion_utils.alpha(gamma_0, target_shape=eps0.X.size())

        pred_X = 1. / alpha_0 * (X_0 - sigma_0 * eps0.X)
        pred_E = 1. / alpha_0.unsqueeze(1) * (E_0 - sigma_0.unsqueeze(1) * eps0.E)
        pred_y = 1. / alpha_0.squeeze(1) * (y_0 - sigma_0.squeeze(1) * eps0.y)

        sampled = diffusion_utils.sample_normal(pred_X, pred_E, pred_y, sigma, node_mask).type_as(pred_X)

        return sampled