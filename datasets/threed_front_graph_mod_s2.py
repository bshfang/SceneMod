import sys, os
import os.path as osp
import numpy as np
import yaml
import pickle
import random
import json
from collections import OrderedDict

from PIL import Image

import torch
import torch.utils.data as data
import torch.nn.functional as F

from torch_geometric.data import Batch, Data

from datasets.utils import scale, descale


class Threed_Front_Graph_Mod_Stage_TWO(data.Dataset):
    """
    Return the scene graph of one sample.
    w/o text description
    """
    def __init__(self,
                 data_cfg,
                 splits=['train'],
                 inference=False):
        
        self._base_dir = data_cfg['dataset_directory']
        self.config = data_cfg

        self._parse_train_stats(data_cfg["train_stats"])

        self.inference = inference

        all_samples = []
        for s in splits:
            if s == 'train':
                scene_list_file = data_cfg["train_file"]
                with open(scene_list_file, 'r') as file:
                    loaded_list = json.load(file)
                all_samples += loaded_list
            elif s == 'val':
                scene_list_file = data_cfg["val_file"]
                with open(scene_list_file, 'r') as file:
                    loaded_list = json.load(file)
                all_samples += loaded_list
            elif s == 'test':
                scene_list_file = data_cfg["test_file"]
                with open(scene_list_file, 'r') as file:
                    loaded_list = json.load(file)
                all_samples += loaded_list

        self._path_to_samples = all_samples

        self.scale_input_feat = data_cfg.get("scale_input_feat", True)
        self.scale_input_class = data_cfg.get("scale_input_class", True)

        self.translation_dim = data_cfg.get("translation_dim", 3)
        self.size_dim = data_cfg.get("size_dim", 3)
        self.rotation_dim = data_cfg.get("rotation_dim", 2)
        self.cat_dim = data_cfg.get("cat_dim", 23)
        self.objfeat_dim = data_cfg.get("objfeat_dim", 0)

        self.edge_cat_dim = data_cfg.get("edge_cat_dim", 0)
        self.use_extra_edge_attri = data_cfg.get("extra_edge_attri", False)
        self.edge_attr_dim = data_cfg.get("edge_attr_dim", 0)

        self.bbox_dim = self.translation_dim + self.size_dim + self.rotation_dim
        # self.node_feat_dim =  + 
        self.source_node_feat_dim = self.cat_dim + self.objfeat_dim
        self.target_node_feat_dim = self.translation_dim + self.size_dim + self.rotation_dim

        self.use_mod_mask = data_cfg.get("use_mod_mask", False)

        self.full_node = data_cfg.get("full_node", False) 
        self.max_node = data_cfg.get("max_node", 13) 
        
        self.load_prompt = data_cfg.get("load_prompt", False)

        self.room_type = data_cfg.get("room_type", 'bedroom')

        # floor bound
        self.load_floor_bound = data_cfg.get("load_floor_bound", False)

        # v5 setting: use visual-combined feature
        self.use_new_vis_objfeat = data_cfg.get("use_new_vis_objfeat", False)
        if self.use_new_vis_objfeat:
            with open(data_cfg['new_objfeat_path'], 'rb') as file:
                self.new_objfeats = pickle.load(file)

            print("len(self.new_objfeats)", len(self.new_objfeats))

            self.new_objfeat_bound = [-4.626937, 4.649943]

    def __len__(self):
        return len(self._path_to_samples)  #  * 10
    
    def _parse_train_stats(self, train_stats):
        with open(os.path.join(self._base_dir, train_stats), "r") as f:
            train_stats = json.load(f)
        self._centroids = train_stats["bounds_translations"]
        self._centroids = (
            np.array(self._centroids[:3]),
            np.array(self._centroids[3:])
        )
        self._sizes = train_stats["bounds_sizes"]
        self._sizes = (np.array(self._sizes[:3]), np.array(self._sizes[3:]))
        self._angles = train_stats["bounds_angles"]
        self._angles = (np.array(self._angles[0]), np.array(self._angles[1]))
        if "bounds_objfeats" in train_stats.keys():
            self._objfeats = train_stats["bounds_objfeats"]
            self._objfeats = ( np.array([self._objfeats[0]]), np.array([self._objfeats[1]]), np.array([self._objfeats[2]]) )
        else:
            self._objfeats = ( np.array([1]), np.array([-1]), np.array([1]) )

        if "bounds_objfeats_32" in train_stats.keys():
            self._objfeats_32 = train_stats["bounds_objfeats_32"]
            self._objfeats_32 = ( np.array([self._objfeats_32[0]]), np.array([self._objfeats_32[1]]), np.array([self._objfeats_32[2]]) )
        else:
            self._objfeats_32 = ( np.array([1]), np.array([-1]), np.array([1]) )


        self._class_labels = train_stats["class_labels"]
        self._object_types = train_stats["object_types"]
        self._class_frequencies = train_stats["class_frequencies"]
        self._class_order = train_stats["class_order"]
        self._count_furniture = train_stats["count_furniture"]
        self._max_length = self.config.get('max_length', 12)

        self.room_bound_scale_low = np.zeros(4)
        self.room_bound_scale_high = np.array([self.bounds['translations'][1][2]-self.bounds['translations'][0][2], 
                                            self.bounds['translations'][1][0]-self.bounds['translations'][0][0], 
                                            self.bounds['translations'][1][2]-self.bounds['translations'][0][2], 
                                            self.bounds['translations'][1][0]-self.bounds['translations'][0][0]])

    @property
    def bounds(self):
        return {
            "translations": self._centroids,
            "sizes": self._sizes,
            "angles": self._angles,
            "objfeats": self._objfeats,
            "objfeats_32": self._objfeats_32,
        }
    
    def __getitem__(self, idx):
        random_id = idx
        
        source_file = os.path.join(self._path_to_samples[random_id], "boxes.npz")
        D_source = dict(np.load(source_file, allow_pickle=True))
        target_file = os.path.join(self._path_to_samples[random_id], "boxes_target.npz")
        D_target = dict(np.load(target_file, allow_pickle=True))

        # load scene graph info
        source_graph_file = os.path.join(self._path_to_samples[random_id], "scene_graph_source.json")
        with open(source_graph_file, 'r') as file:
            source_graph_info = json.load(file)
        target_graph_file = os.path.join(self._path_to_samples[random_id], "scene_graph_target.json")
        with open(target_graph_file, 'r') as file:
            target_graph_info = json.load(file)

        source_condition_scene_graph, target_scene_graph, source_condition_node_bbox, mod_mask = self.construct_scene_graph(D_source, D_target, source_graph_info, target_graph_info, idx)

        # text instruction
        if self.load_prompt:
            des_file = os.path.join(self._path_to_samples[random_id], 'modify_description.txt')
            with open(des_file, 'r') as file:
                prompt = file.read()
        else:
            if mod_type in ['add', 'remove', 'move']:
                prompt, _ = self.generate_description_mod_single(D_source, D_target)
            else:
                mod_type = 'arrange'
                prompt, _ = self.generate_description_mod_scene(D_source, D_target)


        if self.load_floor_bound:
            floor_bound = D_source['room_bound']
            # print("floor_bound", floor_bound)  # x_min, x_max, y_min, y_max

            # rescale floor_bound
            floor_bound[0:2] = scale(floor_bound[0:2], self.bounds['translations'][0][2], self.bounds['translations'][1][2])
            floor_bound[2:4] = scale(floor_bound[2:4], self.bounds['translations'][0][0], self.bounds['translations'][1][0])
            floor_bound = torch.from_numpy(floor_bound).float()
            # print("floor_bound", floor_bound)  # x_min, x_max, y_min, y_max

            output_dict = {
                'source_data': source_condition_scene_graph,
                'target_data': target_scene_graph,
                'source_node_bbox': source_condition_node_bbox,
                'floor_bound': floor_bound.unsqueeze(dim=0),
                'source_file': source_file,
                'prompt': prompt
            }
        else:
            output_dict = {
                'source_data': source_condition_scene_graph,
                'target_data': target_scene_graph,
            }

        if self.use_mod_mask:
            output_dict.update({'mod_mask':mod_mask})
        
        # add the mod type
        output_dict.update({'mod_type':mod_type})

        return output_dict
    
    def construct_scene_graph(self, room_data_source, room_data_target, scene_graph_source, scene_graph_target, idx):
        """
        generate the PyG graph, with no empty node & no room node
        for updated scene graph: scene_graph.json

        return condition graph & target graph & origin_location & mod_mask
        condition graph: 
            x: cat, objfeat (from target scene)
            e: spatial cat, angle cat, parallel cat (from target scene)
        target graph:
            x: trans, size, rotate (from target scene)
            e: angle, distance (from target scene)

        origin_location:
            x: trans, size, rotate (from source scene)
        mod_mask: compare source & target translation

        """

        bounds = self.bounds

        obj_indexes_1 = list(room_data_source['obj_indexes'])
        obj_indexes_2 = list(room_data_target['obj_indexes'])

        graph_obj_num = max(max(obj_indexes_1), max(obj_indexes_2)) + 1

        if self.use_new_vis_objfeat:
            obj_jids_1 = room_data_source['jids']
            obj_jids_2 = room_data_target['jids']

        x_source = torch.zeros(((graph_obj_num, self.source_node_feat_dim))).float()  # cat & objfeat from target scene
        x_target = torch.zeros(((graph_obj_num, self.target_node_feat_dim))).float()  # trans & size & rot from target scene
        empty_node_indexes = []
        for i in range(graph_obj_num):
            if i in obj_indexes_2:
                obj_idx = obj_indexes_2.index(i)
                obj_trans = room_data_target['translations'][obj_idx]
                obj_size = room_data_target['sizes'][obj_idx]
                obj_angle = room_data_target['angles'][obj_idx]
                obj_cat = room_data_target['class_labels'][obj_idx]

                if (obj_cat == -1.0).any():
                    new_obj_cat = np.zeros_like(obj_cat)
                    new_obj_cat[obj_cat > 0] = 1.0
                    obj_cat = new_obj_cat

                if self.objfeat_dim == 32:
                    if self.use_new_vis_objfeat:
                        obj_feat = self.new_objfeats[obj_jids_2[obj_idx]]
                    else:
                        obj_feat = room_data_target['objfeats_32'][obj_idx]

                if self.scale_input_feat:
                    obj_trans = scale(obj_trans, bounds['translations'][0], bounds['translations'][1])
                    obj_size = scale(obj_size, bounds['sizes'][0], bounds['sizes'][1])
                    obj_angle = np.concatenate([np.cos(obj_angle), np.sin(obj_angle)], axis=-1)

                    if self.objfeat_dim > 0:
                        if self.objfeat_dim == 32:
                            if self.use_new_vis_objfeat:
                                obj_feat = scale(
                                    obj_feat, self.new_objfeat_bound[0], self.new_objfeat_bound[1]
                                )
                            else:
                                obj_feat = scale(
                                    obj_feat, bounds['objfeats_32'][1], bounds['objfeats_32'][2]
                                )

                x_target[i, :self.translation_dim] = torch.from_numpy(obj_trans)
                x_target[i, self.translation_dim:self.translation_dim+self.size_dim] = torch.from_numpy(obj_size)
                x_target[i, self.translation_dim+self.size_dim:self.translation_dim+self.size_dim+self.rotation_dim] = torch.from_numpy(obj_angle)

                x_source[i, :self.cat_dim-2] = torch.from_numpy(obj_cat[:-2])
                if self.scale_input_class:
                    x_source[i, :self.cat_dim] = x_source[i, :self.cat_dim] * 2 - 1.0
                if self.objfeat_dim > 0:
                    x_source[i, self.cat_dim:self.cat_dim+self.objfeat_dim] = torch.from_numpy(obj_feat)
            else:
                empty_node_indexes.append(i)
                x_source[i, self.cat_dim-1] = 1  # empty class
                if self.scale_input_class:
                    x_source[i,:self.cat_dim] = x_source[i, :self.cat_dim] * 2 - 1.0
                
        if self.full_node:            
            # source
            x_empty_s = torch.zeros(((self.max_node - graph_obj_num, self.source_node_feat_dim))).float()  # room node
            x_empty_s[:, self.cat_dim-1] = 1.
            if self.scale_input_class:
                x_empty_s[:, :self.cat_dim] = x_empty_s[:, :self.cat_dim] * 2 - 1.0
            x_source = torch.cat((x_source, x_empty_s), dim=0)

            # target
            x_empty_t = torch.zeros(((self.max_node - graph_obj_num, self.target_node_feat_dim))).float()  # room node
            x_target = torch.cat((x_target, x_empty_t), dim=0)

        # edge
        edge_index = [[], []]
        edge_attr = []
        edge_attr_extra_angle = []
        edge_attr_extra_angle_class = []
        edge_attr_extra_dis = []

        if self.use_extra_edge_attri:
            edge_attr_extra_parallel = []
            edge_attr_extra_facing = []

        source_edges = scene_graph_target['edges']
        source_edge_attributes = scene_graph_target['edge_attributes_directed']

        if self.use_extra_edge_attri:
            source_edge_attributes_angle = scene_graph_target['edge_attributes_angle']
            source_edge_attributes_angle_class = scene_graph_target['edge_attributes_angle_class']
            source_edge_attributes_parallel = scene_graph_target['edge_attributes_parallel']
            source_edge_attributes_facing = scene_graph_target['edge_attributes_facing']

        for e_i in range(len(source_edges)):
            edge = source_edges[e_i]
            if (edge[0] not in obj_indexes_2) or (edge[1] not in obj_indexes_2):
                continue

            edge_attri = source_edge_attributes[e_i]

            edge_index[0].append(obj_indexes_2.index(edge[0]))
            edge_index[1].append(obj_indexes_2.index(edge[1]))

            edge_attr.append(edge_attri)
            if self.use_extra_edge_attri:
                edge_attri_angle = source_edge_attributes_angle[e_i]
                edge_attri_angle_class = source_edge_attributes_angle_class[e_i]
                edge_attr_extra_dis.append(np.abs(x_target[edge[0], :self.translation_dim] - x_target[edge[1], :self.translation_dim]))

                edge_attr_extra_angle.append(np.array([np.cos(edge_attri_angle), np.sin(edge_attri_angle)]))
                edge_attr_extra_angle_class.append(edge_attri_angle_class)

                edge_attr_extra_parallel.append(source_edge_attributes_parallel[e_i])
                edge_attr_extra_facing.append(source_edge_attributes_facing[e_i])

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        edge_attr_one_hot = F.one_hot(edge_attr, self.edge_cat_dim).float()
        if self.use_extra_edge_attri:
            edge_attr_extra_dis = torch.from_numpy(np.vstack(edge_attr_extra_dis)).float()

            edge_attr_extra_angle = torch.from_numpy(np.vstack(edge_attr_extra_angle)).float()

            edge_attri_angle_class = torch.tensor(edge_attr_extra_angle_class, dtype=torch.long)
            edge_attr_extra_angle_class_one_hot = F.one_hot(edge_attri_angle_class, 4).float()  # 4: the angle class number

            edge_attr_extra_parallel = torch.tensor(edge_attr_extra_parallel)
            edge_attr_extra_facing = torch.tensor(edge_attr_extra_facing)

            n = len(edge_attr_extra_parallel)
            edge_attr_extra_attr = torch.zeros((n, 3), dtype=torch.float)

            edge_attr_extra_attr[:, 0] = edge_attr_extra_parallel
            edge_attr_extra_attr[:, 1] = edge_attr_extra_facing
            edge_attr_extra_attr[:, 2] = torch.logical_and(torch.logical_not(edge_attr_extra_parallel), torch.logical_not(edge_attr_extra_facing))
        
        y = torch.tensor([[graph_obj_num]], dtype=torch.float)

        if self.scale_input_class:
            edge_attr_one_hot = edge_attr_one_hot * 2 - 1.0
            edge_attr_extra_angle_class_one_hot = edge_attr_extra_angle_class_one_hot * 2 - 1.0

            if self.use_extra_edge_attri:
                edge_attr_extra_attr = edge_attr_extra_attr * 2 - 1.0

        all_edge_attr_source = torch.cat([edge_attr_one_hot, edge_attr_extra_angle_class_one_hot, edge_attr_extra_attr], dim=1)
        all_edge_attr_target = torch.cat([edge_attr_extra_dis, edge_attr_extra_angle], dim=1)

        ## get origin bbox from source data
        x_source_bbox = torch.zeros(((graph_obj_num, self.target_node_feat_dim))).float()
        empty_node_indexes = []
        for i in range(graph_obj_num):
            if i in obj_indexes_1:
                obj_idx = obj_indexes_1.index(i)
                obj_trans = room_data_source['translations'][obj_idx]
                obj_size = room_data_source['sizes'][obj_idx]
                obj_angle = room_data_source['angles'][obj_idx]

                if self.scale_input_feat:
                    obj_trans = scale(obj_trans, bounds['translations'][0], bounds['translations'][1])
                    obj_size = scale(obj_size, bounds['sizes'][0], bounds['sizes'][1])
                    obj_angle = np.concatenate([np.cos(obj_angle), np.sin(obj_angle)], axis=-1)
                x_source_bbox[i, :self.translation_dim] = torch.from_numpy(obj_trans)
                
                x_source_bbox[i, self.translation_dim:self.translation_dim+self.size_dim] = torch.from_numpy(obj_size)
                x_source_bbox[i, self.translation_dim+self.size_dim:self.translation_dim+self.size_dim+self.rotation_dim] = torch.from_numpy(obj_angle)
        if self.full_node:
            # target
            x_empty_t = torch.zeros(((self.max_node - graph_obj_num, self.target_node_feat_dim))).float()  # room node
            x_source_bbox = torch.cat((x_source_bbox, x_empty_t), dim=0)

        if self.use_mod_mask:
            mod_mask = ~torch.all(torch.eq(x_target[:, :self.translation_dim], x_source_bbox[:, :self.translation_dim]), dim=1)
            mod_mask_scale = mod_mask.clone().detach()
            mod_mask_scale = mod_mask_scale.float().unsqueeze(1)
            mod_mask_scale = mod_mask_scale * 2 - 1.0
            x_source_bbox = torch.cat((x_source_bbox, mod_mask_scale), dim=1)

        pyg_scene_graph_source = Data(x=x_source, edge_index=edge_index, edge_attr=all_edge_attr_source, y=y, idx=idx)
        pyg_scene_graph_target = Data(x=x_target, edge_index=edge_index, edge_attr=all_edge_attr_target, y=y, idx=idx)
        return pyg_scene_graph_source, pyg_scene_graph_target, x_source_bbox, mod_mask
    

    def collate_batch_train(self, data_list):
        if False:  # self.inference:
            return Batch.from_data_list(data_list)
        else:
            source_list = []
            target_list = []
            source_node_condition_list = []
            prompts = []
            mod_masks = []
            mod_types = []
            floor_bounds = []

            sample_file_list = []
            for data in data_list:
                source_list.append(data['source_data'])
                target_list.append(data['target_data'])
                source_node_condition_list.append(data['source_node_bbox'])
                prompts.append(data['prompt'])
                if self.use_mod_mask:
                    mod_masks.append(data['mod_mask'].unsqueeze(0))
                
                if self.load_floor_bound:
                    floor_bounds.append(data['floor_bound'])
                
                if 'mod_type' in data:
                    mod_types.append(data['mod_type'])
                if 'source_file' in data:
                    sample_file_list.append(data['source_file'])
            # batch = Batch.from_data_list(data_list)

            batch_output = {
                'source_data': Batch.from_data_list(source_list),
                'target_data': Batch.from_data_list(target_list),
                'source_node_condition': torch.stack(source_node_condition_list, dim=0),
                'prompts': prompts,
            }

            if self.use_mod_mask:
                mod_masks = torch.cat(mod_masks, dim=0)
                batch_output.update({'mod_masks': mod_masks})

            if self.load_floor_bound:
                floor_bounds = torch.cat(floor_bounds, dim=0)
                batch_output.update({'floor_bounds': floor_bounds})

            if len(mod_types) > 0:
                batch_output.update({'mod_types': mod_types})
            if len(sample_file_list) > 0:
                batch_output.update({'source_file': sample_file_list})

            return batch_output
        
    def descale_bound(self, bound):
        
        floor_bound = bound
        # rescale floor_bound
        floor_bound[0:2] = descale(floor_bound[0:2], self.bounds['translations'][0][2], self.bounds['translations'][1][2])
        floor_bound[2:4] = descale(floor_bound[2:4], self.bounds['translations'][0][0], self.bounds['translations'][1][0])
        return floor_bound
        
    def post_process(self, x, input_x):
        """
        if input_x is not None, use its class, size and objfeats
        deocde result of one sample
        x: predicted, trans, size, angle
        input_x: cat, objfeat
        """
        bounds = self.bounds
        obj_info = x[:, :]
        obj_num = x.shape[0]

        if input_x is not None:
            input_obj_info = input_x[:obj_num, :]
    
        room_dict = {}

        room_dict['translations'] = obj_info[:, :self.translation_dim]
        room_dict['sizes'] = obj_info[:, self.translation_dim:self.translation_dim+self.size_dim]
        room_dict['angles'] = obj_info[:, self.translation_dim+self.size_dim:self.translation_dim+self.size_dim+self.rotation_dim]

        room_dict['class_labels'] = input_obj_info[:, :self.cat_dim]
        if self.objfeat_dim == 32:
            room_dict['objfeats_32'] = input_obj_info[:, self.cat_dim:self.cat_dim+self.objfeat_dim]

        for k, v in room_dict.items():
            if k =='angles':
                room_dict[k] = np.arctan2(v[:, 1:2], v[:, 0:1])
            elif k == "objfeats" or k == "objfeats_32":
                if self.use_new_vis_objfeat:
                    room_dict[k] = descale(
                        v, self.new_objfeat_bound[0], self.new_objfeat_bound[1]
                    )
                else:
                    room_dict[k] = descale(
                        v, bounds[k][1], bounds[k][2]
                    )
            elif k in bounds:
                room_dict[k] = descale(
                    v, bounds[k][0], bounds[k][1]
                )
        return room_dict
