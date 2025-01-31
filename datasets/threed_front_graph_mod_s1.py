import sys, os
import os.path as osp
import numpy as np
import yaml
import pickle
import json
from collections import OrderedDict

from PIL import Image

import torch
import torch.utils.data as data
import torch.nn.functional as F

from torch_geometric.data import Batch, Data

from datasets.utils import scale, descale


class Threed_Front_Graph_Mod_Stage_One(data.Dataset):
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
        self.node_feat_dim = self.translation_dim + self.size_dim + self.rotation_dim + self.cat_dim + self.objfeat_dim
        self.node_feat_dim = self.cat_dim + self.objfeat_dim

        self.use_mod_mask = data_cfg.get("use_mod_mask", False)

        self.full_node = data_cfg.get("full_node", True) # True
        self.max_node = data_cfg.get("max_node", 13) 

        self.load_prompt = data_cfg.get("load_prompt", False)

        self.room_type = data_cfg.get("room_type", 'bedroom')

        # floor bound
        self.load_floor_bound = data_cfg.get("load_floor_bound", False)

        # use visual-combined feature
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

        # room bound scale
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
        
        source_graph, target_graph, mod_mask, source_condition_node_bbox, target_node_bbox = self.construct_two_scene_graphs(D_source, D_target, source_graph_info, target_graph_info, idx)

        # text instruction
        if self.load_prompt:
            des_file = os.path.join(self._path_to_samples[random_id], 'modify_description.txt')
            with open(des_file, 'r') as file:
                prompt = file.read()
            prompt, selected_relations = prompt, []
        else:
            if mod_type in ['add', 'remove', 'move']:
                prompt, selected_relations = self.generate_description_mod_single(D_source, D_target)
            else:
                mod_type = 'arrange'
                prompt, selected_relations = self.generate_description_mod_scene(D_source, D_target)

        # floor bound
        if self.load_floor_bound:
            floor_bound = D_source['room_bound']

            # rescale floor_bound
            floor_bound[0:2] = scale(floor_bound[0:2], self.bounds['translations'][0][2], self.bounds['translations'][1][2])
            floor_bound[2:4] = scale(floor_bound[2:4], self.bounds['translations'][0][0], self.bounds['translations'][1][0])
            floor_bound = torch.from_numpy(floor_bound).float()

            output_dict = {
                'source_data': source_graph,
                'target_data': target_graph,
                'floor_bound': floor_bound.unsqueeze(dim=0),
                'source_file': source_file,
                'target_file': target_file,
                'prompt': prompt,
                'gt_rel': selected_relations,
            }

        else:
            output_dict = {
                'source_data': source_graph,
                'target_data': target_graph,
                'prompt': prompt,
                'gt_rel': selected_relations,
            }

        if self.use_mod_mask:
            output_dict.update({'mod_mask':mod_mask})

        if self.inference:
            output_dict.update({'source_node_bbox':source_condition_node_bbox})
            output_dict.update({'target_node_bbox':target_node_bbox})
        
        # add the mod type
        output_dict.update({'mod_type':mod_type})


        return output_dict
    
    def construct_two_scene_graphs(self, room_data_source, room_data_target, scene_graph_source, scene_graph_target, idx):
        """
        generate two PyG graphs, the input source_graph will always have an empty node
        for updated scene graph: scene_graph_source.json, scene_graph_target.json
        """
        
        # get the room boundaries
        # floor_plan_vertices = room_data_source['floor_plan_vertices'] - room_data_source['floor_plan_centroid']
        # floor_min = np.min(floor_plan_vertices, axis=0)
        # floor_max = np.max(floor_plan_vertices, axis=0)
        # x_min, x_max = floor_min[2], floor_max[2]
        # y_min, y_max = floor_min[0], floor_max[0]
        # room_bound = np.array([-x_min, x_max, -y_min, y_max])

        bounds = self.bounds
        graph_obj_num = len(room_data_source['class_labels']) + 1
        # print("graph_obj_num", graph_obj_num)

        # print(room_data_source['obj_indexes'])
        # print(room_data_target['obj_indexes'])
        obj_indexes_1 = list(room_data_source['obj_indexes'])
        obj_indexes_2 = list(room_data_target['obj_indexes'])

        graph_obj_num = max(max(obj_indexes_1), max(obj_indexes_2)) + 1

        if self.use_new_vis_objfeat:
            obj_jids_1 = room_data_source['jids']
            obj_jids_2 = room_data_target['jids']

        # === source scene graph ===
        # node
        x = torch.zeros(((graph_obj_num, self.node_feat_dim))).float()  # room node
        x_trans = torch.zeros(((graph_obj_num, 3))).float()  # to get the mod mask
        x_source_bbox = torch.zeros(((graph_obj_num, 8))).float()  # for source bbox condition
        empty_node_indexes = []
        for i in range(graph_obj_num):
            if i in obj_indexes_1:
                obj_idx = obj_indexes_1.index(i)
                obj_trans = room_data_source['translations'][obj_idx]
                obj_size = room_data_source['sizes'][obj_idx]
                obj_angle = room_data_source['angles'][obj_idx]
                obj_cat = room_data_source['class_labels'][obj_idx]

                if (obj_cat == -1.0).any():
                    new_obj_cat = np.zeros_like(obj_cat)
                    new_obj_cat[obj_cat > 0] = 1.0
                    obj_cat = new_obj_cat

                if self.objfeat_dim == 32:
                    if self.use_new_vis_objfeat:
                        obj_feat = self.new_objfeats[obj_jids_1[obj_idx]]
                    else:
                        obj_feat = room_data_source['objfeats_32'][obj_idx]

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

                x[i, self.bbox_dim:self.bbox_dim+self.cat_dim-2] = torch.from_numpy(obj_cat[:-2])
                if self.scale_input_class:
                    x[i, self.bbox_dim:self.bbox_dim+self.cat_dim] = x[i, self.bbox_dim:self.bbox_dim+self.cat_dim] * 2 - 1.0
                if self.objfeat_dim > 0:
                    x[i, self.bbox_dim+self.cat_dim:self.bbox_dim+self.cat_dim+self.objfeat_dim] = torch.from_numpy(obj_feat)

                x_trans[i, :] = torch.from_numpy(obj_trans)

                # hard code
                x_source_bbox[i, :3] = torch.from_numpy(obj_trans)
                x_source_bbox[i, 3:6] = torch.from_numpy(obj_size)
                x_source_bbox[i, 6:8] = torch.from_numpy(obj_angle)
   
            else:
                empty_node_indexes.append(i)
                x[i, self.translation_dim+self.size_dim+self.rotation_dim+self.cat_dim-1] = 1  # empty class
                if self.scale_input_class:
                    x[i, self.bbox_dim:self.bbox_dim+self.cat_dim] = x[i, self.bbox_dim:self.bbox_dim+self.cat_dim] * 2 - 1.0

        if self.full_node:
            x_empty = torch.zeros(((self.max_node - graph_obj_num, self.node_feat_dim))).float()  # empty node
            x_empty[:, self.bbox_dim+self.cat_dim-1] = 1.
            if self.scale_input_class:
                x_empty[:, self.bbox_dim:self.bbox_dim+self.cat_dim] = x_empty[:, self.bbox_dim:self.bbox_dim+self.cat_dim] * 2 - 1.0
            x = torch.cat((x, x_empty), dim=0)

            x_trans_empty = torch.zeros(((self.max_node - graph_obj_num, 3))).float()  # empty node
            x_trans = torch.cat((x_trans, x_trans_empty), dim=0)

            x_source_empty = torch.zeros(((self.max_node - graph_obj_num, 8))).float()  # room node
            x_source_bbox = torch.cat((x_source_bbox, x_source_empty), dim=0)
        
        # print("x", x.shape)
        # edge
        edge_index = [[], []]
        edge_attr = []
        edge_attr_extra_angle = []
        edge_attr_extra_angle_class = []
        edge_attr_extra_dis = []

        if self.use_extra_edge_attri:
            edge_attr_extra_parallel = []
            edge_attr_extra_facing = []

        source_edges = scene_graph_source['edges']
        source_edge_attributes = scene_graph_source['edge_attributes_directed']
        if self.use_extra_edge_attri:
            source_edge_attributes_angle = scene_graph_source['edge_attributes_angle']
            source_edge_attributes_angle_class = scene_graph_source['edge_attributes_angle_class']
            source_edge_attributes_parallel = scene_graph_source['edge_attributes_parallel']
            source_edge_attributes_facing = scene_graph_source['edge_attributes_facing']
        
        for e_i in range(len(source_edges)):
            edge = source_edges[e_i]
            if (edge[0] not in obj_indexes_1) or (edge[1] not in obj_indexes_1):
                continue
            edge_attri = source_edge_attributes[e_i]

            edge_index[0].append(edge[0])
            edge_index[1].append(edge[1])
            edge_attr.append(edge_attri)
            if self.use_extra_edge_attri:
                edge_attri_angle = source_edge_attributes_angle[e_i]
                edge_attri_angle_class = source_edge_attributes_angle_class[e_i]
                edge_attr_extra_dis.append(np.abs(x[edge[0], :self.translation_dim] - x[edge[1], :self.translation_dim]))

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
        
        y = torch.tensor([[len(room_data_source['class_labels'])]], dtype=torch.float)

        if self.scale_input_class:
            edge_attr_one_hot = edge_attr_one_hot * 2 - 1.0
            edge_attr_extra_angle_class_one_hot = edge_attr_extra_angle_class_one_hot * 2 - 1.0
            if self.use_extra_edge_attri:
                edge_attr_extra_attr = edge_attr_extra_attr * 2 - 1.0
        if not self.use_extra_edge_attri:
            all_edge_attr = edge_attr_one_hot
        else:
            all_edge_attr = torch.cat([edge_attr_one_hot, edge_attr_extra_dis, edge_attr_extra_angle, edge_attr_extra_angle_class_one_hot], dim=1)
            all_edge_attr = torch.cat([all_edge_attr, edge_attr_extra_attr], dim=1)
        
        # stage 1 
        all_edge_attr = torch.cat([edge_attr_one_hot, edge_attr_extra_angle_class_one_hot, edge_attr_extra_attr], dim=1)

        # print("all_edge_attr", all_edge_attr.shape)
        # print(all_edge_attr)
        pyg_scene_graph_source = Data(x=x, edge_index=edge_index, edge_attr=all_edge_attr, y=y, idx=idx)

        # === target scene graph ===
        x_2 = torch.zeros(((graph_obj_num, self.node_feat_dim))).float()  # room node
        x_2_trans = torch.zeros(((graph_obj_num, 3))).float()  # to get the mod mask
        x_target_bbox = torch.zeros(((graph_obj_num, 8))).float()  # for source bbox condition
        empty_node_indexes_2 = []
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
                        obj_feat = room_data_source['objfeats_32'][obj_idx]
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

                x_2[i, self.bbox_dim:self.bbox_dim+self.cat_dim-2] = torch.from_numpy(obj_cat[:-2])
                if self.scale_input_class:
                    x_2[i, self.bbox_dim:self.bbox_dim+self.cat_dim] = x_2[i, self.bbox_dim:self.bbox_dim+self.cat_dim] * 2 - 1.0
                if self.objfeat_dim > 0:
                    x_2[i, self.bbox_dim+self.cat_dim:self.bbox_dim+self.cat_dim+self.objfeat_dim] = torch.from_numpy(obj_feat)

                x_2_trans[i, :] = torch.from_numpy(obj_trans)

                x_target_bbox[i, :3] = torch.from_numpy(obj_trans)
                x_target_bbox[i, 3:6] = torch.from_numpy(obj_size)
                x_target_bbox[i, 6:8] = torch.from_numpy(obj_angle)
   
            else:
                empty_node_indexes_2.append(i)
                x_2[i, self.translation_dim+self.size_dim+self.rotation_dim+self.cat_dim-1] = 1  # empty class
                if self.scale_input_class:
                    x_2[i, self.bbox_dim:self.bbox_dim+self.cat_dim] = x_2[i, self.bbox_dim:self.bbox_dim+self.cat_dim] * 2 - 1.0
                
        # print("x_2", x_2.shape)
        # print(x_2)
        if self.full_node:
            x_2_empty = torch.zeros(((self.max_node - graph_obj_num, self.node_feat_dim))).float()  # empty node
            x_2_empty[:, self.bbox_dim+self.cat_dim-1] = 1.
            if self.scale_input_class:
                x_2_empty[:, self.bbox_dim:self.bbox_dim+self.cat_dim] = x_2_empty[:, self.bbox_dim:self.bbox_dim+self.cat_dim] * 2 - 1.0
            x_2 = torch.cat((x_2, x_2_empty), dim=0)

            x_2_trans_empty = torch.zeros(((self.max_node - graph_obj_num, 3))).float()  # empty node
            x_2_trans = torch.cat((x_2_trans, x_2_trans_empty), dim=0)

            x_target_empty = torch.zeros(((self.max_node - graph_obj_num, 8))).float()  # room node
            x_target_bbox = torch.cat((x_target_bbox, x_target_empty), dim=0)

        # edge
        edge_index = [[], []]
        edge_attr = []
        edge_attr_extra_angle = []
        edge_attr_extra_angle_class = []
        edge_attr_extra_dis = []
        if self.use_extra_edge_attri:
            edge_attr_extra_parallel = []
            edge_attr_extra_facing = []

        target_edges = scene_graph_target['edges']
        target_edge_attributes = scene_graph_target['edge_attributes_directed']
        if self.use_extra_edge_attri:
            target_edge_attributes_angle = scene_graph_target['edge_attributes_angle']
            target_edge_attributes_angle_class = scene_graph_target['edge_attributes_angle_class']

            target_edge_attributes_parallel = scene_graph_target['edge_attributes_parallel']
            target_edge_attributes_facing = scene_graph_target['edge_attributes_facing']
        
        for e_i in range(len(target_edges)):
            edge = target_edges[e_i]
            edge_attri = target_edge_attributes[e_i]

            edge_index[0].append(edge[0])
            edge_index[1].append(edge[1])
            edge_attr.append(edge_attri)
            if self.use_extra_edge_attri:
                edge_attri_angle = target_edge_attributes_angle[e_i]
                edge_attri_angle_class = target_edge_attributes_angle_class[e_i]
                edge_attr_extra_dis.append(np.abs(x_2[edge[0], :self.translation_dim] - x_2[edge[1], :self.translation_dim]))

                edge_attr_extra_angle.append(np.array([np.cos(edge_attri_angle), np.sin(edge_attri_angle)]))
                edge_attr_extra_angle_class.append(edge_attri_angle_class)

                edge_attr_extra_parallel.append(target_edge_attributes_parallel[e_i])
                edge_attr_extra_facing.append(target_edge_attributes_facing[e_i])

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
        
        y = torch.tensor([[len(room_data_source['class_labels'])]], dtype=torch.float)

        if self.scale_input_class:
            edge_attr_one_hot = edge_attr_one_hot * 2 - 1.0
            edge_attr_extra_angle_class_one_hot = edge_attr_extra_angle_class_one_hot * 2 - 1.0

            if self.use_extra_edge_attri:
                edge_attr_extra_attr = edge_attr_extra_attr * 2 - 1.0
        if not self.use_extra_edge_attri:
            all_edge_attr = edge_attr_one_hot
        else:
            all_edge_attr = torch.cat([edge_attr_one_hot, edge_attr_extra_dis, edge_attr_extra_angle, edge_attr_extra_angle_class_one_hot], dim=1)
            all_edge_attr = torch.cat([all_edge_attr, edge_attr_extra_attr], dim=1)
        
        # stage 1 
        all_edge_attr = torch.cat([edge_attr_one_hot, edge_attr_extra_angle_class_one_hot, edge_attr_extra_attr], dim=1)
        
        # print("all_edge_attr", all_edge_attr.shape)
        # print(all_edge_attr)

        if self.use_mod_mask:
            # mod_mask = ~torch.all(torch.eq(x[:, :self.translation_dim], x_2[:, :self.translation_dim]), dim=1)
            
            # print("x_trans", x_trans)
            # print("x_2_trans", x_2_trans)
            mod_mask = ~torch.all(torch.eq(x_trans, x_2_trans), dim=1)
            # print(x_2.shape)
            # print("obj_indexes_1", obj_indexes_1)
            # print(x)
            # print("obj_indexes_2", obj_indexes_2)
            # print(x_2)
            # print(mod_mask)
            # print(torch.sum(mod_mask))
            # print(torch.eq(x[2, :], x_2[2, :]))

            mod_mask_scale = mod_mask.clone().detach()
            mod_mask_scale = mod_mask_scale.float().unsqueeze(1)
            mod_mask_scale = mod_mask_scale * 2 - 1.0
            # print(mod_mask_scale)
            x_2 = torch.cat((x_2, mod_mask_scale), dim=1)
            # print(x_2.shape)

        pyg_scene_graph_target = Data(x=x_2, edge_index=edge_index, edge_attr=all_edge_attr, y=y, idx=idx)
        
        if self.use_mod_mask:
            return pyg_scene_graph_source, pyg_scene_graph_target, mod_mask, x_source_bbox, x_target_bbox
        else:
            return pyg_scene_graph_source, pyg_scene_graph_target, None, x_source_bbox, x_target_bbox

    def collate_batch_train(self, data_list):
        if False:  # self.inference:
            return Batch.from_data_list(data_list)
        else:
            source_list = []
            target_list = []
            prompts = []
            gt_rel_list = []

            mod_masks = []
            floor_bounds = []
            mod_types = []
            sample_file_list = []
            sample_file_list_target = []
            source_node_condition_list = []
            target_node_condition_list = []

            for data in data_list:
                source_list.append(data['source_data'])
                target_list.append(data['target_data'])
                prompts.append(data['prompt'])
                gt_rel_list.append(data['gt_rel'])

                if self.use_mod_mask:
                    mod_masks.append(data['mod_mask'].unsqueeze(0))

                if self.load_floor_bound:
                    floor_bounds.append(data['floor_bound'])

                if 'mod_type' in data:
                    mod_types.append(data['mod_type'])
                if 'source_file' in data:
                    sample_file_list.append(data['source_file'])
                if 'target_file' in data:
                    sample_file_list_target.append(data['target_file'])
                if 'source_node_bbox' in data:
                    source_node_condition_list.append(data['source_node_bbox'])
                if 'target_node_bbox' in data:
                    target_node_condition_list.append(data['target_node_bbox'])

            batch_output = {
                'source_data': Batch.from_data_list(source_list),
                'target_data': Batch.from_data_list(target_list),
                'prompts': prompts,
                'gt_rel': gt_rel_list,
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
            if len(sample_file_list_target) > 0:
                batch_output.update({'target_file': sample_file_list_target})
            if len(source_node_condition_list) > 0:
                batch_output.update({'source_node_condition': torch.stack(source_node_condition_list, dim=0)})
            if len(source_node_condition_list) > 0:
                batch_output.update({'target_node_condition': torch.stack(target_node_condition_list, dim=0)})
            return batch_output
    
    def descale_bound(self, bound):
        
        floor_bound = bound
        # rescale floor_bound
        floor_bound[0:2] = descale(floor_bound[0:2], self.bounds['translations'][0][2], self.bounds['translations'][1][2])
        floor_bound[2:4] = descale(floor_bound[2:4], self.bounds['translations'][0][0], self.bounds['translations'][1][0])
        return floor_bound
        
    def post_process(self, x, input_x, mod_mask=None, source_bbox_data=None, gt_input_x=None):
        """
        if input_x is not None, use its class, size and objfeats
        deocde result of one sample
        x: (n,8)
        input_x: (n,23+32)
        mod_mask: (n,)
        source_bbox_data: (n,8)
        gt_input_x: (n,23+32)
        """
        bounds = self.bounds
        # obj_info = x[1:, :]
        obj_info = x[:, :]
        obj_num = x.shape[0]

        input_obj_info = input_x[:obj_num, :]

        if mod_mask is not None:

            obj_info[~mod_mask, :] = source_bbox_data[~mod_mask, :]
            input_obj_info[~mod_mask, :self.cat_dim+self.objfeat_dim] = gt_input_x[~mod_mask, :self.cat_dim+self.objfeat_dim]


        room_dict = {}

        room_dict['translations'] = obj_info[:, :3]
        room_dict['sizes'] = obj_info[:, 3:6]
        room_dict['angles'] = obj_info[:, 6:8]

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
    
    def process_semantic_output(self, pred_node, pred_edge, threed_future_dataset=None, decode_mod_mask=False):

        condition_node_data = torch.zeros_like(pred_node)

        node_mask = pred_node[:, self.cat_dim - 1] < 0
        # print("node_mask", node_mask)

        pred_class_index = torch.argmax(pred_node[:, :self.cat_dim], dim=1)

        one_hot = torch.zeros_like(pred_node[:, :self.cat_dim])
        one_hot[torch.arange(len(pred_class_index)), pred_class_index] = 1.0
        one_hot  =one_hot * 2 - 1.0
        condition_node_data[:, :self.cat_dim] = one_hot

        # TODO: retrieve nearest obj_feat
        if threed_future_dataset is None:
            condition_node_data[node_mask, self.cat_dim:self.cat_dim+self.objfeat_dim] = pred_node[node_mask, self.cat_dim:self.cat_dim+self.objfeat_dim]
        else:
            for i in range(len(node_mask)):
                if node_mask[i]:
                    # print(i)
                    pred_obj_feat = pred_node[i, self.cat_dim:self.cat_dim+self.objfeat_dim]
                    pred_obj_feat_descale = descale(
                        pred_obj_feat, self.new_objfeat_bound[0], self.new_objfeat_bound[1]
                    )
                    pred_obj_feat_np = pred_obj_feat_descale.detach().numpy()
                    nearest_obj = threed_future_dataset.get_closest_objfeat_to_objfeat(pred_class_index[i], pred_obj_feat_np)

                    if nearest_obj is not None:
                        nearest_obj_feat = nearest_obj.objfeats_32
                        nearest_obj_feat = scale(
                                    nearest_obj_feat, self.new_objfeat_bound[0], self.new_objfeat_bound[1]
                                )
                        nearest_obj_feat = torch.from_numpy(nearest_obj_feat)
                    else:
                        nearest_obj_feat = pred_obj_feat

                    condition_node_data[i, self.cat_dim:self.cat_dim+self.objfeat_dim] = nearest_obj_feat

        ## update the edge feature
        condition_edge_data = torch.zeros_like(pred_edge)

        pred_edge_cat_1 = pred_edge[:, :, :self.edge_cat_dim]
        argmax_indices = pred_edge_cat_1.argmax(dim=-1)
        one_hot_tensor_1 = torch.zeros_like(pred_edge_cat_1)
        one_hot_tensor_1.scatter_(2, argmax_indices.unsqueeze(-1), 1)
        one_hot_tensor_1 = one_hot_tensor_1 * 2 - 1.0
        condition_edge_data[:, :, :self.edge_cat_dim] = one_hot_tensor_1

        pred_edge_cat_2 = pred_edge[:, :, self.edge_cat_dim:self.edge_cat_dim+4]
        argmax_indices = pred_edge_cat_2.argmax(dim=-1)
        one_hot_tensor_2 = torch.zeros_like(pred_edge_cat_2)
        one_hot_tensor_2.scatter_(2, argmax_indices.unsqueeze(-1), 1)
        one_hot_tensor_2 = one_hot_tensor_2 * 2 - 1.0
        condition_edge_data[:, :, self.edge_cat_dim:self.edge_cat_dim+4] = one_hot_tensor_2

        pred_edge_cat_3 = pred_edge[:, :, self.edge_cat_dim+4:self.edge_cat_dim+7]
        argmax_indices = pred_edge_cat_3.argmax(dim=-1)
        one_hot_tensor_3 = torch.zeros_like(pred_edge_cat_3)
        one_hot_tensor_3.scatter_(2, argmax_indices.unsqueeze(-1), 1)
        one_hot_tensor_3 = one_hot_tensor_3 * 2 - 1.0
        condition_edge_data[:, :, self.edge_cat_dim+4:self.edge_cat_dim+7] = one_hot_tensor_3

        # mask edge and then encode no edge
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(1)
        e_mask2 = x_mask.unsqueeze(0)
        condition_edge_data = condition_edge_data * e_mask1 * e_mask2

        no_edge = torch.sum(condition_edge_data, dim=2) == 0

        no_edge_feat = torch.zeros((condition_edge_data.shape[2]), dtype=condition_edge_data.dtype, device=condition_edge_data.device)
        no_edge_feat[0] = 1.0
        no_edge_feat[self.edge_cat_dim+4-1] = 1.0
        no_edge_feat[self.edge_cat_dim+7-1] = 1.0
        no_edge_feat = no_edge_feat * 2 - 1.0

        condition_edge_data[no_edge] = no_edge_feat
        diag = torch.eye(condition_edge_data.shape[0], dtype=torch.bool)
        condition_edge_data[diag] = 0.0
        # print(condition_edge_data[0])

        if decode_mod_mask:
            pred_mod_mask = pred_node[:, - 1] > 0

            return condition_node_data, condition_edge_data, pred_mod_mask
        else:
            return condition_node_data, condition_edge_data