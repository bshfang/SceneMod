import os
import os.path as osp
from tqdm import tqdm
import yaml
import time
import socket
import argparse
import numpy as np
import json

import matplotlib.pyplot as plt

import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer, CLIPTextModel

from datasets import build_dataset

from models.diffusion_scene_graph import DiffusionSG

from utils.training_utils import load_config


def train():
    parser = argparse.ArgumentParser(
        description="Train a generative model on bounding boxes"
    )

    parser.add_argument('--config', default='SceneMod/config/bedroom_stage1_mod.yaml', type=str, help='Config files')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--room_type', type=str, default='bedroom', help='room_type')
    parser.add_argument('--ddp', action='store_true', default=False)
    parser.add_argument("--local_rank", type=int, default=os.environ.get('LOCAL_RANK', 0))
    args = parser.parse_args()

    # === device === 
    if args.ddp:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        device = torch.device('cuda', args.local_rank)
        world_size = torch.distributed.get_world_size()
        args.world_size = world_size
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(args.local_rank)

    # === config ===
    configs = load_config(args.config)

    # === log ===
    if args.local_rank == 0:
        log_path = os.path.join('logs', configs['tag'])
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        save_dir = os.path.join(
            log_path,
            time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname())
        writer = SummaryWriter(save_dir)
        with open(osp.join(save_dir, 'config.yaml'), mode='w') as f:
            yaml.dump(configs, f)
    else:
        save_dir, writer = None, None

    # === Dataset Setup ===
    train_dataset = build_dataset(configs, split=['train'])
    print("train_dataset", len(train_dataset))  # 
    val_dataset = build_dataset(configs, split=['val'], inference=True)
    print("val_dataset", len(val_dataset))

    if args.ddp:
        sampler_train = DistributedSampler(train_dataset)
        sampler_val = DistributedSampler(val_dataset, shuffle=False)

        train_loader = DataLoader(train_dataset,
                                batch_size=configs['batch_size'],
                                num_workers=configs['num_workers'],
                                collate_fn=train_dataset.collate_batch_train,
                                pin_memory=True,
                                sampler=sampler_train,
                                drop_last=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=configs['batch_size'],
                                num_workers=configs['num_workers'],
                                collate_fn=val_dataset.collate_batch_train,
                                pin_memory=True,
                                sampler=sampler_val)
    else:
        train_loader = DataLoader(train_dataset,
                                    batch_size=configs['batch_size'],
                                    num_workers=configs['num_workers'],
                                    collate_fn=train_dataset.collate_batch_train,
                                    pin_memory=True,
                                    shuffle=True,
                                    drop_last=True)
        val_loader = DataLoader(val_dataset,
                                    batch_size=configs['batch_size'],
                                    num_workers=configs['num_workers'],
                                    collate_fn=val_dataset.collate_batch_train,
                                    pin_memory=True,
                                    shuffle=False,
                                    drop_last=False)
    print('train loader:', len(train_loader), ', val loader:', len(val_loader))
    
    # === model setup ===
    model = DiffusionSG(configs).cuda(args.local_rank)

    # === load from checkpoint ===
    if configs['load_model'] != "":
        if not args.ddp:
            model.load_state_dict(torch.load(configs['load_model']))
        else:
            model.load_state_dict(torch.load(configs['load_model'], map_location=lambda storage, loc: storage.cuda(args.local_rank)))
            
        print("Model loaded from {}".format(configs['load_model']))

    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args.local_rank], 
                                                          output_device=args.local_rank, 
                                                          find_unused_parameters=True)

    # === text encoder ===
    pretrained_model_path = "stable-diffusion-v1-5"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    text_encoder.to(device)
    text_encoder.requires_grad_(False)    

    # === optimizer & scheduler ===
    optimizer = optim.AdamW(model.parameters(), lr=configs['optimizer']['lr'],
                           weight_decay=configs['optimizer']['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               milestones=configs['scheduler']['milestones'],
                                               gamma=configs['scheduler']['gamma'])
    
    # === Training ===
    epochs = configs['epochs']
    itrs = 0

    best_loss_val = 100.
    best_epoch_val = 0

    for epoch in range(epochs):
        if args.local_rank == 0:
            print('epoch:', epoch)

            total_loss = 0.0
            total_batches = 0
            all_losses_dict = {}

        if args.ddp:
            train_loader.sampler.set_epoch(epoch)

        for i, batch_data in enumerate(train_loader):
            itrs += 1

            use_text = True
            if use_text:
                text_encoder.eval()
                prompt = batch_data["prompts"]
                prompt_ids = tokenizer(prompt, truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
                text_features = text_encoder(prompt_ids.to(device))[0]
            
            model.train()

            model.zero_grad()
            optimizer.zero_grad()

            for k, v in batch_data.items():
                if k in ['source_data', 'target_data', 'floor_bounds', 'mod_masks', 'source_node_condition']:
                    batch_data[k] = batch_data[k].to(device)
            if args.ddp:
                loss, loss_dict = model.module.get_loss(batch_data, condition_feat=text_features)
            else:
                loss, loss_dict = model.get_loss(batch_data, condition_feat=text_features)

            if args.local_rank == 0:
                total_loss += loss.item()
                total_batches += 1
                for key, value in loss_dict.items():
                    if key not in all_losses_dict:
                        all_losses_dict[key] = value
                    else:
                        all_losses_dict[key] += value

            loss.backward()
            optimizer.step()
            scheduler.step()

        # write to log
        if args.local_rank == 0:
            for key, value in all_losses_dict.items():
                average = value / total_batches
                writer.add_scalar(f'train_{key}_epoch', average, global_step=epoch)
                print(f"{key}: {average}")

        # === validation ===
        if epoch % configs['eval_freq'] == 0:
            if args.local_rank == 0:
                print("=== validation start ===")
                val_total_loss = 0.
                val_loss_dict = {}
                val_batches = 0

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    text_encoder.eval()
                    prompt = batch_data["prompts"]
                    prompt_ids = tokenizer(prompt, truncation=True, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
                    text_features = text_encoder(prompt_ids.to(device))[0]

                    for k, v in batch_data.items():
                        if k in ['source_data', 'target_data', 'floor_bounds', 'mod_masks', 'source_node_condition']:
                            batch_data[k] = batch_data[k].to(device)
                    
                    if args.ddp:
                        loss, loss_dict = model.module.get_loss(batch_data, condition_feat=text_features)
                    else:
                        loss, loss_dict = model.get_loss(batch_data, condition_feat=text_features)
                    
                    if args.local_rank == 0:
                        print(loss_dict)
                        val_total_loss += loss.item()
                        for key, value in loss_dict.items():
                            if key not in val_loss_dict:
                                val_loss_dict[key] = value
                            else:
                                val_loss_dict[key] += value
                        val_batches += 1

                
            if args.local_rank == 0:

                for key in val_loss_dict:
                    val_loss_dict[key] = val_loss_dict[key] / val_batches
                for key, value in val_loss_dict.items():
                    writer.add_scalar('val_'+key, value, global_step=epoch)

                val_average_loss = val_total_loss / val_batches
                if val_average_loss < best_loss_val:
                    best_loss_val = val_average_loss
                    best_epoch_val = epoch
                    if args.ddp:
                        torch.save(model.module.state_dict(), os.path.join(save_dir, 'best_val_model.pth'))
                    else:
                        torch.save(model.state_dict(), os.path.join(save_dir, 'best_val_model.pth'))
                    print(f"New val best model saved at epoch {best_epoch_val} with loss {best_loss_val}")
            
                print("=== validation end ===")


if __name__=='__main__':
    train()