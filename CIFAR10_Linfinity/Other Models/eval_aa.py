import os
import shutil
import numpy as np
import argparse
import importlib
import time
import logging

import torch
import torch.nn as nn
import utils_sparse_imagenet
from utils_sparse_imagenet import *

from robustbench.utils import load_model
from robustbench.data import load_imagenet
from torchvision import transforms, datasets
import torch.nn as nn
from robustbench.data import load_imagenet
from utils_sparse_cifar import *
from data import load_cifar10
from huggingface_hub import hf_hub_download

def eval_aa_cifar(device, args, logger, result_sub_dir):
    ########################### Creating model ############################
    model = load_model(model_name=args.model_name, dataset='cifar10', threat_model='Linf')
    add_custom_layer(model, MeanSparse, parent_path='', prev_features=None)
    file_path = os.path.join(args.directory_checkpoint,'%s_WS.pt'%args.model_name)

    if not os.path.exists(file_path):
        cache_dir = 'temp_cache'
        downloaded_path = hf_hub_download(repo_id='MeanSparse/MeanSparse', filename='cifar10/%s_WS.pt'%args.model_name, cache_dir=cache_dir)
        os.makedirs(args.directory_checkpoint, exist_ok=True)
        shutil.copy(downloaded_path, file_path)
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        print(f"Model saved to {args.directory_checkpoint}")
    
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    ########################### Loading Data ############################
    x_test, y_test = load_cifar10(n_examples=10000)
    print('Dataset Size:', len(y_test))
    ############################ Sparsifying ############################### 
    if args.flag_calculate_acc:
        results = calculate_accuracy(model, x_test.to(device), y_test.to(device), batch_size=args.batch_size)
        logger.info(f"{args.model_name + ' Original Model'} Clean Accuracy: {results['top1'].item():.2f}%")

    for name, module in model.named_modules():
        if isinstance(module, MeanSparse):
            module.threshold.data.fill_(args.threshold)
    
    if args.flag_calculate_acc:
        results = calculate_accuracy(model, x_test.to(device), y_test.to(device), batch_size=args.batch_size)
        logger.info(f"{args.model_name + ' Sparsified Model'} Clean Accuracy: {results['top1'].item():.2f}%")

    ########################### Applying auto attack ############################
    results = AutoAttack_Wrapper(model, device, x_test, y_test, args, logger, sub_dir_path=result_sub_dir,
                                start_batch=args.start_batch, end_batch=args.end_batch, checkpoint_path=args.ckpt, 
                                distance=args.distance, epsilon=args.epsilon, attack_type=args.attacks, batch_size=args.batch_size)

    logger.info('AutoAttack Results: ' + str(results))

def eval_aa_imagenet(device, args, logger, result_sub_dir):
    ########################### Creating model ############################
    model = load_model(model_name=args.model_name, dataset='imagenet', threat_model='Linf')
    utils_sparse_imagenet.__dict__[args.add_custom_layer](model, utils_sparse_imagenet.__dict__[args.module_name])
    file_path = os.path.join(args.directory_checkpoint, '%s_WS.pt'%args.model_name)
    
    if not os.path.exists(file_path):
        cache_dir = 'temp_cache'
        downloaded_path = hf_hub_download(repo_id='MeanSparse/MeanSparse', filename='imagenet/%s_WS.pt'%args.model_name, cache_dir=cache_dir)
        os.makedirs(args.directory_checkpoint, exist_ok=True)
        shutil.copy(downloaded_path, file_path)
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        print(f"Model saved to {args.directory_checkpoint}")
    
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    ########################### Loading Data ############################
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode("bicubic")),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    x_test, y_test = load_imagenet(data_dir=args.data_dir, transforms_test=transform)
    print('Dataset size:', len(y_test))
   ############################ Sparsifying ###############################    
    if args.flag_calculate_acc:
        results = calculate_accuracy(model, x_test.to(device), y_test.to(device), batch_size=args.batch_size)
        logger.info(f"{args.model_name + ' Original Model'} Clean Accuracy: {results['top1'].item():.2f}%")

    for name, module in model.named_modules():
        if isinstance(module, utils_sparse_imagenet.__dict__[args.module_name]):
            module.threshold.data.fill_(args.threshold)

    if args.flag_calculate_acc:
        results = calculate_accuracy(model, x_test.to(device), y_test.to(device), batch_size=args.batch_size)
        logger.info(f"{args.model_name + ' Sparsified Model'} Clean Accuracy: {results['top1'].item():.2f}%")
        
    ########################### Applying auto attack ############################
    results = AutoAttack_Wrapper(model, device, x_test, y_test, args, logger, sub_dir_path=result_sub_dir,
                                start_batch=args.start_batch, end_batch=args.end_batch, checkpoint_path=args.ckpt, 
                                distance=args.distance, epsilon=args.epsilon, attack_type=args.attacks, batch_size=args.batch_size)

    logger.info('AutoAttack Results: ' + str(results))
