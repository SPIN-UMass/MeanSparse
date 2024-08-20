import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import importlib
import time
import logging
from robustbench.utils import load_model
from utils_sparse import *
from data import load_cifar10
from MeanSparse_robustarch_wide_resnet import NormalizedWideResNet
from MeanSparse_robustarch_wide_resnet import MeanSparse


def main():
    torch.cuda.empty_cache()
    print('GPU type:', torch.cuda.get_device_name(0))
    print('GPU Count:', torch.cuda.device_count())
    ########################### Initilization ############################
    parser = argparse.ArgumentParser(description="Applying Auto Attack")
    parser.add_argument("--exp-name", type=str, default="AA")
    parser.add_argument("--start_batch", default=0, type=int)
    parser.add_argument("--end_batch", default=100, type=int)

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--ckpt", type=str, help="checkpoint path for pretrained classifier")
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--flag_calculate_acc", type=bool, default=True)
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    result_sub_dir = os.path.join(args.results_dir, args.exp_name)
    create_subdirs(result_sub_dir)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    logger.addHandler(
        logging.FileHandler(os.path.join(result_sub_dir, "Results.log"), "a")
    )
    logger.info(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    ########################### Creating model ############################
    name_model, threshold ='Peng2023Robust', 0.25 #Rank 1
    directory_WS = "models_WS"

    if not os.path.exists(directory_WS):
        os.makedirs(directory_WS, exist_ok=True)

    model_original = load_model(model_name=name_model, dataset='cifar10', threat_model='Linf')
    model = NormalizedWideResNet(
        mean = (0.4914, 0.4822, 0.4465),
        std = (0.2471, 0.2435, 0.2616),
        stem_width = 96,
        depth = [30, 31, 10],
        stage_width = [216, 432, 864],
        groups = [1, 1, 1],
        activation_fn = torch.nn.modules.activation.SiLU,
        se_ratio = 0.25,
        se_activation = torch.nn.modules.activation.ReLU,
        se_order = 2,
        num_classes = 10,
        padding = 0,
        num_input_channels = 3
    )

    original_state_dict = model_original.state_dict()
    new_state_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in original_state_dict.items() if k in new_state_dict}
    new_state_dict.update(pretrained_dict)
    model.load_state_dict(new_state_dict)
    model.to(device)

    file_path = os.path.join(directory_WS, '%s_WS.pt'%name_model)
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint)
    model.to(device)

    for name, module in model.named_modules():
        if isinstance(module, MeanSparse):
            module.threshold.data.fill_(threshold)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    ########################### Loading Data ############################
    x_test, y_test = load_cifar10(n_examples=10000)
    print('Dataset Size:', len(y_test))
    ########################### Applying auto attack ############################
    torch.save(model.state_dict(), file_path)
    if args.flag_calculate_acc:
        results = calculate_accuracy(model, x_test.to(device), y_test.to(device), batch_size=args.batch_size)
        print('MeanSparse Model Clean Accuracy:', results)

    ckpt_path = ''
    results = AutoAttack_Wrapper(model, device, x_test, y_test, args, logger, sub_dir_path=result_sub_dir, start_batch=args.start_batch,
                                end_batch=args.end_batch, checkpoint_path=None, attack_type=None, batch_size=args.batch_size)

    logger.info('Results: ' + str(results))

if __name__ == "__main__":
    main()
