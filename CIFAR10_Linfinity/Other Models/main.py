import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import importlib
import time
import logging

import eval_aa
from eval_aa import *

def main():
    torch.cuda.empty_cache()
    print('GPU type:', torch.cuda.get_device_name(0))
    print('GPU Count:', torch.cuda.device_count())
    ########################### Initilization ############################
    parser = argparse.ArgumentParser(description="Applying Auto Attack")
    parser.add_argument("--configs", type=str, default="./configs/cifar-10/configs_Peng2023Robust_RaWideResNet-70-16.yml")
    parser.add_argument("--exp-name", type=str, default="temp")
    parser.add_argument("--start_batch", default=0, type=int)
    parser.add_argument("--end_batch", default=None, type=int)
    parser.add_argument("--attacks", default=None, type=list)

    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--flag_calculate_acc", default=True, type=bool)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint path")
    parser.add_argument("--print-freq", type=int, default=1)
    args = update_args(parser.parse_args())

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
    ############################ Evaluation ############################
    eval_aa.__dict__[args.eval](device, args, logger, result_sub_dir)
    
if __name__ == "__main__":
    main()
