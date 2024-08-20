import numpy as np
import os
import torch
import argparse
import importlib
import time
import logging
import torch.nn as nn
from torchvision import transforms, datasets
from robustbench.utils import load_model
from robustbench.data import load_imagenet
from utils_sparse import *
from MeanSparse_swin_transformer import swin_large_patch4_window7_224_with_MeanSparse


def main():
    torch.cuda.empty_cache()
    print('GPU type:', torch.cuda.get_device_name(0))
    print('GPU Count:', torch.cuda.device_count())
    ########################### Initilization ############################
    parser = argparse.ArgumentParser(description="Applying Auto Attack")
    parser.add_argument("--exp-name", type=str, default="AA__Imagenet_Swin_L")
    parser.add_argument("--start_batch", default=0, type=int)
    parser.add_argument("--end_batch", default=100, type=int)

    parser.add_argument("--batch_size", default=64, type=int)
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
    name_model, threshold ='Liu2023Comprehensive_Swin-L', 0.10 #Rank 1
    directory_WS = "models_WS"

    if not os.path.exists(directory_WS):
        os.makedirs(directory_WS, exist_ok=True)

    model_original = load_model(model_name=name_model, dataset='imagenet', threat_model='Linf')
    model =  swin_large_patch4_window7_224_with_MeanSparse(pretrained=False, pretrained_cfg=None,pretrained_cfg_overlay=None)
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
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode("bicubic")),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    x_test, y_test = load_imagenet(data_dir='/work/pi_ahoumansadr_umass_edu/mteymoorianf/Research/MeanSparse/Dataset/ImageNet-1k', transforms_test=transform)
    print('Dataset Size:', len(y_test))
    ########################### Applying auto attack ############################
    torch.save(model.state_dict(), file_path)
    if args.flag_calculate_acc:
        results = calculate_accuracy(model, x_test.to(device), y_test.to(device), batch_size=args.batch_size)
        print('MeanSparse Model Clean Accuracy:', results)

    results = AutoAttack_Wrapper(model, device, x_test, y_test, args, logger, sub_dir_path=result_sub_dir, start_batch=args.start_batch,
                                end_batch=args.end_batch, checkpoint_path=None, attack_type=None, batch_size=args.batch_size)

    logger.info('Results: ' + str(results))

if __name__ == "__main__":
    main()
