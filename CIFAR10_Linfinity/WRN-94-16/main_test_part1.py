import os
import torch
from robustbench.utils import load_model
from utils_sparse import calculate_accuracy, calculate_statistics, search_fine
import torch.nn as nn
from data import load_cifar10
from MeanSparse_robustarch_wide_resnet import NormalizedWideResNet
import numpy as np

def main() -> None:

    vec_threshold = torch.linspace(0.00,0.05,2)
    print(vec_threshold)

    name_model='Peng2023Robust' #Rank 1
    batch_size = 200
    flag_calculate_acc = False
    directory_WS = "models_WS"
    directory_result = "results"

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Always move to device "cuda:0" first
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    os.makedirs(os.path.join(directory_result, name_model), exist_ok=True)
    os.makedirs(directory_WS, exist_ok=True)

    x_test, y_test = load_cifar10(n_examples=10000)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    x_train, y_train = load_cifar10(n_examples=50000, flag_train=True)
    x_train = x_train.to(device)
    y_train = y_train.to(device)

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

    if flag_calculate_acc:
        results = calculate_accuracy(model, x_test, y_test, batch_size=512)
        print('MeanSparse Model Accuracy:', results)

    if not os.path.exists(os.path.join(directory_WS, '%s_WS.pt'%name_model )):
        calculate_statistics(model, x_train, y_train, batch_size=512)
        torch.save(model.state_dict(), os.path.join(directory_WS, '%s_WS.pt'%name_model))
    else:
        file_path = os.path.join(directory_WS, '%s_WS.pt'%name_model)
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint)
        model.to(device)

    attacks = ['apgd-ce']
    name_file = 'attacks_%s_threshold_%03d.pth'%( "_".join(attacks), int(np.round(vec_threshold[0].numpy()*100)))
    print(name_file)
    complete_address = os.path.join(directory_result, name_model, name_file)
    threshold_auto_best = search_fine(model, x_test, y_test, vec_threshold, batch_size=batch_size, complete_address = complete_address, attacks = attacks)

if __name__ == "__main__":
    main()