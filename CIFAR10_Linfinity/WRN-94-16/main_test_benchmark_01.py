import os
import torch
from robustbench.utils import load_model
from utils_sparse import calculate_accuracy, add_custom_layer, calculate_statistics, benchmark
import torch.nn as nn
from data import load_cifar10
from MeanSparse_robustarch_wide_resnet import NormalizedWideResNet, MeanSparse

parts = range(0,10)

threshold = 0.25
batch_size = 100


active_dataset = 'test'

name_model='Peng2023Robust' #Rank 1
# name_model='Wang2023Better_WRN-70-16' #Rank 2

flag_calculate_acc = False
directory_result = "results"
directory_WS = "models_WS"

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Always move to device "cuda:0" first
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

if not os.path.exists(directory_result):
    os.makedirs(directory_result, exist_ok=True)
    
x_test, y_test = load_cifar10(n_examples=10000)
x_test = x_test.to(device)
y_test = y_test.to(device)

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

file_path = os.path.join(directory_WS, '%s_WS.pt'%name_model)
checkpoint = torch.load(file_path)
model.load_state_dict(checkpoint)

for name, module in model.named_modules():
    if isinstance(module, MeanSparse):
        module.threshold.data.fill_(threshold)

model.to(device)

if flag_calculate_acc:
    results = calculate_accuracy(model, x_test, y_test, batch_size=512)
    print('MeanSparse Model Accuracy:', results)

for part in parts:
    x_active = x_test[part*batch_size:(part+1)*batch_size,:,:,:]
    y_active = y_test[part*batch_size:(part+1)*batch_size]

    name_file = '%s_Benchmark_test_part_%02d.pt'%(name_model, part)
    if not os.path.exists(os.path.join(directory_result,name_file)):
        threshold_auto_best = benchmark(model, x_active, y_active, batch_size=batch_size, directory = directory_result, name_file = name_file)
