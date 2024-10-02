# MeanSparse: Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification

This repository contains the code to reproduce our record-breaking results in the paper [“MeanSparse: Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification"](https://arxiv.org/pdf/2406.05927).

We introduce MeanSparse, a technique that applies a mean-centered feature sparsification operator to post-process adversarially trained models. Using this operator, the MeanSparse technique effectively blocks some capacity used by adversaries without significantly impacting the model’s utility. Our empirical results demonstrate that MeanSparse sets new records in robustness for both CIFAR-10 and ImageNet datasets.

In this repository we publicly share several models on CIFAR-10, CIFAR-100, and ImageNet which are imporoved versions of top ranked models on [RobustBench [4]](https://robustbench.github.io/).

## Results & Model Weights
We apply the MeanSparse technique to several top-ranked models on RobustBench [4]. The complete results are summarized in the table below:

Note: We use the trained models from RobustBench [4] and generate the post-processed models using the MeanSparse technique. The robust accuracy is measured using
[AutoAttack](https://github.com/fra31/auto-attack). The models can also be downloaded through the links.

| Original Model  | Dataset| Clean (Original) | AA (Original) | Clean (with MeanSparse) | AA (with MeanSparse) |  MeanSparse integrated Model Weights |
|-----------------|--------|:----------------:|:-------------:|:------------------:|:---------------:|:------------------------------:|
| WRN-94-16 [5]  |CIFAR-10($L_\inf$) |      93.68%      |    73.71%     |      93.63%        |     75.28%      | [Sparsified_WRN_94_16_CIFAR](https://drive.google.com/file/d/1wHkKzSD4nk6IT0uOZa23ZFXSMBSAQ7mt/view?usp=share_link) |
| RaWRN-70-16 [1]|CIFAR-10($L_\inf$) |      93.27%      |    71.07%     |      93.27%        |     72.78%      | [Sparsified_RaWRN_70_16_CIFAR](https://drive.google.com/file/d/1y-7wjdZI_UEvtt33pDLOhw41zx_oEqIp/view?usp=share_link) |
| WRN-70-16 [2]  |CIFAR-10($L_\inf$) |      93.26%      |    70.69%     |      93.18%        |     71.41%      | [Sparsified_WRN_70_16_CIFAR](https://drive.google.com/file/d/1aaLxiSTViNB3hyG1UpOKj9Rw_bNeBUTk/view?usp=share_link)  |
| WRN-70-16 [2]  |CIFAR-10($L_2$) |      95.54%      |    84.97%     |      95.49%        |     87.28%      | [Sparsified_WRN_70_16_CIFAR_L2](https://drive.google.com/file/d/1pBkO7aBB5CsoDHT2yaJVNv1vTni5yRGu/view?usp=share_link)  |
| WRN-70-16 [2]  |CIFAR-100($L_\inf$)|      75.22%      |    42.67%     |      75.17%        |     44.78%      | [Sparsified_WRN_70_16_CIFAR_100](https://drive.google.com/file/d/1VYlfRrkaKnsaZqunCQ7K-iR8PXFYT6CL/view?usp=share_link)  |
| Swin-L [3]      |ImageNet($L_\inf$)|      78.92%      |    59.56%     |      78.86%        |     62.12%      | [Sparsified_Swin_L_ImageNet](https://drive.google.com/file/d/1hL_cFQxNa7ZHKJ-f2CrcbxBBy8TTofW3/view?usp=share_link) |
| ConvNeXt-L [3]  |ImageNet($L_\inf$)|      78.02%      |    58.48%     |      77.96%        |     59.64%      | [Sparsified_ConvNeXt-L_ImageNet](https://drive.google.com/file/d/1X6wihZ4Jm4Zm_O0tSgd13Bv-naAoA8hN/view?usp=share_link) |
| RaWRN [1]|ImageNet($L_\inf$)|      73.58%      |    48.94%     |      73.28%        |     52.98%      | [Sparsified_RaWideResNet_ImageNet](https://drive.google.com/file/d/1VzGCiVEHE6lv_uU7bLvFmI4p9sQnLx6W/view?usp=share_link) |

## Requirements
- Install or download [AutoAttack](https://github.com/fra31/auto-attack):
```.bash
pip install git+https://github.com/fra31/auto-attack
```
- Install or download the RobustBench (The compelete instructions can be found [here](https://github.com/RobustBench/robustbench?tab=readme-ov-file#model-zoo-quick-tour))
```.bash
pip install git+https://github.com/RobustBench/robustbench.git
```

## Dataset
Running the code will automatically download the CIFAR-10 and CIFAR-100 datasets. However, the ImageNet dataset must be downloaded manually due to licensing restrictions.

Get the download link [here](https://image-net.org/download.php) (you'll need to sign up with an academic email, and approval is automatic and immediate). Then, follow the instructions [here](https://github.com/soumith/imagenet-multiGPU.torch#data-processing) to extract the validation set into the `val` folder in a PyTorch-compatible format.

Important: Update the `data_dir` arguments in the `configs` files located in the `imagenet` folder to reflect the local path of ImageNet-1k on your machine.

## Reproducing the Results
To reproduce the results for each model:
1. Navigate to the respective directory for the model.
2. Download the model weights from the table above.
3. Create a models_WS directory within the model's directory.
4. Move the downloaded weights into the models_WS directory.

For example, the directory structure for CIFAR-100 should look like this:
```
CIFAR100_Linfinity
│
└───models_WS 
│   └───Wang2023Better_WRN-70-16_WS.pt
```
Finally, run the Python script that starts with AutoAttack to execute the relevant tests.

## References

[1] ShengYun Peng, Weilin Xu, Cory Cornelius, Matthew Hull, Kevin Li, Rahul Duggal, Mansi Phute, Jason Martin, and Duen Horng Chau. Robust principles: Architectural design principles for adversarially robust cnns. In 34th British Machine Vision Conference 2023, BMVC 2023, Aberdeen, UK, November 20-24, 2023. BMVA, 2023.

[2] Zekai Wang, Tianyu Pang, Chao Du, Min Lin, Weiwei Liu, and Shuicheng Yan. Better diffusion models further improve adversarial training. In International Conference on Machine Learning, pages 36246–36263. PMLR, 2023.

[3] Chang Liu, Yinpeng Dong,Wenzhao Xiang, Xiao Yang, Hang Su, Jun Zhu, Yuefeng Chen, Yuan He, Hui Xue, and Shibao Zheng. A comprehensive study on robustness of image classification models: Benchmarking and rethinking. arXiv preprint arXiv:2302.14301, 2023.

[4] Francesco Croce, Maksym Andriushchenko, Vikash Sehwag, Edoardo Debenedetti, Nicolas Flammarion, Mung Chiang, Prateek Mittal, and Matthias Hein. Robustbench: a standardized adversarial robustness benchmark. arXiv preprint arXiv:2010.09670, 2020.

[5] Bartoldson, Brian R., James Diffenderfer, Konstantinos Parasyris, and Bhavya Kailkhura. "Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies." arXiv preprint arXiv:2404.09349 (2024).
