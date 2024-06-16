# MeanSparse: Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification

This repository contains the code to reproduce our record-breaking results in the paper [“MeanSparse: Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification"](https://arxiv.org/pdf/2406.05927).

We introduce MeanSparse, a technique that applies a mean-centered feature sparsification operator to post-process adversarially trained models. Using this operator, the MeanSparse technique effectively blocks some capacity used by adversaries without significantly impacting the model’s utility. Our empirical results demonstrate that MeanSparse sets new records in robustness for both CIFAR-10 and ImageNet datasets.

In this repository we publicly share two models on CIFAR-10 and two models on ImageNet which are imporoved versions of top ranked models on [RobustBench [4]](https://robustbench.github.io/).

## Results & Model Weights
We apply the MeanSparse technique to RaWideResNet [1] and WideResNet [2] models, the first and second rank models in RobustBench [4], respectively. Both methods demonstrated robustness improvement while their benign accuracy remained almost unchanged. For the ImageNet dataset, we apply MeanSparse to ConvNeXt-L [3] and RaWideResNet [1] architectures, and again we achieve improved robustness without utility loss. The complete results are summarized in the table below:

Note: We use the trained models from RobustBench [4] and generate the post-processed models using the MeanSparse technique. The robust accuracy is measured using
[AutoAttack](https://github.com/fra31/auto-attack). The models can also be downloaded through the links.

| Original Model  | Dataset| Clean (Original) | AA (Original) | Clean (with MeanSparse) | AA (with MeanSparse) |  MeanSparse integrated Model Weights |
|-----------------|--------|:----------------:|:-------------:|:------------------:|:---------------:|:------------------------------:|
| RaWideResNet [1]|CIFAR-10|      93.27%      |    71.07%     |      93.24%        |     72.08%      | [Sparsified_RaWideResNet_CIFAR](https://huggingface.co/MeanSparse/MeanSparse/blob/main/cifar10/Peng2023Robust_WS.pt) |
| WideResNet [2]  |CIFAR-10|      93.26%      |    70.69%     |      93.18%        |     71.41%      | [Sparsified_WideResNet_CIFAR](https://huggingface.co/MeanSparse/MeanSparse/blob/main/cifar10/Wang2023Better_WRN-70-16_WS.pt)  |
| ConvNeXt-L [3]  |ImageNet|      78.02%      |    58.48%     |      77.96%        |     59.64%      | [Sparsified_ConvNeXt-L_ImageNet](https://huggingface.co/MeanSparse/MeanSparse/blob/main/imagenet/Liu2023Comprehensive_ConvNeXt-L_WS.pt) |
| RaWideResNet [1]|ImageNet|      73.58%      |    48.94%     |      73.28%        |     52.98%      | [Sparsified_RaWideResNet_ImageNet](https://huggingface.co/MeanSparse/MeanSparse/blob/main/imagenet/Peng2023Robust_WS.pt) |

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
Running the code will automatically download the CIFAR-10 dataset. However, the ImageNet dataset must be downloaded manually due to licensing restrictions.

Get the download link [here](https://image-net.org/download.php) (you'll need to sign up with an academic email, and approval is automatic and immediate). Then, follow the instructions [here](https://github.com/soumith/imagenet-multiGPU.torch#data-processing) to extract the validation set into the `val` folder in a PyTorch-compatible format.

Important: Update the `data_dir` arguments in the `configs` files located in the `imagenet` folder to reflect the local path of ImageNet-1k on your machine.

## Reproducing the Results
To regenerate the results, select the configuration file for each model and run `main.py`. For example, to reproduce the results for the RaWideResNet model with the CIFAR-10 dataset, use the following command:
```.bash
python main.py --configs configs/cifar-10/configs_Peng2023Robust_RaWideResNet-70-16.yml
```

## Contact
If you have any questions, feel free to contact us through email (mteymoorianf@umass.edu) or Github issues.

## References

[1] ShengYun Peng, Weilin Xu, Cory Cornelius, Matthew Hull, Kevin Li, Rahul Duggal, Mansi Phute, Jason Martin, and Duen Horng Chau. Robust principles: Architectural design principles for adversarially robust cnns. In 34th British Machine Vision Conference 2023, BMVC 2023, Aberdeen, UK, November 20-24, 2023. BMVA, 2023.

[2] Zekai Wang, Tianyu Pang, Chao Du, Min Lin, Weiwei Liu, and Shuicheng Yan. Better diffusion models further improve adversarial training. In International Conference on Machine Learning, pages 36246–36263. PMLR, 2023.

[3] Chang Liu, Yinpeng Dong,Wenzhao Xiang, Xiao Yang, Hang Su, Jun Zhu, Yuefeng Chen, Yuan He, Hui Xue, and Shibao Zheng. A comprehensive study on robustness of image classification models: Benchmarking and rethinking. arXiv preprint arXiv:2302.14301, 2023.

[4] Francesco Croce, Maksym Andriushchenko, Vikash Sehwag, Edoardo Debenedetti, Nicolas Flammarion, Mung Chiang, Prateek Mittal, and Matthias Hein. Robustbench: a standardized adversarial robustness benchmark. arXiv preprint arXiv:2010.09670, 2020.

## BibTex

If you find this code or idea useful, please consider citing our work:
```
@article{amini2024meansparse,
  title={MeanSparse: Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification},
  author={Amini, Sajjad and Teymoorianfard, Mohammadreza and Ma, Shiqing and Houmansadr, Amir},
  journal={arXiv preprint arXiv:2406.05927},
  year={2024}
}
```
