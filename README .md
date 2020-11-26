# Hierarchical Adaptive Lasso: Learning Sparse Neural Networks via Single Stage Training
This repository accompanies the paper, Hierarchical Adaptive Lasso: Learning Sparse Neural Networks via Single Stage Training under review at NeurIPS 2020.


## About the Repository
In our paper we introduce HALO, a penalty with learnable parameters for sparsifying models (particularly deep neural networks). This repository contains the code necessary to replicate the experiments in Sections 4 and in the supplementary material and use HALO.


## Setup and Requirements
This code was developed and tested on Nvidia V100 in the following environment:

- CentOS 7
- Python 3.6.8

A ```requirements.txt``` file with all relevant python packages is included.  Apart from python packages, you will want to make sure your workspace has enough room for downloading CIFAR-10/0 and storing several copies of models (VGG models are typically around 100MB unpruned).

### Evaluating pre-trained models
We provide two pre-trained models for VGG-16 on CIFAR-100. Running this command will give the test accuracy as well as the prune ratio for the model to compare with results in Table 4 of the paper.

```
python check_params.py \
  -d cifar100  \
  -a vgg16_bn \
  --depth 16 \
  --checkpoint ../../pretrained_models/vgg16_c100_pruned_0.95.pth.tar
```


### Checking sparsity for pruned models
Code is also provided for obtaining the eigenvalues of the ouput covariance matrices to obtain Figure 1c. The provided command will save the eigenvalues for each convolutional layer.

```
python compute_energy.py \
  -d cifar100 \
  -a vgg16_bn \
  --depth 16 \
  --checkpoint ../../pretrained_models/vgg16_c100_pruned_0.95.pth.tar \
  --save $SAVE_FILE$
```

For the other types of learned sparsity, the ```check_params.py``` code outputs sparsity per layer, and ```cifar.py``` saves the convolutional layer weights and regularization coefficients as numpy arrays.


### Training models with HALO on CIFAR10/0

This command will train a VGG-like architecture on the CIFAR-10 dataset with the HALO penalty. Running this command should give you an accuracy close to 93.61. You can change the architecture to resnet and depth to 50 to replicate our experiments with a different architecture, and the dataset to cifar100. Information about proper hyperparameters can be found in the appendix of our paper. A similar command can be used for MNIST experiments using the mnist.py file in the mnist directory.
```
python cifar.py \
  -d cifar10  \
  -a vgg16_bn \
  --depth 16 \
  --wd 1e-4 \
  --halo \
  --psi 1e-8 \
  --xi 1e-8 \
  --save_dir $SAVE_DIRECTORY$
```

Note, the learning rate and initial value of regularization coefficients can be specified using --lr, and --lambda_init respectively.  By default the same weight decay and learning rate are used for the regularization coefficients. You can also edit other training parameters such as the number of epochs, decay epoch specifications, etc. by looking at other arguments, and switch the regularizer with either --l1 or --ws (SWS).

### Pruning models
This command will prune a VGG-like architecture trained on CIFAR-10 at a specific prune percentage and perform test evaluation to get the accuracy before and after pruning and compare with Tables 2. You will need to specify the save and load locations, and the output file will be a pruned model with the named pruned_$percent$.path.tar. A similar command can be used for MNIST experiments using the mnist.py file in the mnist directory.
```
python cifar_prune_iterative.py \
  -d cifar10  \
  -a vgg16_bn \
  --depth 16 \
  --percent 0.95 \
  --resume $MODEL_PATH$ \
  --save_dir $SAVE_DIRECTORY$
```

Please see the library for rethinking the value of network pruning for guidance on how to run this file, and license information: https://github.com/Eric-mingjie/rethinking-network-pruning


### Citations
This codebase uses code from the following papers as indicated in the files and readme. Please cite them accordingly.

@inproceedings{liu2018rethinking,
  title={Rethinking the Value of Network Pruning},
  author={Liu, Zhuang and Sun, Mingjie and Zhou, Tinghui and Huang, Gao and Darrell, Trevor},
  booktitle={ICLR},
  year={2019}
}


### Distribution
Please do not distribute this code or release it elsewhere without permission.
