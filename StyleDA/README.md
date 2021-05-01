# SPGAN
This repo is based on Weijian's [SPGAN](https://github.com/Simon4Yan/eSPGAN). The code structure shows as follows:

```
~
└───StyleDA
    └───datasets
    │   └───sys2real
    |   │   │ trainA
    |   │   │ trainB
    |   │   │ testA
    |   │   │ 
    └───code
    │   │ train_spgan.py
    │   │ test_spgan.py
    │   │ ...
    └───checkpoints
    │   │ ...
```

# Train with SPGAN

You will need to get the dataset prepared first. Please fill the empty folder trainA, trainB, testA and testB in the dataset folder before running the code. For training:

```shell script
CUDA_VISIBLE_DEVICES='0' python train_spgan.py 
```

it will train a model that perform image translation from domain A to domain B. 

# Inference with SPGAN

We provide our trained model at [google drive](https://drive.google.com/open?id=1bFX1KxNcBkyxWXdO_hOmP6t-GPATO9hK). Once you download them, please store them in ./checkpoints. After this, you may preform inference by 

```shell script
CUDA_VISIBLE_DEVICES='0' python test_spgan.py 
```

Reference:

```
@inproceedings{image-image18,
 author    = {Weijian Deng and
              Liang Zheng and
              Qixiang Ye and
              Guoliang Kang and
              Yi Yang and
              Jianbin Jiao},
 title     = {Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity
              for Person Re-identification},
 booktitle = {CVPR},
 year      = {2018},
}
```
