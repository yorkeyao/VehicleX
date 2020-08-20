# Open-ReID-tracking

This repo is based on Cysu's [open-reid](https://github.com/Cysu/open-reid) with features from Yunzhong's [open-reid](https://github.com/hou-yz/open-reid-tracking). 

### Data
Please download [AI-City 2020 datasets (track-2)](https://www.aicitychallenge.org/2020-track2-download/), [VeRi](https://github.com/JDAI-CV/VeRidataset#2-download), [VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html) and [adaptaed synthetic dataset]() for the following steps. Many datasets request link by email the holder. They should be stored in a file structure like this:

```
~
└───Data
    └───AIC20-reid
    │   │ image_train
    |   | AIC20_ReID_Simulation
    │   │ ...
    │
    └───VeRi
    |   │ image_train
    |   | VeRi_ReID_Simulation
    |   │ ...
    |
    └───VehicleID_V1.0
    |   │ image
    |   | VID_ReID_Simulation
    |   │ ...
```

### Single dataset training

```shell script
CUDA_VISIBLE_DEVICES=0,1 python IDE.py --train -d veri -r true -s false --combine-trainval --logs-dir logs/veri/ide_baseline -b 64 --re 1 --height 256 --width 256 --epoch 60
```

this will automatically save your logs and checkpoints at `./logs/ide/veri/ide_baseline`. -r means real world data will be use and -s means synthetic data will be used. 

### Joint training in two stage
We provide bash file for two stage training. You may check 'train_veri.sh', 'train_vehicleID.sh' and 'train_aic.sh'. For example, the joint training for VeRi dataset involve two command: 

```shell script
CUDA_VISIBLE_DEVICES='0,1' python IDE.py --train -d veri -r true -s true --combine-trainval --logs-dir logs/veri/ide_joint_stageI -b 64 --re 1 --height 256 --width 256 --epoch 60

CUDA_VISIBLE_DEVICES='0,1' python IDE.py --finetune -d veri -r true -s false --combine-trainval --resume logs/veri/ide_joint_stageI/model_best.pth.tar --logs-dir logs/veri/ide_joint_stageII -b 64 --re 1 --height 256 --width 256 --epoch 70
```

For VehicleID, the evaluation need to iterate 10 times, thus we need to have an additional evaluation. 

```shell script
CUDA_VISIBLE_DEVICES='0,1' python IDE.py --train -d vihicle_id -r true -s true --combine-trainval --logs-dir logs/VID/ide_joint_stageI -b 64 --re 1 --height 256 --width 256 

CUDA_VISIBLE_DEVICES='0,1' python IDE.py --finetune -d vihicle_id -r true -s false --combine-trainval --resume logs/VID/ide_joint_stageI/model_best.pth.tar --logs-dir logs/VID/ide_joint_stageII -b 64 --re 1 --height 256 --width 256 --epochs 67

CUDA_VISIBLE_DEVICES='0,1' python IDE.py --evaluate_VID -d vihicle_id --combine-trainval --resume logs/VID/ide_joint_stageII/model_best.pth.tar --logs-dir logs/VID/ide_joint_stageII -b 64 --re 1 --height 256 --width 256 
```

For aicity, the final result will be checked by online server, thus we have an additional inference procedure. 

```shell script
CUDA_VISIBLE_DEVICES='0,1' python ZJU_baseline.py --train -d aic_reid_sys -r true -s true --logs-dir logs/AIC/zju_joint_stageI --height 256 --width 256 --lr 0.01 --step-size 30,60,80 --warmup 10 --LSR --backbone densenet121 --features 1024 --BNneck -ls 1 -b 64 --epochs 120

CUDA_VISIBLE_DEVICES='0,1' python ZJU_baseline.py --finetune -d aic_reid_sys -r true -s false --resume logs/AIC/zju_joint_stageI/model_best.pth.tar --logs-dir logs/AIC/zju_joint_stageII --height 256 --width 256 --lr 0.01 --step-size 30,60,80 --warmup 10 --LSR --backbone densenet121 --features 1024 --BNneck -ls 1 -b 64 

CUDA_VISIBLE_DEVICES='0,1' python ZJU_baseline.py --inference -d aic_reid_sys --resume logs/AIC/zju_joint_stageII/model_best.pth.tar --logs-dir logs/AIC/zju_joint_stageII --height 256 --width 256 --lr 0.01 --step-size 30,60,80 --warmup 10 --LSR --backbone densenet121 --features 1024 --BNneck -ls 1 -b 64 --epochs 120
```

After inference, it will generate ./result.txt for online test. 