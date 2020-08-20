CUDA_VISIBLE_DEVICES='0,1' python IDE.py --train -d vihicle_id -r true -s true --combine-trainval --logs-dir logs/VID/ide_joint_stageI -b 64 --re 1 --height 256 --width 256 

CUDA_VISIBLE_DEVICES='0,1' python IDE.py --finetune -d vihicle_id -r true -s false --combine-trainval --resume logs/VID/ide_joint_stageI/model_best.pth.tar --logs-dir logs/VID/ide_joint_stageII -b 64 --re 1 --height 256 --width 256 --epochs 67

CUDA_VISIBLE_DEVICES='0,1' python IDE.py --evaluate_VID -d vihicle_id --combine-trainval --resume logs/VID/ide_joint_stageII/model_best.pth.tar --logs-dir logs/VID/ide_joint_stageII -b 64 --re 1 --height 256 --width 256 