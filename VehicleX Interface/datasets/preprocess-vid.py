import shutil
import os.path as osp
import re
from glob import glob
from random import choice
import random
import argparse
import os

random.seed (1)

parser = argparse.ArgumentParser(description='outputs')
parser.add_argument('--ori_path',default="./VehicleID_V1.0/",type=str, help='./target dataset and attribute definition')
parser.add_argument('--target_path',default='./VID_split/',type=str, help='./output attributes')
opt = parser.parse_args()

ori_path = opt.ori_path
target_path = opt.target_path

def frames_percam(single_id = False):
    all_pids = {}
    real_path = ori_path + "/train_test_split/train_list.txt"
    only_name = []
    with open(real_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip().split(' ') for line in lines]
        for line in lines:
            fname, pid = line
            fname = fname + ".jpg"
            if pid == -1: continue
            if pid not in all_pids:
                all_pids[pid] = pid
                only_name.append(fname)
            pid = all_pids[pid]
    return only_name

if __name__ == "__main__":
    for i in frames_percam():
        shutil.copy(os.path.join(opt_path + '/image/', i), target_path)

