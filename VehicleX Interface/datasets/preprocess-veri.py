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
parser.add_argument('--ori_path',default="./VeRi/image_train/",type=str, help='./target dataset and attribute definition')
parser.add_argument('--target_path',default='./VeRi/cam_split/',type=str, help='./output attributes')
opt = parser.parse_args()

ori_path = opt.ori_path
target_path = opt.target_path

def frames_percam(cam_id, single_id = False):
    all_pids = {}
    only_name = []
    pid_fname = {}
    pattern = re.compile(r'(\d+)_c(\d+)')
    fpaths = sorted(glob(osp.join(ori_path, '*.jpg')))
    for fpath in fpaths:
        fname = osp.basename(fpath)
        pid, cam = map(int, pattern.search(fname).groups())
        if cam != cam_id:
            continue
        if pid == -1: continue
        if pid not in all_pids:
            all_pids[pid] = len(all_pids)
            pid_fname[all_pids[pid]] = [fname]
        else:
            pid_fname[all_pids[pid]].append(fname)
        pid = all_pids[pid]
    for key in pid_fname.keys():
        if single_id:
            only_name.append(choice(pid_fname[key]))
        else:
            only_name.extend(pid_fname[key]) 
    return only_name

if __name__ == "__main__":
    for i in range(1,21):
        print ("cam:", i, "frame num:", len (frames_percam(i)))
        for j in frames_percam(i):
            dir_path = os.path.join(target_path, str(i))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            shutil.copy(ori_path + j, dir_path)