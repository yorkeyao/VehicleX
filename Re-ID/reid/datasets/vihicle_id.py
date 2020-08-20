from __future__ import print_function, absolute_import

import os.path as osp
import re
import xml.dom.minidom as XD
from collections import defaultdict
from glob import glob
import random


class Vihicle_ID_Sys(object):

    def __init__(self, root, real = True, synthetic = True):

        backbone_path = "./data/"
        self.root = osp.expanduser( backbone_path + 'VehicleID_V1.0')
        label_path = osp.join (self.root, 'train_test_split')
        self.train_path = osp.join(self.root, 'image')
        self.real = real
        self.synthetic = synthetic
        self.query_path = osp.join(self.root, 'image')
        self.gallery_path = osp.join(self.root, 'image')
        self.train_path_label = osp.join(label_path, 'train_list.txt')
        self.query_path_label = osp.join(label_path, 'test_list_2400.txt') 
        sys_dir = backbone_path + '/VehicleID_V1.0/VID_ReID_Simulation' 
        self.train_path_label = osp.expanduser(self.train_path_label)
        self.query_path_label = osp.expanduser(self.query_path_label)
        self.sys_path = osp.expanduser(sys_dir)

        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        self.load()

    def preprocess_query_gallery(self, real_path, relabel=True, random_test = True):
        all_pids = {}
        ret_query = []
        ret_gallery = []
        if real_path is None:
            return ret_query, int(len(all_pids))
        cnt = 0
        with open(real_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip().split(' ') for line in lines]
            if random_test:
                random.shuffle(lines)
            for line in lines:
                fname, pid = line
                fname = fname + ".jpg"
                gallery_judge = False
                if pid == -1: continue
                if relabel:
                    if pid not in all_pids:
                        all_pids[pid] = len(all_pids)
                        gallery_judge = True
                else:
                    if pid not in all_pids:
                        all_pids[pid] = pid
                pid = all_pids[pid]
                if gallery_judge:
                    ret_gallery.append((fname, pid, cnt))
                else:
                    ret_query.append((fname, pid, cnt))
                cnt = cnt + 1
        return ret_query, int(len(all_pids)), ret_gallery, int(len(all_pids))

    def preprocess_joint(self, real_path, sys_path, relabel=True, real=True, synthetic=True):
        all_pids = {}
        ret = []

        if real_path is None:
            return ret, int(len(all_pids))

        real_cnt = 0

        if real:
            with open(real_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                lines = [line.strip().split(' ') for line in lines]
                for line in lines:
                    fname, pid = line
                    fname = fname + ".jpg"
                    if pid == -1: continue
                    if relabel:
                        if pid not in all_pids:
                            all_pids[pid] = len(all_pids)
                    else:
                        if pid not in all_pids:
                            all_pids[pid] = pid
                    pid = all_pids[pid]
                    ret.append((fname, pid, 1))
                    real_cnt = real_cnt + 1
        if synthetic:
            pattern = re.compile(r'(\d+)_c(\d+)')
            fpaths = sorted(glob(osp.join(sys_path, '*.jpg')))
            sys_cnt = 0
            random.shuffle(fpaths)
            # print (len(ret), int(len (ret) * 0.25))
            if real:
                fpaths = fpaths[:int(len (ret) * 0.5)]
            else:
                fpaths = sorted(fpaths[:45538])
            for fpath in fpaths:
                fname = "../VID_ReID_Simulation/" + osp.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                pid = -pid 
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
                pid = all_pids[pid]
                # print (fname)
                ret.append((fname, pid, cam))
                sys_cnt = sys_cnt + 1

        return ret, int(len(all_pids))

    def load(self):
        print ("real path:", self.train_path, "synthetic path:", self.sys_path)
        self.train, self.num_train_ids = self.preprocess_joint(self.train_path_label, self.sys_path, True, self.real, self.synthetic)
        self.query, self.num_query_ids, self.gallery, self.num_gallery_ids = self.preprocess_query_gallery(self.query_path_label, True)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
        # assert (0)
