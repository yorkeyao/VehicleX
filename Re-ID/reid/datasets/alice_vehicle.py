from __future__ import print_function, absolute_import

import os.path as osp
import re
from glob import glob
import random


class alice(object):
    def __init__(self, root): # sys_image_by_test_baseline_multidistractor_AIC_1211_domain_transfer(Veri)_fix_intensity/
        train_dir = './data/VeRi/image_train/'
        sys_dir = './data/alice-vehicle/random_vehicles/'
        query_dir = './data/alice-vehicle/query/'
        gallery_dir = './data/alice-vehicle/gallery/'
        self.train_path = osp.expanduser(train_dir)
        self.gallery_path = osp.expanduser(gallery_dir)
        self.query_path = osp.expanduser(query_dir)
        self.sys_path = osp.expanduser(sys_dir)

        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        self.load()

    def preprocess_t(self, path, training = True, sys = False):
        pattern = re.compile(r'(\d+)_c(\d+)')
        all_pids = {}
        ret = []

        fpaths = sorted(glob(osp.join( self.sys_path, '*.jpg')))
        random.shuffle(fpaths)
        fpaths = fpaths[:45338]
        # fpaths = sorted(fpaths)
        for fpath in fpaths:
            fname = "../../alice-vehicle/random_vehicles/" + osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            # if cam > 20: continue
            if pid == -1: continue
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]
            
            ret.append((fname, pid, cam))

        return ret, int(len(all_pids))

    def preprocess_q_g(self, path):
        pattern = re.compile(r'(\d+)_c(\d+)')
        all_pids = {}
        ret = []

        fpaths = sorted(glob(osp.join(path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            # if cam > 15: continue
            if pid == -1: continue
            if pid not in all_pids:
                all_pids[pid] = len(all_pids)
            pid = all_pids[pid]
            # print (pid, cam)
            ret.append((fname, pid, cam))

        return ret, int(len(all_pids))

    def load(self):
        print (self.train_path)
        print (self.sys_path)
        self.train, self.num_train_ids = self.preprocess_t(self.train_path, training = False, sys = True)
        self.gallery, self.num_gallery_ids = self.preprocess_q_g(self.gallery_path)
        self.query, self.num_query_ids = self.preprocess_q_g(self.query_path)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
              .format(self.num_train_ids, len(self.train)))
        print("  query    | {:5d} | {:8d}"
              .format(self.num_query_ids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
              .format(self.num_gallery_ids, len(self.gallery)))
