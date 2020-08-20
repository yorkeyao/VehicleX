from __future__ import print_function, absolute_import

import os.path as osp
import re
import xml.dom.minidom as XD
from collections import defaultdict
from glob import glob
import random


class AI_City_Sys(object):

    def __init__(self, root, real = True, synthetic = True, type='reid', fps=10, trainval=False, gt_type='gt'):

        backbone_path = "./data/"

        if type == 'tracking_gt':
            if not trainval:
                train_dir = '/AIC19/ALL_{}_bbox/train'.format(gt_type)
            else:
                train_dir = '/AIC19/ALL_{}_bbox/trainval'.format(gt_type)
            val_dir = '/AIC19/ALL_gt_bbox/val'
            self.train_path = osp.join(osp.expanduser(train_dir), ('gt_bbox_{}_fps'.format(fps)))
            self.gallery_path = osp.join(osp.expanduser(val_dir), 'gt_bbox_1_fps')
            self.query_path = osp.join(osp.expanduser(val_dir), 'gt_bbox_1_fps')
        elif type == 'tracking_det':
            self.train_path = root
            self.gallery_path = None
            self.query_path = None
        elif type == 'reid':  # reid
            root = osp.expanduser( backbone_path + '/AIC19-reid')
            self.train_path = osp.join(root, 'image_train')
            
            sys_dir = backbone_path + '/AIC19-reid/AIC20_ReID_Simulation'
            query_dir = backbone_path + '/VeRi/image_query/'
            gallery_dir = backbone_path + '/VeRi/image_test/'

            self.gallery_path = osp.expanduser(gallery_dir)
            self.query_path = osp.expanduser(query_dir)
            self.sys_path = osp.expanduser(sys_dir)

            xml_dir = osp.join(root, 'train_label.xml')
            self.reid_info = XD.parse(xml_dir).documentElement.getElementsByTagName('Item')
            self.index_by_fname_dict = defaultdict()
            for index in range(len(self.reid_info)):
                fname = self.reid_info[index].getAttribute('imageName')
                self.index_by_fname_dict[fname] = index
            
            self.real = real
            self.synthetic = synthetic

        elif type == 'reid_inference':  # reid_test
            self.train_path = None
            self.sys_path = None
            root = osp.expanduser( backbone_path + '/AIC19-reid')
            self.gallery_path = osp.join(root, 'image_test')
            self.query_path = osp.join(root, 'image_query')
            self.real = real
            self.synthetic = synthetic

            # print (root)

        self.train, self.query, self.gallery = [], [], []
        self.num_train_ids, self.num_query_ids, self.num_gallery_ids = 0, 0, 0

        self.type = type
        self.load()

    def preprocess_joint(self, real_path, sys_path, relabel=True, type='reid'):
        if type == 'tracking_det':
            pattern = re.compile(r'c([-\d]+)_f(\d+)')
        elif type == 'tracking_gt':
            pattern = re.compile(r'([-\d]+)_c(\d+)')
        else:  # reid
            pattern = None
        all_pids = {}
        ret = []
        if self.real:
            if real_path is None:
                return ret, int(len(all_pids))
            fpaths = sorted(glob(osp.join(real_path, '*.jpg')))
            for fpath in fpaths:
                fname = osp.basename(fpath)
                if type == 'tracking_det':
                    cam, frame = map(int, pattern.search(fname).groups())
                    pid = 1
                elif type == 'tracking_gt':
                    pid, cam = map(int, pattern.search(fname).groups())
                elif type == 'reid':  # reid
                    pid, cam = map(int, [self.reid_info[self.index_by_fname_dict[fname]].getAttribute('vehicleID'),
                                        self.reid_info[self.index_by_fname_dict[fname]].getAttribute('cameraID')[1:]])
                else:  # reid test
                    pid, cam = 1, 1
                if pid == -1: continue
                if relabel:
                    if pid not in all_pids:
                        all_pids[pid] = len(all_pids)
                else:
                    if pid not in all_pids:
                        all_pids[pid] = pid
                pid = all_pids[pid]
                cam -= 1
                ret.append((fname, pid, cam))

        if self.synthetic:
            pattern = re.compile(r'(\d+)_c(\d+)')
            fpaths = sorted(glob(osp.join(sys_path, '*.jpg')))
            for fpath in fpaths:
                fname = "../AIC20_ReID_Simulation/" + osp.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                if pid == -1: continue
                pid = -pid 
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
                pid = all_pids[pid]
                # print (fname)
                ret.append((fname, pid, cam))
        return ret, int(len(all_pids))

    def preprocess(self, path, relabel=True, type='reid'):
        if type == 'tracking_det':
            pattern = re.compile(r'c([-\d]+)_f(\d+)')
        elif type == 'tracking_gt':
            pattern = re.compile(r'([-\d]+)_c(\d+)')
        else:  # reid
            pattern = None
        all_pids = {}
        ret = []
        if path is None:
            return ret, int(len(all_pids))
        fpaths = sorted(glob(osp.join(path, '*.jpg')))
        for fpath in fpaths:
            fname = osp.basename(fpath)
            if type == 'tracking_det':
                cam, frame = map(int, pattern.search(fname).groups())
                pid = 1
            elif type == 'tracking_gt':
                pid, cam = map(int, pattern.search(fname).groups())
            elif type == 'reid':  # reid
                pid, cam = map(int, [self.reid_info[self.index_by_fname_dict[fname]].getAttribute('vehicleID'),
                                     self.reid_info[self.index_by_fname_dict[fname]].getAttribute('cameraID')[1:]])
            else:  # reid test
                pid, cam = 1, 1
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append((fname, pid, cam))
        return ret, int(len(all_pids))

    def load(self):
        self.train, self.num_train_ids = self.preprocess_joint(self.train_path, self.sys_path, True, self.type)
        self.gallery, self.num_gallery_ids = self.preprocess(self.gallery_path, False,
                                                             'reid_inference' if self.type == 'reid_inference' else 'tracking_gt')
        self.query, self.num_query_ids = self.preprocess(self.query_path, False,
                                                         'reid_inference' if self.type == 'reid_inference' else 'tracking_gt')

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
