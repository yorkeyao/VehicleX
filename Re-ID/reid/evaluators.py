from __future__ import print_function, absolute_import

import time
from collections import OrderedDict

import torch

from .evaluation_metrics import cmc, mean_ap, accuracy
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
from .utils import to_numpy
import numpy as np
from torch.autograd import Variable


def extract_features(model, data_loader, eval_only, print_freq=100):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs, eval_only)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels

def calculate_acc(model, data_loader, eval_only, print_freq=100):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    precisions = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        with torch.no_grad():
            outputs = model(imgs)
        prediction = outputs[1][0]
        print (np.shape (prediction))
        print (pids)
        prec, = accuracy(prediction.data, Variable(pids.cuda()))#.items()
        prec = prec.item()
        print (np.shape (pids))
        print (prec)
        assert (0)
        precisions.update(prec, pids.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  'Prec {:.2%} ({:.2%})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg,
                          precisions.val, precisions.avg))

    return None

def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10)):
    
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)
    
    

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    # print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
        # 'allshots': dict(separate_camera_set=False,
        #                  single_gallery_shot=False,
        #                  first_match_break=False),
        # 'cuhk03': dict(separate_camera_set=True,
        #                single_gallery_shot=True,
        #                first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]'
          .format(mAP, *cmc_scores['market1501'][[0, 4, 9]]))

    # Use the allshots cmc top-1 score for validation criterion
    return mAP, cmc_scores['market1501'][0],  cmc_scores['market1501'][4],  cmc_scores['market1501'][9]  # cmc_scores['market1501'][0]

def inference_all (distmat, query=None, gallery=None):
    # print (len (query))
    # print (len (gallery))
    # print (query[0])
    # print (gallery[0])
    # print (np.shape (distmat))
    distmat = to_numpy(distmat)
    m, n = distmat.shape

    

    f = open ("./result.txt", "w+")
    indices = np.argsort(distmat, axis=1)

    for i in range(m):
        for j in range(100):
            # print (distmat[i][indices[i][j]])
            # print (gallery[indices[i][j]][0].split('.')[0] + " ")
            f.write (gallery[indices[i][j]][0].split('.')[0] + " ")
            # -distmat[i][indices[i]][valid]
        # assert (0)
        f.write("\n")
    assert (0)

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery, metric=None, eval_only=True):
        print('extracting query features\n')
        query_features, _ = extract_features(self.model, query_loader, eval_only)
        print('extracting gallery features\n')
        gallery_features, _ = extract_features(self.model, gallery_loader, eval_only)
        distmat = pairwise_distance(query_features, gallery_features, query, gallery)
        return evaluate_all(distmat, query=query, gallery=gallery)

    def classification_evaluate(self, query_loader, gallery_loader, query, gallery, metric=None, eval_only=True):
        print('extracting query features\n')
        calculate_acc(self.model, query_loader, eval_only)

        return None

    def inference(self, query_loader, gallery_loader, query, gallery, metric=None, eval_only=True):
        print('extracting query features\n')
        query_features, _ = extract_features(self.model, query_loader, eval_only)
        print('extracting gallery features\n')
        gallery_features, _ = extract_features(self.model, gallery_loader, eval_only)
        distmat = pairwise_distance(query_features, gallery_features, query, gallery)
        return inference_all(distmat, query=query, gallery=gallery)
