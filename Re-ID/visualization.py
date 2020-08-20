import argparse
import os
import time

os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch

from reid import models
from reid.feature_extraction import extract_cnn_feature
from reid.utils.meters import AverageMeter
from reid.utils.my_utils import *

matplotlib.use('agg')


def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


parser = argparse.ArgumentParser(description="Softmax loss classification")
parser.add_argument('--log', type=bool, default=1)
# data
parser.add_argument('-d', '--dataset', type=str, default='aic_tracking', choices=datasets.names())
parser.add_argument('-b', '--batch-size', type=int, default=64, help="batch size")
parser.add_argument('-j', '--num-workers', type=int, default=1)
parser.add_argument('--height', type=int, default=256, help="input height, default: 256 for resnet*")
parser.add_argument('--width', type=int, default=128,
                    help="input width, default: 128 for resnet*")
parser.add_argument('--combine-trainval', action='store_true',
                    help="train and val sets together for training, val set alone for validation")
parser.add_argument('--tracking_icams', type=int, default=0, help="specify if train on single iCam")
parser.add_argument('--tracking_fps', type=int, default=10, help="specify if train on single iCam")
parser.add_argument('--re', type=float, default=0, help="random erasing")
# model
parser.add_argument('--features', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('-s', '--last_stride', type=int, default=2, choices=[1, 2])
parser.add_argument('--output_feature', type=str, default='fc', choices=['pool5', 'fc'])
# optimizer
parser.add_argument('--lr', type=float, default=0.1,
                    help="learning rate of new parameters, for pretrained "
                         "parameters it is 10 times smaller than this")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
# training configs
parser.add_argument('--train', action='store_true', help="train IDE model from start")
parser.add_argument('--crop', action='store_true', help="resize then crop, default: False")
parser.add_argument('--fix_bn', type=bool, default=0, help="fix (skip training) BN in base network")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--step-size', type=int, default=40)
parser.add_argument('--start_save', type=int, default=0, help="start saving checkpoints after specific epoch")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=1)
# camstyle batchsize
parser.add_argument('--camstyle', type=int, default=0)
parser.add_argument('--fake_pooling', type=int, default=1)
# misc
working_dir = osp.dirname(osp.abspath(__file__))
parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
parser.add_argument('--logs-dir', type=str, metavar='PATH')

parser.add_argument('--query_index', type=int, default=10)
parser.add_argument('--visual', type=str, default='val')

args = parser.parse_args()


def visualization(args):
    dataset, num_classes, train_loader, query_loader, gallery_loader, camstyle_loader = \
        get_data(args.dataset, args.data_dir, args.height, args.width, args.batch_size, args.num_workers,
                 args.combine_trainval, args.crop, args.tracking_icams, args.tracking_fps, args.re, 0, args.camstyle,
                 visual=args.visual)

    model = models.create('ide', num_features=args.features,
                          dropout=args.dropout, num_classes=num_classes, last_stride=args.last_stride,
                          output_feature=args.output_feature)

    model, start_epoch, best_top1 = checkpoint_loader(model, osp.join(args.logs_dir, 'model_best.pth.tar'),
                                                      eval_only=True)
    model = nn.DataParallel(model).cuda()

    def extract_features(model, data_loader, eval_only, print_freq=100):
        model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        features = []
        labels = []
        cameras = []

        end = time.time()
        for i, (imgs, fnames, pids, cids) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs, eval_only)
            for fname, output, pid, cid in zip(fnames, outputs, pids, cids):
                features.append(output)
                labels.append(int(pid.numpy()))
                cameras.append(int(cid.numpy()))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

        output_features = torch.stack(features, 0)

        return output_features, labels, cameras

    # sort the images
    def sort_img(qf, ql, qc, gf, gl, gc):
        query = qf.view(-1, 1)
        # print(query.shape)
        score = torch.mm(gf, query)
        score = score.squeeze(1).cpu()
        score = score.numpy()
        # predict index
        index = np.argsort(score)  # from small to large
        index = index[::-1]
        # index = index[0:2000]
        # good index
        query_index = np.argwhere(gl == ql)
        # same camera
        camera_index = np.argwhere(gc == qc)

        # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index1 = np.argwhere(gl == -1)
        junk_index2 = np.intersect1d(query_index, camera_index)
        junk_index = np.append(junk_index2, junk_index1)

        mask = np.in1d(index, junk_index, invert=True)
        index = index[mask]
        return index

    query_features, query_labels, query_cams = extract_features(model, query_loader, eval_only=True)
    gallery_feature, gallery_label, gallery_cam = extract_features(model, gallery_loader, eval_only=True)

    for select in range(args.query_index):
        if args.visual == 'val':
            selected = select * 10
        elif args.visual == 'trainval':
            selected = select * 30

        if args.dataset == 'aic_reid':
            selected = select

        index = sort_img(query_features[selected], query_labels[selected], query_cams[selected], gallery_feature,
                         gallery_label, gallery_cam)

        ########################################################################
        # Visualize the rank result

        query_path = dataset.query[selected][0]
        query_label = query_labels[selected]
        if args.dataset == 'aic_tracking':
            query_dir = '~/Data/AIC19/ALL_gt_bbox/trainval'
            query_path = os.path.join(os.path.expanduser(query_dir), 'gt_bbox_10_fps', query_path)
        elif args.dataset == 'veri':
            query_dir = '~/Data/VeRi/image_query'
            query_path = os.path.join(os.path.expanduser(query_dir), query_path)
        elif args.dataset == 'aic_reid':
            query_dir = '~/Data/AIC19_ReID/image_test'
            query_path = os.path.join(os.path.expanduser(query_dir), query_path)
        print(query_path)
        print('Top 10 images are as follow:')
        try:  # Visualize Ranking Result
            # Graphical User Interface is needed
            fig = plt.figure(figsize=(16, 4))
            ax = plt.subplot(1, 11, 1)
            ax.axis('off')
            imshow(query_path, 'query')
            for i in range(10):
                ax = plt.subplot(1, 11, i + 2)
                ax.axis('off')
                img_path = dataset.gallery[index[i]][0]
                if args.dataset == 'aic_tracking':
                    img_path = os.path.join(os.path.expanduser(query_dir), 'gt_bbox_10_fps', img_path)
                elif args.dataset == 'veri':
                    gallery_dir = '~/Data/VeRi/image_test'
                    img_path = os.path.join(os.path.expanduser(gallery_dir), img_path)
                elif args.dataset == 'aic_reid':
                    gallery_dir = '~/Data/AIC19_ReID/image_test'
                    img_path = os.path.join(os.path.expanduser(gallery_dir), img_path)
                label = gallery_label[index[i]]
                imshow(img_path)
                if label == query_label:
                    ax.set_title('%d' % (i + 1), color='green')
                else:
                    ax.set_title('%d' % (i + 1), color='red')
                print(img_path)
        except RuntimeError:
            for i in range(10):
                img_path = dataset.query[index[i]]
                print(img_path[0])
            print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

        save_dir = os.path.expanduser('~/Data/ReID_Visualization/')
        if args.dataset == 'aic_tracking':
            save_dir = os.path.join(save_dir, 'AIC19_tracking')
        elif args.dataset == 'veri':
            save_dir = os.path.join(save_dir, 'VeRi')
        elif args.dataset == 'aic_reid':
            save_dir = os.path.join(save_dir, 'AIC19_reid')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'show_{}.png'.format(selected))
        fig.savefig(save_path)
        print(save_path, 'has been output!')


if __name__ == '__main__':
    visualization(args)

# CUDA_VISIBLE_DEVICES=3 python visualization.py -d aic_tracking --logs-dir logs/ide_new/256/veri_vehicleid/train/5_fps/basis --query_index 1
