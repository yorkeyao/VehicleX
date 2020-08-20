import os
import os.path as osp
import re
from glob import glob

import h5py
import numpy as np
from sklearn.preprocessing import normalize

models = ['lr001', 'lr001_softmargin', 'lr001_colorjitter']
dirs = ['gt_all', 'gt_mini', ]  # 'test', 'trainval',

for data_dir in dirs:

    models_feat = {}
    models_header = {}
    for model in models:
        if data_dir == 'gt_mini':
            folder = osp.join('/home/houyz/Code/DeepCC/experiments', 'zju_{}_gt_trainval'.format(model))
        elif data_dir == 'gt_all':
            folder = osp.join('/home/houyz/Data/AIC19/L0-features', 'gt_features_zju_{}'.format(model))
        else:
            folder = osp.join('/home/houyz/Data/AIC19/L0-features',
                              'det_features_zju_{}_{}_ssd'.format(model, data_dir))
        fnames = sorted(glob(osp.join(folder, '*.h5')))

        pattern = re.compile(r'(\d+)')
        for fname in fnames:
            h5file = h5py.File(fname, 'r')
            data = np.array(h5file['emb'])
            cam = int(pattern.search(osp.basename(fname)).groups()[0])
            if cam not in models_feat:
                models_feat[cam] = np.array([])
                models_header[cam] = data[:, :3 if 'gt' in data_dir else 2]

            data = data[:, 3 if 'gt' in data_dir else 2:]
            data = normalize(data, axis=1)
            models_feat[cam] = np.hstack([models_feat[cam], data]) if models_feat[cam].size else data
    pass
    for cam in models_feat.keys():
        models_feat[cam] /= len(models) ** 0.5
        ensemble_feat = np.hstack([models_header[cam], models_feat[cam]])
        if data_dir == 'gt_mini':
            folder = osp.join('/home/houyz/Code/DeepCC/experiments', 'zju_lr001_ensemble_gt_trainval')
        elif data_dir == 'gt_all':
            folder = osp.join('/home/houyz/Data/AIC19/L0-features', 'gt_features_zju_lr001_ensemble')
        else:
            folder = osp.join('/home/houyz/Data/AIC19/L0-features',
                              'det_features_zju_lr001_ensemble_{}_ssd'.format(data_dir))

        output_fname = folder + '/features%d.h5' % cam
        if not osp.exists(folder):
            os.makedirs(folder)
        with h5py.File(output_fname, 'w') as f:
            f.create_dataset('emb', data=ensemble_feat, dtype=float, maxshape=(None, None))
            pass

    pass
