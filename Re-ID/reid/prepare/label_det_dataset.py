import datetime
import os

import numpy as np
import pandas as pd


def bbox_ious(boxA, boxB):
    if boxA.size == 0 or boxB.size == 0:
        return np.array([])
    boxA[:, 2:4] = np.array([boxA[:, 0] + boxA[:, 2], boxA[:, 1] + boxA[:, 3]]).T
    boxB[:, 2:4] = np.array([boxB[:, 0] + boxB[:, 2], boxB[:, 1] + boxB[:, 3]]).T
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(np.array([boxA[:, 0]]).T, boxB[:, 0])
    yA = np.maximum(np.array([boxA[:, 1]]).T, boxB[:, 1])
    xB = np.minimum(np.array([boxA[:, 2]]).T, boxB[:, 2])
    yB = np.minimum(np.array([boxA[:, 3]]).T, boxB[:, 3])
    # compute the area of intersection rectangle
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = np.array([(boxA[:, 2] - boxA[:, 0] + 1) * (boxA[:, 3] - boxA[:, 1] + 1)]).T
    boxBArea = (boxB[:, 2] - boxB[:, 0] + 1) * (boxB[:, 3] - boxB[:, 1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    ious = interArea / (boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return ious


def main(det_time='train', IoUthreshold=0.3):
    data_dir = os.path.expanduser('~/Data/AIC19/train')

    if det_time == 'train':
        scenes = ['S03', 'S04']
    elif det_time == 'trainval':
        scenes = ['S01', 'S03', 'S04']
    elif det_time == 'val':
        scenes = ['S01']
    else:
        scenes = None

    # loop for subsets
    for scene in scenes:
        scene_dir = os.path.join(data_dir, scene)
        # savedir = os.path.join(data_dir, 'labeled')
        # if not os.path.exists(savedir):
        #     os.mkdir(savedir)

        # loop for cameras
        for camera in os.listdir(scene_dir):
            gt_file_path = os.path.join(scene_dir, camera, 'gt', 'gt.txt')
            det_file_path = os.path.join(scene_dir, camera, 'det', 'det_ssd512.txt')
            gt_file = np.array(pd.read_csv(gt_file_path, header=None))
            det_file = np.array(pd.read_csv(det_file_path, header=None))
            # frame, id, bbox*4, score
            frames = np.unique(gt_file[:, 0])
            for frame in frames:
                gt_line_ids = np.where(gt_file[:, 0] == frame)[0]
                same_frame_gt_bboxs = gt_file[gt_line_ids, 2:6]
                det_line_ids = np.where(det_file[:, 0] == frame)[0]
                same_frame_det_bboxs = det_file[det_line_ids, 2:6]
                ious = bbox_ious(same_frame_gt_bboxs, same_frame_det_bboxs)
                if ious.size == 0:
                    continue
                label = np.argmax(ious, axis=1)
                det_file[det_line_ids[label], 1] = gt_file[gt_line_ids, 1]
                det_file[det_line_ids[np.max(ious, axis=0) < IoUthreshold], 1] = -1
                pass
            np.savetxt(os.path.join(scene_dir, camera, 'det', 'det_ssd512_labeled.txt'),
                       det_file, delimiter=',', fmt='%d')
            # np.savetxt(os.path.join(savedir, '{}_det_ssd512_labeled.txt'.format(camera)),
            #            det_file, delimiter=',', fmt='%d')

            print(camera, 'is completed')
        print(scene, 'is completed')


if __name__ == '__main__':
    print('{}'.format(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')))
    main(det_time='trainval')
    print('{}'.format(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')))
    print('Job Completed!')
