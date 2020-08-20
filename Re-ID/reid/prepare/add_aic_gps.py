import os
import os.path as osp

import cv2
import numpy as np
from numpy.linalg import inv

data_path = 'D:/Data/AIC19/' if os.name == 'nt' else osp.expanduser('~/Data/AIC19/')
scenes = [1, 2, 3, 4, 5]
folder_by_scene = {1: 'train',
                   2: 'test',
                   3: 'train',
                   4: 'train',
                   5: 'test', }
world_centers = {1: np.array([42.525678, -90.723601]),
                 2: np.array([42.491916, -90.723723]),
                 3: np.array([42.498780, -90.686393]),
                 4: np.array([42.498780, -90.686393]),
                 5: np.array([42.498780, -90.686393]), }

world_scale = 6371000 / 180 * np.pi


def image2gps(feet_pos, parameters, scene):
    feet_pos = feet_pos.reshape(-1, 1, 2)
    if 'intrinsic' in parameters:
        # Have to provide P matrix for appropriate scaling
        feet_pos = cv2.undistortPoints(feet_pos, parameters['intrinsic'], parameters['distortion'],
                                       P=parameters['intrinsic'])
    world_pos = cv2.perspectiveTransform(feet_pos, inv(parameters['homography'])).reshape(-1, 2)
    world_pos = (world_pos - world_centers[scene]) * world_scale
    return world_pos[:, ::-1]


def gps2image(world_pos, parameters, scene):
    world_pos = world_pos[:, ::-1] / world_scale + world_centers[scene]
    world_pos = world_pos.reshape(-1, 1, 2)
    feet_pos = cv2.perspectiveTransform(world_pos, parameters['homography']).reshape(-1, 2)
    if 'intrinsic' in parameters:
        rvec = np.array([0, 0, 0], dtype=np.float32)
        tvec = np.array([0, 0, 0], dtype=np.float32)
        feet_pos, _ = cv2.projectPoints(
            np.matmul(inv(parameters['intrinsic']),
                      np.concatenate((feet_pos, np.ones(feet_pos.shape[0]).reshape(-1, 1)), axis=1).T,
                      ).T,
            rvec, tvec, parameters['intrinsic'], parameters['distortion'])
    return feet_pos


if __name__ == '__main__':
    for scene in scenes:
        scene_path = osp.join(data_path, folder_by_scene[scene], 'S{:02d}'.format(scene))
        frame_offset_fname = osp.join(data_path, 'cam_timestamp', 'S{:02d}.txt'.format(scene))
        frame_offset = {}
        with open(frame_offset_fname) as f:
            for line in f:
                (key, val) = line.split(' ')
                key = int(key[1:])
                val = 10 * float(val)
                frame_offset[key] = val
        for camera_dir in sorted(os.listdir(scene_path)):
            iCam = int(camera_dir[1:])
            calibration_fname = osp.join(data_path, 'calibration', camera_dir, 'calibration.txt')
            parameters = {}
            with open(calibration_fname) as f:
                for line in f:
                    (key, val) = line.split(':')
                    key = key.split(' ')[0].lower()
                    if key == 'reprojection': key = 'error'
                    if ';' in val:
                        val = np.fromstring(val.replace(';', ' '), dtype=float, sep=' ').reshape([3, 3])
                    else:
                        val = np.fromstring(val, dtype=float, sep=' ')
                    parameters[key] = val
            pass
            bbox_types = ['gt', 'det'] if folder_by_scene[scene] == 'train' else ['det']
            for bbox_type in bbox_types:
                bbox_file = osp.join(scene_path, camera_dir, bbox_type,
                                     'gt.txt' if bbox_type == 'gt' else 'det_ssd512.txt')
                bboxs = np.loadtxt(bbox_file, delimiter=',')
                feet_pos = np.array([bboxs[:, 2] + bboxs[:, 4] / 2, bboxs[:, 3] + bboxs[:, 5]]).T
                world_pos = image2gps(feet_pos, parameters, scene)
                new_feet_pos = gps2image(world_pos, parameters, scene)
                error = np.mean(np.sum(new_feet_pos - feet_pos, axis=1))

                bboxs[:, 7] = iCam
                bboxs[:, 8] = bboxs[:, 0] + frame_offset[iCam]
                bboxs = bboxs[:, :9]
                bboxs = np.concatenate((bboxs, world_pos), axis=1)
                bbox_gps_file = osp.join(scene_path, camera_dir, bbox_type,
                                         'gt_gps.txt' if bbox_type == 'gt' else 'det_ssd512_gps.txt')
                np.savetxt(bbox_gps_file, bboxs, delimiter=',', fmt='%g')
                pass
