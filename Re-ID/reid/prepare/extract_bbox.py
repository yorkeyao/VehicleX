import datetime
import os
import os.path as osp

import cv2
import numpy as np
import psutil

path = '~/Data/AIC19/'
og_fps = 10


def get_bbox(type='gt', det_time='train', fps=10, det_type='ssd'):
    # type = ['gt','det','labeled']
    data_path = osp.join(osp.expanduser(path), 'test' if det_time == 'test' else 'train')
    save_path = osp.join(osp.expanduser('~/Data/AIC19/ALL_{}_bbox/'.format(type)), det_time)

    if type == 'gt' or type == 'labeled':
        save_path = osp.join(save_path, 'gt_bbox_{}_fps'.format(fps))
        fps_pooling = int(og_fps / fps)  # use minimal number of gt's to train ide model
    else:
        save_path = osp.join(save_path, det_type)

    os.makedirs(save_path, exist_ok=True)

    # scene selection for train/val
    if det_time == 'train':
        scenes = ['S03', 'S04']
    elif det_time == 'trainval':
        scenes = ['S01', 'S03', 'S04']
    elif det_time == 'val':
        scenes = ['S01']
    elif det_time == 'test' and type == 'gt':
        scenes = ['S02', 'S06']
    else:  # test
        scenes = os.listdir(data_path)

    for scene in scenes:
        scene_path = osp.join(data_path, scene)
        for camera_dir in os.listdir(scene_path):
            iCam = int(camera_dir[1:])
            # get bboxs
            if type == 'gt':
                if det_time == 'test':
                    bbox_filename = osp.join('/home/houyz/Code/DeepCC/experiments/aic_label_det/L3-identities',
                                             'cam{}_test.txt'.format(iCam))
                    delimiter = None
                else:
                    bbox_filename = osp.join(scene_path, camera_dir, 'gt', 'gt.txt')
                    delimiter = ','
            elif type == 'labeled':
                bbox_filename = osp.join(scene_path, camera_dir, 'det',
                                         'det_{}_labeled.txt'.format('ssd512' if det_type == 'ssd' else 'yolo3'))
                delimiter = ','
            else:  # det
                bbox_filename = osp.join(scene_path, camera_dir, 'det',
                                         'det_{}.txt'.format('ssd512' if det_type == 'ssd' else 'yolo3'))
                delimiter = ','
            bboxs = np.loadtxt(bbox_filename, delimiter=delimiter)
            if type == 'gt' or type == 'labeled':
                bboxs = bboxs[np.where(bboxs[:, 0] % fps_pooling == 0)[0], :]

            # get frame_pics
            video_file = osp.join(scene_path, camera_dir, 'vdo.avi')
            video_reader = cv2.VideoCapture(video_file)
            # get vcap property
            width = video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

            # enlarge by 40 pixel for detection
            if type == 'det' or type == 'labeled':
                bboxs[:, 2:4] = bboxs[:, 2:4] - 20
                bboxs[:, 4:6] = bboxs[:, 4:6] + 40

            # bboxs
            bbox_top = np.maximum(bboxs[:, 3], 0)
            bbox_bottom = np.minimum(bboxs[:, 3] + bboxs[:, 5], height - 1)
            bbox_left = np.maximum(bboxs[:, 2], 0)
            bbox_right = np.minimum(bboxs[:, 2] + bboxs[:, 4], width - 1)
            bboxs[:, 2:6] = np.stack((bbox_top, bbox_bottom, bbox_left, bbox_right), axis=1)

            # frame_pics = []
            frame_num = 0
            success = video_reader.isOpened()
            printed_img_count = 0
            while (success):
                assert psutil.virtual_memory().percent < 95, "reading video will be killed!!!!!!"

                success, frame_pic = video_reader.read()
                frame_num = frame_num + 1
                bboxs_in_frame = bboxs[bboxs[:, 0] == frame_num, :]

                for index in range(bboxs_in_frame.shape[0]):
                    frame = int(bboxs_in_frame[index, 0])
                    pid = int(bboxs_in_frame[index, 1])
                    bbox_top = int(bboxs_in_frame[index, 2])
                    bbox_bottom = int(bboxs_in_frame[index, 3])
                    bbox_left = int(bboxs_in_frame[index, 4])
                    bbox_right = int(bboxs_in_frame[index, 5])

                    bbox_pic = frame_pic[bbox_top:bbox_bottom, bbox_left:bbox_right]
                    if bbox_pic.size == 0:
                        continue

                    if type == 'gt' or type == 'labeled':
                        save_file = osp.join(save_path, "{:04d}_c{:02d}_f{:05d}.jpg".format(pid, iCam, frame))
                    else:
                        save_file = osp.join(save_path, 'c{:02d}_f{:05d}_{:03d}.jpg'.format(iCam, frame, index))

                    cv2.imwrite(save_file, bbox_pic)
                    cv2.waitKey(0)
                    printed_img_count += 1

                cv2.waitKey(0)
            video_reader.release()
            # assert printed_img_count == bboxs.shape[0]

            print(video_file, 'completed!')
        print(scene, 'completed!')
    print(save_path, 'complete d!')


if __name__ == '__main__':
    print('{}'.format(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')))
    get_bbox(type='gt', fps=10, det_time='trainval')
    # get_bbox(type='labeled', det_time='trainval', fps=1)
    get_bbox(type='det', det_time='trainval', det_type='ssd')
    get_bbox(type='det', det_time='test', det_type='ssd')
    print('{}'.format(datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')))
    print('Job Completed!')
