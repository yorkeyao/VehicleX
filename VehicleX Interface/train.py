import os
import shutil
from glob import glob
import matplotlib.pyplot as plt
import cv2
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import random
from torch.autograd import Variable
from torch.distributions import MultivariateNormal
from skimage import io
from skimage import img_as_ubyte
from collections import Counter
from scipy import misc
import copy
from domain_gap.fd_score import calculate_fd_given_paths
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image
import shutil
import argparse
from mlagents.envs.environment import UnityEnvironment
import json
from utils import *

parser = argparse.ArgumentParser(description='outputs')
parser.add_argument('--setting',default='./settings/VehicleID.json',type=str, help='./target dataset and attribute definition')
parser.add_argument('--output',default='./settings/VehicleID-test.json',type=str, help='./output attributes')
parser.add_argument('--save_dir',default='./datasets/temporary/',type=str, help='./temporary dataset path')
parser.add_argument('--train_mode',type=str2bool, nargs='?',
                        const=True, default=True, help="Whether to run the environment in training or inference mode")
parser.add_argument('--FD_model', type=str, default='inception', choices=['inception', 'posenet'], help='model to calculate FD distance')
parser.add_argument('--log_path', type=str, default='./logs/log.txt', help='Name of the Unity environment binary to launch')
parser.add_argument('--env_path', type=str, default='./Build-win/VehicleX')
parser.add_argument('--sample_size', type=int, default=400, help="temporary dataset generation size")
parser.add_argument('--port', type=int, default=5005, help="port to connect Unity environment")

opt = parser.parse_args()
env_name = opt.env_path
train_mode = opt.train_mode 
save_dir = opt.save_dir
dataset_size = opt.sample_size
random.seed(1)
np.random.seed(1)

print("Python version:")
print(sys.version)

# check Python version
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")

env = UnityEnvironment(file_name=env_name, base_port=opt.port)
# Set the default brain to work with
default_brain = env.brain_names[0]
brain = env.brains[default_brain]

def get_images_by_attributes(attribute_list, variance_list, save_dir):
    angle = np.random.permutation(ancestral_sampler(mu = attribute_list[:6], sigma = variance_list[:6], size=dataset_size * 3))
    temp_intensity_list = np.random.normal(loc=attribute_list[6], scale=variance_list[6], size=dataset_size + 100)  
    temp_light_direction_x_list = np.random.normal(loc=attribute_list[7], scale=variance_list[7], size=dataset_size + 100)
    Cam_height_list = np.random.normal(loc=attribute_list[8], scale=variance_list[8], size=dataset_size + 100) 
    Cam_distance_y_list = np.random.normal(loc=attribute_list[9], scale=variance_list[9], size=dataset_size + 100) 
    env_info = env.reset(train_mode=True)[default_brain]
    images = []
    for z in range (dataset_size):
        done = False
        if angle[len(images)] > 360:
            angle[len(images)] = angle[len(images)] % 360
        while angle[len(images)] < 0:
            angle[len(images)] =  angle[len(images)] + 360
        Cam_distance_x = 0 
        scene_id = 3
        env_info = env.step([[angle[len(images)], temp_intensity_list[len(images)], temp_light_direction_x_list[len(images)], Cam_distance_y_list[len(images)], Cam_distance_x, Cam_height_list[len(images)], scene_id, train_mode]])[default_brain] 
        if not os.path.exists(save_dir): 
            os.mkdir(save_dir)
        done = env_info.local_done[0]
        if done:
            env_info = env.reset(train_mode=True)[default_brain]
            continue
        observation_gray = np.array(env_info.visual_observations[1])
        x, y = (observation_gray[0,:,:,0] > 0).nonzero()
        observation = np.array(env_info.visual_observations[0])
        if observation.shape[3] == 3 and len(y) > 0 and min(y) > 10 and min(x) > 10:
            ori_img = np.array(observation[0,min(x):max(x),min(y):max(y),:])
            resized = misc.imresize(ori_img, size=[256, 256])
            images.append(resized)
            io.imsave(save_dir + "/" +str(z) + ".jpg",img_as_ubyte(ori_img))
    
def get_reward(images, best_list, attribute_list, best_score, output_log, cam_id, target_path):
    samples_generated = np.array(images).astype(np.float32)
    fd, npz_path = calculate_fd_given_paths([target_path, save_dir], opt)
    if npz_path != None:
        target_path = npz_path
    if fd < best_score:
        best_list = copy.deepcopy(attribute_list)
        best_score = fd
    print ("Cam:", cam_id, "current: ", attribute_list, fd, "best:", best_list, "score", best_score)
    output_log.write("Cam:" + str(cam_id) + "current: " + str(attribute_list) + str(fd) + "best:" + str(best_list) + "score" + str(best_score) + "\n")   
    return best_score, copy.deepcopy(best_list), target_path

with open(opt.setting) as f:
    task_info = json.load(f)
    
output_log = open(opt.log_path,'w')
    
for cam_id in range(task_info['camera number']):
    cam_info = task_info['camera list'][cam_id]
    control_list, attribute_list, variance_list = get_cam_attr(cam_info)
    target_path = cam_info['target dir']  
    for epoch in range(2):
        for i in range(0, len(attribute_list)): 
            best_score = float('inf')
            best_list = []
            for j in control_list[i]:
                attribute_list[i] = j
                images = get_images_by_attributes(attribute_list, variance_list, save_dir)
                best_score, best_list, target_path = get_reward(images, best_list, attribute_list, best_score, output_log, cam_info['camera id'], target_path)  
            attribute_list = copy.deepcopy(best_list)
            write_json(opt.output, cam_info, attribute_list, task_info, best_score)
output_log.close()
env.close()

