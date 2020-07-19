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
import random
from domain_gap.fd_score import calculate_fd_given_paths
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image
import shutil
import argparse
from mlagents.envs.environment import UnityEnvironment
import json

parser = argparse.ArgumentParser(description='outputs')
parser.add_argument('--setting',default='./VehicleID.json',type=str, help='./target dataset and attribute definition')
parser.add_argument('--output',default='./VehicleID-out.json',type=str, help='./output attributes')
parser.add_argument('--save_dir',default='./datasets/temporary/',type=str, help='./temporary dataset path')
parser.add_argument('--train_mode',default=True, help='Whether to run the environment in training or inference mode')
parser.add_argument('--FD_model', type=str, default='inception', choices=['posenet', 'inception'], help='model to calculate FD distance')
parser.add_argument('--log_path', type=str, default='.log.txt', help='Name of the Unity environment binary to launch')
parser.add_argument('--env_path', type=str, default='./Build-win/Unity Environment')
parser.add_argument('--sample_size', type=int, default=400, help="temporary dataset generation size")

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

env = UnityEnvironment(file_name=env_name)
# Set the default brain to work with
default_brain = env.brain_names[0]
brain = env.brains[default_brain]


def ancestral_sampler(mu=[0, 180], sigma=[20, 20], size=1): 
    pi = [0.16 for i in range(6)]
    sample = []
    z_list = np.random.uniform(size=size)    
    low = 0 # low bound of a pi interval
    high = 0 # higg bound of a pi interval
    for index in range(len(pi)):
        if index >0:
            low += pi[index - 1]
        high += pi[index]
        s = len([z for z in z_list if low <= z < high])
        sample.extend(np.random.normal(loc=mu[index], scale=np.sqrt(sigma[index]), size=s))
    return sample

def make_square(image, max_dim = 512):
    h, w = image.shape[:2]
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    image = np.pad(image, padding, mode='constant', constant_values=0)
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image

def Get_images_by_attributes(attribute_list, variance_list):
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
        scene_id = 1 
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
    
def get_reward(images, best_list, attribute_list, best_score, output_log, cam_id, mode):
    samples_generated = np.array(images).astype(np.float32)
    fd = calculate_fd_given_paths([target_path, save_dir], opt)
    if fd < best_score:
        best_list = copy.deepcopy(attribute_list)
        best_score = fd
    print ("Cam:", cam_id, "current: ", mode, attribute_list, fd, "best:", best_list, "score", best_score)
    output_log.write("Cam:" + str(cam_id) + "current: " + mode + str(attribute_list) + str(fd) + "best:" + str(best_list) + "score" + str(best_score) + "\n")   
    return best_score, copy.deepcopy(best_list)

def get_cam_attr(cam_info):
    control_list = []
    attribute_list = []
    variance_list = []
    for attribute in cam_info['attributes'].items():
        attribute_name = attribute[0]
        attribute_content = attribute[1]
        if attribute_content[0] == 'Gaussian Mixture':
            range_info = attribute_content[1]
            mean_list = attribute_content[2]
            var_list = attribute_content[3]
            control_list.extend([np.arange(range_info[0], range_info[1], range_info[2]) for i in range (len(mean_list))])
            attribute_list.extend(mean_list)
            variance_list.extend(var_list)
        if attribute_content[0] == 'Gaussian':
            range_info = attribute_content[1]
            mean_list = attribute_content[2]
            var_list = attribute_content[3]
            control_list.append (np.arange(range_info[0], range_info[1], range_info[2]))
            attribute_list.append (mean_list)
            variance_list.append (var_list)
    return control_list, attribute_list, variance_list

with open(opt.setting) as f:
    task_info = json.load(f)

def write_json (json_path, cam_info, attribute_list, task_info):
    idx = 0
    for attribute in cam_info['attributes'].items():
        attribute_name = attribute[0]
        attribute_content = attribute[1]
        if attribute_content[0] == 'Gaussian Mixture':
            attribute_content[2] = [int(i) for i in attribute_list[idx: idx + len(attribute_content[2])]]
            idx += len(attribute_content[2])
        if attribute_content[0] == 'Gaussian':
            attribute_content[2] = int(attribute_list[idx])
            idx += 1
    with open(json_path, 'w') as outfile:
        json.dump(task_info, outfile, indent=4)
    
output_log = open(opt.log_path,'w')
    
for cam_id in range(task_info['camera number']):
    cam_info = task_info['camera list'][cam_id]
    control_list, attribute_list, variance_list = get_cam_attr(cam_info)
    target_path = cam_info['target dir']  
    for epoch in range(2):
        for i in range(0, len(attribute_list)): 
            if cam_info["FD distance"] == "None":
                best_score = float('inf')
            else:
                best_score = cam_info["FD distance"]
            best_list = []
            for j in control_list[i]:
                attribute_list[i] = j
                images = Get_images_by_attributes(attribute_list, variance_list)
                best_score, best_list = get_reward(images, best_list, attribute_list, best_score, output_log, cam_id, "mean_")  
            attribute_list = copy.deepcopy(best_list)
            write_json(opt.output, cam_info, attribute_list, task_info)
output_log.close()
env.close()

