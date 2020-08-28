import numpy as np
import random
from skimage import io
from skimage import img_as_ubyte
import argparse
import os
import sys
from xml.dom.minidom import Document
import json
from mlagents.envs.environment import UnityEnvironment
from utils import * 

parser = argparse.ArgumentParser(description='outputs')
parser.add_argument('--setting',default='./settings/VeRi-out-final.json',type=str, help='./target dataset and attribute definition')
parser.add_argument('--train_mode',type=str2bool, nargs='?',
                        const=True, default=False, help="Whether to run the environment in training or inference mode")
parser.add_argument('--env_path', type=str, default='./Build-win/VehicleX')
parser.add_argument('--out_lab_file', type=str, default='VeRi_label.xml')

opt = parser.parse_args()
env_name = opt.env_path # is ./Build-linux/VehicleX if linux is used
train_mode = opt.train_mode  # Whether to run the environment in training or inference mode

print("Python version:")
print(sys.version)

# check Python version
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")
if (not os.path.exists("./Background_imgs") and train_mode == False):
    raise Exception("The inference mode requre background images")

# env = UnityEnvironment(file_name=None)
env = UnityEnvironment(file_name=env_name) # is None if you use Unity Editor
# Set the default brain to work with

default_brain = env.brain_names[0]
brain = env.brains[default_brain]
distance_bias = 12.11

print ("Begin generation")

doc = Document()
TrainingImages = doc.createElement('TrainingImages')
TrainingImages.setAttribute("Version", "1.0")  
doc.appendChild(TrainingImages)
Items = doc.createElement('Items')
Items.setAttribute("number", "-")  
TrainingImages.appendChild(Items)

def get_save_images_by_attributes(attribute_list, control_list, cam_id, dataset_size, output_dir):
    if not os.path.isdir(output_dir):  
        os.mkdir(output_dir)
    z = 0
    cnt = 0
    angle = np.random.permutation (ancestral_sampler_fix_sigma(mu = attribute_list[:6], size=dataset_size * 3))
    temp_intensity_list = np.random.normal(loc=attribute_list[6], scale=np.sqrt(0.4), size=dataset_size * 3)  
    temp_light_direction_x_list = np.random.normal(loc=attribute_list[7], scale=np.sqrt(50), size=dataset_size * 3)
    Cam_height_list = np.random.normal(loc=attribute_list[8], scale=2, size=dataset_size * 3) 
    Cam_distance_y_list = np.random.normal(loc=attribute_list[9], scale=3, size=dataset_size * 3) 
    cam_str = "c" + str(cam_id).zfill(3)
    env_info = env.reset(train_mode=True)[default_brain]
    images = []
    while cnt < dataset_size:
        done = False
        angle[z] = angle[z] % 360
        temp_intensity_list[z] = min(max(min(control_list[6]), temp_intensity_list[z]), max(control_list[6]))
        temp_light_direction_x_list[z] = min(max(min(control_list[7]), temp_light_direction_x_list[z]), max(control_list[7]))
        Cam_height_list[z] = min(max(min(control_list[8]), Cam_height_list[z]), max(control_list[8]))
        Cam_distance_y_list[z] = min(max(min(control_list[9]), Cam_distance_y_list[z]), max(control_list[9]))
        Cam_distance_x = random.uniform(-5, 5)
        scene_id = random.randint(1,59) 
        env_info = env.step([[angle[z], temp_intensity_list[z], temp_light_direction_x_list[z], Cam_distance_y_list[z], Cam_distance_x, Cam_height_list[z], scene_id, train_mode]])[default_brain] 
        done = env_info.local_done[0]
        car_id = int(env_info.vector_observations[0][4])
        color_id = int(env_info.vector_observations[0][5])
        type_id = int(env_info.vector_observations[0][6])
        if done:
            env_info = env.reset(train_mode=True)[default_brain]
            continue
        observation_gray = np.array(env_info.visual_observations[1])
        x, y = (observation_gray[0,:,:,0] > 0).nonzero()
        observation = np.array(env_info.visual_observations[0])
        if observation.shape[3] == 3 and len(y) > 0 and min(y) > 10 and min(x) > 10:
            print (cam_id, cnt, angle[z], temp_intensity_list[z], temp_light_direction_x_list[z], Cam_distance_y_list[z], Cam_distance_x, Cam_height_list[z], scene_id)
            ori_img = observation[0,min(x)-10:max(x)+10,min(y)-10:max(y)+10,:]
            cnt = cnt + 1
            filename = "0" + str(car_id).zfill(4) + "_" + cam_str + "_" + str(cnt) + ".jpg"
            io.imsave(output_dir + filename,img_as_ubyte(ori_img))
            Item = doc.createElement('Item')
            Item.setAttribute("typeID", str(type_id))  
            Item.setAttribute("imageName", filename)   
            Item.setAttribute("cameraID", cam_str)  
            Item.setAttribute("vehicleID", str(car_id).zfill(4))  
            Item.setAttribute("colorID", str(color_id))  
            Item.setAttribute("orientation",str(round(angle[z], 1)))
            Item.setAttribute("lightInt",str(round(temp_intensity_list[z], 1)))
            Item.setAttribute("lightDir",str(round(temp_light_direction_x_list[z], 1)))
            Item.setAttribute("camHei",str(round(Cam_height_list[z], 1)))
            Item.setAttribute("camDis",str(round(Cam_distance_y_list[z] + distance_bias, 1)))
            Items.appendChild(Item)
        z = z + 1

with open(opt.setting) as f:
    task_info = json.load(f)

for cam_id in range(task_info['camera number']):
    cam_info = task_info['camera list'][cam_id]
    control_list, attribute_list, variance_list = get_cam_attr(cam_info)
    get_save_images_by_attributes(attribute_list, control_list, int(cam_info["camera id"]), cam_info['data size'], cam_info['output dir'])

with open(opt.out_lab_file, 'wb') as f:
    f.write(doc.toprettyxml(indent='\t', newl = "\n", encoding='utf-8'))
f.close()  
