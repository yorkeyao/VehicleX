import numpy as np
import random
from skimage import io
from skimage import img_as_ubyte
import argparse
import os
import sys
from xml.dom.minidom import Document
from mlagents.envs.environment import UnityEnvironment

parser = argparse.ArgumentParser(description='outputs')
parser.add_argument('--data_dir',default='./outputs/',type=str, help='./output path')

opt = parser.parse_args()
output_dir = opt.data_dir

if not os.path.isdir(output_dir):  
    os.mkdir(output_dir)

env_name = "./Build-win/VehicleX" # is ./Build-linux/VehicleX if linux is used
train_mode = True  # Whether to run the environment in training or inference mode

print("Python version:")
print(sys.version)

# check Python version
if (sys.version_info[0] < 3):
    raise Exception("ERROR: ML-Agents Toolkit (v0.3 onwards) requires Python 3")

env = UnityEnvironment(file_name=env_name)

# Set the default brain to work with
default_brain = env.brain_names[0]
brain = env.brains[default_brain]

def ancestral_sampler_1(pi=[0.5, 0.5], 
                      mu=[0, 180], sigma=[20, 20], size=1):
    sigma = [20 for i in range(6)]
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

print ("Begin generation")

doc = Document()
TrainingImages = doc.createElement('TrainingImages')
TrainingImages.setAttribute("Version", "1.0")  
doc.appendChild(TrainingImages)
Items = doc.createElement('Items')
Items.setAttribute("number", "-")  
TrainingImages.appendChild(Items)

training = False

def Get_Save_images_by_attributes(attribute_list, cam_id):
    z = 0
    cnt = 0
    angle = np.random.permutation (ancestral_sampler_1(pi = [], mu = attribute_list[:6], size=dataset_size + 100))
    temp_intensity_list = np.random.normal(loc=attribute_list[6], scale=np.sqrt(0.4), size=dataset_size + 100)  
    temp_light_direction_x_list = np.random.normal(loc=attribute_list[7], scale=np.sqrt(50), size=dataset_size + 100)
    Cam_height_list = np.random.normal(loc=attribute_list[8], scale=2, size=dataset_size + 100) 
    Cam_distance_y_list = np.random.normal(loc=attribute_list[9], scale=3, size=dataset_size + 100) 
    cam_str = "c" + str(i).zfill(3)

    env_info = env.reset(train_mode=train_mode)[default_brain]
    images = []
    while cnt < dataset_size:
        done = False
        if angle[z] > 360:
            angle[z] = angle[z] % 360
        while angle[z] < 0:
            angle[z] =  angle[z] + 360
        
        Cam_distance_x = random.uniform(-5, 5)
        scene_id = random.randint(1,59) 
        env_info = env.step([[angle[z], temp_intensity_list[z], temp_light_direction_x_list[z], Cam_distance_y_list[z], Cam_distance_x, Cam_height_list[z], scene_id, training]])[default_brain] 
        done = env_info.local_done[0]
        car_id = int(env_info.vector_observations[0][4])
        color_id = int(env_info.vector_observations[0][5])
        type_id = int(env_info.vector_observations[0][6])
        if done:
            env_info = env.reset(train_mode=train_mode)[default_brain]
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
            Items.appendChild(Item)
        z = z + 1

dataset_size = 3500
attribute_list = [270, 90, 90, 270, 240, 300, 1.2, 45, 7, 1]

for i in range (1, 100):
    Get_Save_images_by_attributes(attribute_list, i)

filename = "train_label.xml"

with open(filename, 'wb') as f:
    f.write(doc.toprettyxml(indent='\t', newl = "\n", encoding='utf-8'))
f.close()  
