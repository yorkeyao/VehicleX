import numpy as np
import json
import random


def write_json (json_path, cam_info, attribute_list, task_info, best_score):
    idx = 0
    for attribute in cam_info['attributes'].items():
        attribute_name = attribute[0]
        attribute_content = attribute[1]
        if attribute_content[0] == 'Gaussian Mixture':
            attribute_content[2] = [float(i) for i in attribute_list[idx: idx + len(attribute_content[2])]]
            idx += len(attribute_content[2])
        if attribute_content[0] == 'Gaussian':
            attribute_content[2] = float(attribute_list[idx])
            idx += 1
    cam_info["FD distance"] = best_score
    with open(json_path, 'w') as outfile:
        json.dump(task_info, outfile, indent=4)

def sample_values(attribute_list, variance_list, order_list, dataset_size):
    cnt = 0
    for attribute_name in order_list:
        if attribute_name == "orientation":
            angle = np.random.permutation(ancestral_sampler(mu = attribute_list[cnt:cnt+6], sigma = variance_list[:6], size=dataset_size * 3))
            cnt = cnt + 6
        if attribute_name == "light intensity":
            temp_intensity_list = np.random.normal(loc=attribute_list[cnt], scale=variance_list[cnt], size=dataset_size + 100)  
            cnt = cnt + 1
        if attribute_name == "light direction":
            temp_light_direction_x_list = np.random.normal(loc=attribute_list[cnt], scale=variance_list[cnt], size=dataset_size + 100)
            cnt = cnt + 1
        if attribute_name == "camera height":
            Cam_height_list = np.random.normal(loc=attribute_list[cnt], scale=variance_list[cnt], size=dataset_size + 100) 
            cnt = cnt + 1
        if attribute_name == "camera distance":
            Cam_distance_y_list = np.random.normal(loc=attribute_list[9], scale=variance_list[9], size=dataset_size + 100) 
            cnt = cnt + 1
    return angle, temp_intensity_list, temp_light_direction_x_list, Cam_height_list, Cam_distance_y_list
    
def get_color_distribution(id_num, model_num):
    vehicle_info = {}
    vehicle_info['model'] = []
    vehicle_info['R'] = []
    vehicle_info['G'] = []
    vehicle_info['B'] = []
    for i in range(id_num):
        vehicle_info['model'].append(i % model_num)
        vehicle_info['R'].append(float(random.randint(0,255)) / 255)
        vehicle_info['G'].append(float(random.randint(0,255)) / 255)
        vehicle_info['B'].append(float(random.randint(0,255)) / 255)
    return vehicle_info

def get_cam_attr(cam_info):
    control_list = []
    attribute_list = []
    variance_list = []
    order_list = []
    for attribute in cam_info['attributes'].items():
        attribute_name = attribute[0]
        order_list.append(attribute_name)
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

def ancestral_sampler_fix_sigma(mu=[0, 180], size=1):
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

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
