import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import os
import models.models_spgan as models
import torch
import torch._utils
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision
import utils.utils as utils
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='spgan test')
parser.add_argument('--source_path',default='../datasets/testA/',type=str, help='./source dataset')
parser.add_argument('--save_path',default='./testA(VeRi_style)/',type=str, help='./source dataset')
parser.add_argument('--checkpoint',default='./checkpoints/sys2VeRi.ckpt',type=str, help='./source dataset')

opt = parser.parse_args()

"""params"""
lr = 0.0002
crop_size_w = 256
crop_size_h = 256
batch_size = 1

device = torch.device('cuda')

# Ga = models.Generator()
# Gb = models.Generator()

# Ga = nn.DataParallel(models.Generator()).to(device)
Gb = nn.DataParallel(models.Generator()).to(device)

transform = transforms.Compose([
    transforms.Resize((crop_size_h, crop_size_w)),
 #   transforms.RandomCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

a_test_dir = opt.source_path

os.symlink(os.path.abspath(a_test_dir), os.path.join(a_test_dir, '0'))
# b_test_dir = '../datasets/sys2veri/testB'
a_test_data = dsets.ImageFolder(a_test_dir, transform=transform)
# b_test_data = dsets.ImageFolder(b_test_dir, transform=transform)
a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=batch_size, num_workers=0)
# b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=batch_size, num_workers=0)

ckpt_dir = opt.checkpoint
ckpt = utils.load_checkpoint(ckpt_dir)

# Ga.load_state_dict(ckpt['Ga'])
Gb.load_state_dict(ckpt['Gb'])

# Ga.load_state_dict(torch.load('./checkpoints/spgan_last/Epoch_(1).ckpt', map_location=lambda storage, loc: storage)['Ga'])
# Gb.load_state_dict(torch.load('./checkpoints/spgan_last/Epoch_(1).ckpt', map_location=lambda storage, loc: storage)['Gb'])
dirpatha, a, filenamea = os.walk(a_test_dir + '0').__next__()
filenamea.sort()
# dirpathb, b, filenameb = os.walk('./datasets/sys2veri/testB/0').__next__()
# filenameb.sort()

save_dir_a = opt.save_path
utils.mkdir(save_dir_a)

# Ga = Ga.cuda()
Gb = Gb.cuda()

i = 0

for  a_test in (a_test_loader):
    Gb.eval()
    a_test = a_test[0]
    a_test = Variable(a_test.cuda(), volatile=True)
    a_out = Gb(a_test)
    for j in range(batch_size):
        torchvision.utils.save_image((a_out.data[j] + 1) / 2.0, save_dir_a + filenamea[i+j], padding=0)
    i+=batch_size
    if i%128 ==0: print(i)

# i = 0
# for  b_test in (b_test_loader):
#     Ga.eval()
#     b_test = b_test[0]
#     b_test = Variable(b_test.cuda(), volatile=True)
#     b_out = Ga(b_test)
#     for j in range(batch_size):
    
#         torchvision.utils.save_image((b_out.data[j] + 1) / 2.0, save_dir_b + filenameb[i+j], padding=0)
#     i+=batch_size
#     if i%128 ==0: print(i)