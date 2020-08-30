from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import models.models as models
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision
import utils.utils as utils

"""params"""
lr = 0.0002
crop_size_w = 128
crop_size_h = 256
batch_size = 1

Ga = models.Generator()
Gb = models.Generator()
transform = transforms.Compose([
    transforms.Resize((crop_size_h, crop_size_w)),
 #   transforms.RandomCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
a_test_dir = '../datasets/summer2winter_yosemite/testA'  #'../Cycledata/market2duke/link_testA'
b_test_dir = '../datasets/summer2winter_yosemite/testB'  #'../Cycledata/market2duke/link_testB'


a_test_data = dsets.ImageFolder(a_test_dir, transform=transform)
b_test_data = dsets.ImageFolder(b_test_dir, transform=transform)
a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=batch_size, num_workers=0)
b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=batch_size, num_workers=0)

Ga.load_state_dict(torch.load('./checkpoints/cyclegan/Epoch_(6).ckpt', map_location=lambda storage, loc: storage)['Ga'])
Gb.load_state_dict(torch.load('./checkpoints/cyclegan/Epoch_(6).ckpt', map_location=lambda storage, loc: storage)['Gb'])
dirpatha, a, filenamea = os.walk('../datasets/summer2winter_yosemite/testA/0').__next__()
filenamea.sort()
dirpathb, b, filenameb = os.walk('../datasets/summer2winter_yosemite/testB/0').__next__()
filenameb.sort()

save_dir_a = './market/bounding_box_train_cyclegan_new/'
save_dir_b = './duke/bounding_box_train_cyclegan_new/'
utils.mkdir([save_dir_a, save_dir_b])

Ga = Ga.cuda()
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
#     Gb.eval()
#     b_test = b_test[0]
#     b_test = Variable(b_test.cuda(), volatile=True)
#     b_out = Ga(b_test)
#     torchvision.utils.save_image((b_out.data + 1) / 2.0, save_dir_b + filenameb[i], padding=0)
#     i+=1
#     if i%100 ==0: print(i)
