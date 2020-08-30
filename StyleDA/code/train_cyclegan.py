# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import itertools

import models.models as models
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils.utils as utils
from PIL import Image
from utils.logger import Logger
import numpy as np

""" gpu """
gpu_id = [0, 1]
utils.cuda_devices(gpu_id)

""" param """
epochs = 6
batch_size = 3
lr = 0.0002
dataset_dir = '../datasets/summer2winter_yosemite'
use_tensorboard = 1
if use_tensorboard:
    log_dir = './checkpoints/cyclegan'
    utils.mkdir(log_dir)
    Logger = Logger(log_dir)

""" data """
load_size_w = 144
load_size_h = 286
crop_size_w = 128
crop_size_h = 256
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.Resize((load_size_h, load_size_w), Image.BICUBIC),
     transforms.RandomCrop((crop_size_h, crop_size_w)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
	 
test_transform = transforms.Compose(
    [transforms.Resize((crop_size_h, crop_size_w), Image.BICUBIC), 
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
	 
dataset_dirs = utils.reorganize(dataset_dir)
a_data = dsets.ImageFolder(dataset_dirs['trainA'], transform=transform)
b_data = dsets.ImageFolder(dataset_dirs['trainB'], transform=transform)
a_test_data = dsets.ImageFolder(dataset_dirs['testA'], transform=test_transform)
b_test_data = dsets.ImageFolder(dataset_dirs['testB'], transform=test_transform)
a_loader = torch.utils.data.DataLoader(a_data, batch_size=batch_size, shuffle=True, num_workers=0)
b_loader = torch.utils.data.DataLoader(b_data, batch_size=batch_size, shuffle=True, num_workers=0)
a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=1, shuffle=True, num_workers=0)
b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=1, shuffle=True, num_workers=0)

a_fake_pool = utils.ItemPool()
b_fake_pool = utils.ItemPool()


""" model """
Da = models.Discriminator()
Db = models.Discriminator()
Ga = models.Generator()
Gb = models.Generator()
MSE = nn.MSELoss()
L1 = nn.L1Loss()
utils.cuda([Da, Db, Ga, Gb])

da_optimizer = torch.optim.Adam(Da.parameters(), lr=lr, betas=(0.5, 0.999))
db_optimizer = torch.optim.Adam(Db.parameters(), lr=lr, betas=(0.5, 0.999))
ga_optimizer = torch.optim.Adam(Ga.parameters(), lr=lr, betas=(0.5, 0.999))
gb_optimizer = torch.optim.Adam(Gb.parameters(), lr=lr, betas=(0.5, 0.999))


""" load checkpoint """
ckpt_dir = './checkpoints/cyclegan'
utils.mkdir(ckpt_dir)
try:
    ckpt = utils.load_checkpoint(ckpt_dir)
    start_epoch = ckpt['epoch']
    Da.load_state_dict(ckpt['Da'])
    Db.load_state_dict(ckpt['Db'])
    Ga.load_state_dict(ckpt['Ga'])
    Gb.load_state_dict(ckpt['Gb'])
    da_optimizer.load_state_dict(ckpt['da_optimizer'])
    db_optimizer.load_state_dict(ckpt['db_optimizer'])
    ga_optimizer.load_state_dict(ckpt['ga_optimizer'])
    gb_optimizer.load_state_dict(ckpt['gb_optimizer'])
except:
    print(' [*] No checkpoint!')
    start_epoch = 0


""" run """
loss = {}
for epoch in range(start_epoch, epochs):
    for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
        # step
        # print (len(a_real))
        if len(a_real) < batch_size:
            continue
        # print (np.shape (a_real[0]))

        # print (np.shape (b_real[0]))
        step = epoch * min(len(a_loader), len(b_loader)) + i + 1

        # set train
        Ga.train()
        Gb.train()

        # leaves
        a_real = Variable(a_real[0])
        b_real = Variable(b_real[0])
        a_real, b_real = utils.cuda([a_real, b_real])

        # train G
        a_fake = Ga(b_real)
        b_fake = Gb(a_real)

        a_rec = Ga(b_fake)
        b_rec = Gb(a_fake)

        # gen losses
        a_f_dis = Da(a_fake)
        b_f_dis = Db(b_fake)
        r_label = utils.cuda(Variable(torch.ones(a_f_dis.size())))
        print (np.shape (r_label), np.shape (a_f_dis))
        a_gen_loss = MSE(a_f_dis, r_label)
        b_gen_loss = MSE(b_f_dis, r_label)
		
        # identity loss
        b2b = Gb(b_real)
        a2a = Ga(a_real)
        idt_loss_b = L1(b2b, b_real)
        idt_loss_a = L1(a2a, a_real)
        idt_loss = idt_loss_a + idt_loss_b
        # rec losses
        a_rec_loss = L1(a_rec, a_real)
        b_rec_loss = L1(b_rec, b_real)
        rec_loss = a_rec_loss + b_rec_loss
        # g loss
        g_loss = a_gen_loss + b_gen_loss + rec_loss * 10.0 + 5.0 * idt_loss
        loss['G/a_gen_loss'] = a_gen_loss.item()
        loss['G/b_gen_loss'] = b_gen_loss.item()
        loss['G/rec_loss'] = rec_loss.item()
        loss['G/idt_loss'] = idt_loss.item()
        loss['G/g_loss'] = g_loss.item()
        # backward
        Ga.zero_grad()
        Gb.zero_grad()
        g_loss.backward()
        ga_optimizer.step()
        gb_optimizer.step()

        # leaves
        a_fake = Variable(torch.Tensor(a_fake_pool([a_fake.cpu().data.numpy()])[0]))
        b_fake = Variable(torch.Tensor(b_fake_pool([b_fake.cpu().data.numpy()])[0]))
        a_fake, b_fake = utils.cuda([a_fake, b_fake])

        # train D
        a_r_dis = Da(a_real)
        a_f_dis = Da(a_fake)
        b_r_dis = Db(b_real)
        b_f_dis = Db(b_fake)
        r_label = utils.cuda(Variable(torch.ones(a_f_dis.size())))
        f_label = utils.cuda(Variable(torch.zeros(a_f_dis.size())))

        # d loss
        a_d_r_loss = MSE(a_r_dis, r_label)
        a_d_f_loss = MSE(a_f_dis, f_label)
        b_d_r_loss = MSE(b_r_dis, r_label)
        b_d_f_loss = MSE(b_f_dis, f_label)

        a_d_loss = (a_d_r_loss + a_d_f_loss)*0.5
        b_d_loss = (b_d_r_loss + b_d_f_loss)*0.5
        loss['D/a_d_f_loss'] = a_d_f_loss.item()
        loss['D/b_d_f_loss'] = b_d_f_loss.item()
        loss['D/a_d_r_loss'] = a_d_r_loss.item()
        loss['D/b_d_r_loss'] = b_d_r_loss.item()
        # backward
        Da.zero_grad()
        Db.zero_grad()
        a_d_loss.backward()
        b_d_loss.backward()
        da_optimizer.step()
        db_optimizer.step()

        if (i + 1) % 10 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, i + 1, min(len(a_loader), len(b_loader))))
            print("g_loss: (%f)  a_d_loss: (%f)   b_d_loss: (%f)" % (g_loss, a_d_loss, b_d_loss ))
            if use_tensorboard:
                for tag, value in loss.items():
                     Logger.scalar_summary(tag, value, i) 
        if (i + 1) % 50 == 0:
            with torch.no_grad():
                Ga.eval()
                Gb.eval()
                a_real_test = Variable(iter(a_test_loader).next()[0])
                b_real_test = Variable(iter(b_test_loader).next()[0])
                a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
                # train G
                a_fake_test = Ga(b_real_test)
                b_fake_test = Gb(a_real_test)

                a_rec_test = Ga(b_fake_test)
                b_rec_test = Gb(a_fake_test)

                pic = (torch.cat([a_real_test, b_fake_test, a_rec_test, b_real_test, a_fake_test, b_rec_test], dim=0).data + 1) / 2.0
                
                save_dir = './sample_images_while_training/cyclegan'
                utils.mkdir(save_dir)
                torchvision.utils.save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, min(len(a_loader), len(b_loader))), nrow=3)

    utils.save_checkpoint({'epoch': epoch + 1,
                           'Da': Da.state_dict(),
                           'Db': Db.state_dict(),
                           'Ga': Ga.state_dict(),
                           'Gb': Gb.state_dict(),
                           'da_optimizer': da_optimizer.state_dict(),
                           'db_optimizer': db_optimizer.state_dict(),
                           'ga_optimizer': ga_optimizer.state_dict(),
                           'gb_optimizer': gb_optimizer.state_dict()},
                          '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1),
                          max_keep=4)
