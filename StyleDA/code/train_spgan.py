import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1, 0"

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import itertools
import models.models_spgan as models
import torch
from torch import nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils.utils
from PIL import Image
from utils.logger import Logger
import numpy as np
#from torch.optim import lr_scheduler

# gpu_id = [0, 1]
# utils.utils.cuda_devices(gpu_id)
""" gpu """
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" param """
epochs = 20
batch_size = 4
lr = 0.0002
dataset_dir = '../datasets/sys2real'
lambda1 = 10.0
lambda2 = 5.0
lambda3 = 2.0
margin = 2.0
use_tensorboard = True
if use_tensorboard:
    log_dir = './checkpoints/sys2real'
    utils.utils.mkdir(log_dir)
    Logger = Logger(log_dir)
	
""" data """
load_size_w = 512
load_size_h = 512
crop_size_w = 256
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

dataset_dirs = utils.utils.reorganize(dataset_dir)
a_data = dsets.ImageFolder(dataset_dirs['trainA'], transform=transform)
b_data = dsets.ImageFolder(dataset_dirs['trainB'], transform=transform)
a_test_data = dsets.ImageFolder(dataset_dirs['testA'], transform=test_transform)
b_test_data = dsets.ImageFolder(dataset_dirs['testB'], transform=test_transform)
a_loader = torch.utils.data.DataLoader(a_data, batch_size=batch_size, shuffle=True, num_workers=0)
b_loader = torch.utils.data.DataLoader(b_data, batch_size=batch_size, shuffle=True, num_workers=0)
a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=1, shuffle=True, num_workers=0)
b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=1, shuffle=True, num_workers=0)

a_fake_pool = utils.utils.ItemPool()
b_fake_pool = utils.utils.ItemPool()

# model = nn.DataParallel(model).cuda()
""" model """
Da = nn.DataParallel(models.Discriminator()).to(device)
Db = nn.DataParallel(models.Discriminator()).to(device)
Ga = nn.DataParallel(models.Generator()).to(device)
Gb = nn.DataParallel(models.Generator()).to(device)
Me = nn.DataParallel(models.Metric_Net()).to(device)
MSE = nn.MSELoss()
L1 = nn.L1Loss()

da_optimizer = torch.optim.Adam(Da.parameters(), lr=lr, betas=(0.5, 0.999))
db_optimizer = torch.optim.Adam(Db.parameters(), lr=lr, betas=(0.5, 0.999))
ga_optimizer = torch.optim.Adam(Ga.parameters(), lr=lr, betas=(0.5, 0.999))
gb_optimizer = torch.optim.Adam(Gb.parameters(), lr=lr, betas=(0.5, 0.999))
gb_optimizer = torch.optim.Adam(Gb.parameters(), lr=lr, betas=(0.5, 0.999))
me_optimizer = torch.optim.Adam(Me.parameters(), lr=lr, betas=(0.5, 0.999))

# SGD for me network
#me_optimizer = torch.optim.SGD(Me.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
# Decay LR by a factor of 0.1 every 2 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(me_optimizer, step_size=3, gamma=0.1)

""" load checkpoint """
ckpt_dir = './checkpoints/sys2real/'
utils.utils.mkdir(ckpt_dir)
try:
    ckpt = torch.load(ckpt_dir)
    start_epoch = ckpt['epoch']
    Da.load_state_dict(ckpt['Da'])
    Db.load_state_dict(ckpt['Db'])
    Ga.load_state_dict(ckpt['Ga'])
    Gb.load_state_dict(ckpt['Gb'])
    Me.load_state_dict(ckpt['Me'])
    da_optimizer.load_state_dict(ckpt['da_optimizer'])
    db_optimizer.load_state_dict(ckpt['db_optimizer'])
    ga_optimizer.load_state_dict(ckpt['ga_optimizer'])
    gb_optimizer.load_state_dict(ckpt['gb_optimizer'])
    me_optimizer.load_state_dict(ckpt['me_optimizer'])
except:
    print(' [*] No checkpoint!')
    start_epoch = 0
# assert (0)
""" run """
for epoch in range(start_epoch, epochs):
    for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
        # step
        # if i % 100 == 0:
        #     # print (i)
        #     print (i, len (a_real), len(b_real))
        # if i < 9401:
        #     continue
        
        if len(a_real) != len(b_real):
            print ("heheda")
            # assert (0)
            continue
        # continue
        step = epoch * min(len(a_loader), len(b_loader)) + i + 1

        # set train
        Ga.train()
        Gb.train()

        # leaves
        a_real = a_real[0].to(device)
        b_real = b_real[0].to(device)
        a_fake = Ga(b_real)
        b_fake = Gb(a_real)
        a_rec = Ga(b_fake)
        b_rec = Gb(a_fake)
        loss = {}
        # =================================================================================== #
        #                               1. Train the Generator (Ga and Gb)                    #
        # =================================================================================== #
        if np.shape(a_real)[0] != np.shape(b_real)[0]:
                # print ("heheda")
            # assert (0)
                continue
        if i % 2 ==0:
            #siamese network	
        
            a_metric = Me(a_real)
            a2b_metric = Me(b_fake)
            b_metric = Me(b_real)
            b2a_metric = Me(a_fake)
    		
            sia_criterion = models.ContrastiveLoss() 	

            # print (np.shape (a_real), np.shape (b_real))
            if np.shape(a_real)[0] != np.shape(b_real)[0]:
                # print ("heheda")
            # assert (0)
                continue
            # print (np.shape (b_real))
            # print (np.shape (a_metric))
            # print (np.shape (a2b_metric))	
            #Positive Pair
            loss_pos0 = sia_criterion(a_metric, a2b_metric, 0)
            loss_pos1 = sia_criterion(b_metric, b2a_metric, 0)
            #Negative_Pair
            loss_neg0 = sia_criterion(a2b_metric, b_metric, 1)
            loss_neg1 = sia_criterion(b2a_metric, a_metric, 1)
            loss_neg= sia_criterion(a_metric, b_metric, 1)
            #contrastive loss for G
            m_loss_G = (loss_pos0 + loss_pos1 + 0.5*(loss_neg0 + loss_neg1)) / 4.0
            m_loss_M = (loss_pos0 + loss_pos1 + 2* loss_neg) / 3.0
            # identity loss
            b2b = Gb(b_real)
            a2a = Ga(a_real)
            idt_loss_b = L1(b2b, b_real)
            idt_loss_a = L1(a2a, a_real)
            idt_loss = idt_loss_a + idt_loss_b
    
            # gen losses
            a_f_dis = Da(a_fake)
            b_f_dis = Db(b_fake)
            r_label = torch.ones(a_f_dis.size()).to(device)
            a_gen_loss = MSE(a_f_dis, r_label)
            b_gen_loss = MSE(b_f_dis, r_label)
    
            # rec losses
            a_rec_loss = L1(a_rec, a_real)
            b_rec_loss = L1(b_rec, b_real)
            rec_loss = a_rec_loss + b_rec_loss
            # g loss
            if epoch >1:			
                g_loss = a_gen_loss + b_gen_loss + lambda1* rec_loss + lambda2 * idt_loss + lambda3 * m_loss_G	
            else:
                g_loss = a_gen_loss + b_gen_loss + lambda1* rec_loss + lambda2 * idt_loss
            loss['G/a_gen_loss'] = a_gen_loss.item()
            loss['G/b_gen_loss'] = b_gen_loss.item()
            loss['G/rec_loss'] = rec_loss.item()
            loss['G/idt_loss'] = idt_loss.item()
            loss['G/m_loss_G'] = m_loss_G.item()
            
            Ga.zero_grad()
            Gb.zero_grad()   
            g_loss.backward()
            ga_optimizer.step()
            gb_optimizer.step()
        # =================================================================================== #
        #                               2. Train the Siamese Network                          #
        # =================================================================================== #
        if epoch >0:
            a_metric = Me(a_real)
            a2b_metric = Me(b_fake.detach())
            b_metric = Me(b_real)
            b2a_metric = Me(a_fake.detach())		
            #Positive Pair
            loss_pos0 = sia_criterion(a_metric, a2b_metric, 0)
            loss_pos1 = sia_criterion(b_metric, b2a_metric, 0)
            #print(loss_pos1, type(loss_pos1))
            #Negative_Pair
            loss_neg= sia_criterion(a_metric, b_metric, 1)
            #contrastive loss for Sia
            m_loss_M = (loss_pos0 + loss_pos1 + 2*loss_neg) / 3.0
            
            loss['M/loss_pos0'] = loss_pos0.item()
            loss['M/loss_pos1'] = loss_pos1.item()
            loss['M/loss_neg'] = loss_neg.item()
            loss['M/m_loss_M'] = m_loss_M.item()
            Me.zero_grad()
            m_loss_M.backward()
            me_optimizer.step()
        # =================================================================================== #
        #                               3. Train the Discriminator                            #
        # =================================================================================== #        
        # leaves
        a_fake = torch.Tensor(a_fake_pool([a_fake.cpu().data.numpy()])[0]).to(device)
        b_fake = torch.Tensor(b_fake_pool([b_fake.cpu().data.numpy()])[0]).to(device)

        if np.shape(a_fake)[0] != np.shape(b_fake)[0]:
                # print ("heheda")
            # assert (0)
                continue
        
        ## Training D
        a_r_dis = Da(a_real)
        a_f_dis = Da(a_fake)
        b_r_dis = Db(b_real)
        b_f_dis = Db(b_fake)
        r_label = torch.ones(a_f_dis.size()).to(device)
        f_label = torch.zeros(a_f_dis.size()).to(device)

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
        #==================================================================================== #
        #                                 4. Miscellaneous                                    #
        # =================================================================================== #
        if i % 100 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, i + 1, min(len(a_loader), len(b_loader))))
            print("sia_loss: (%f) mloss_G: (%f)  pos: (%f) g_loss: (%f)  a_d_loss: (%f)   b_d_loss: (%f) " % (m_loss_M.item(), m_loss_G.item(), loss_pos0.item() + loss_pos1.item(), g_loss.item(), a_d_loss.item(), b_d_loss.item()))
            if use_tensorboard:
                for tag, value in loss.items():
                     Logger.scalar_summary(tag, value, i) 
        if i % 100 == 0:
            with torch.no_grad():
                Ga.eval()
                Gb.eval()  
                a_real_test =(iter(a_test_loader).next()[0]).to(device)
                b_real_test = (iter(b_test_loader).next()[0]).to(device)
                #print(a_real_test.size(), b_real_test.size())
                # train G
                a_fake_test = Ga(b_real_test)
                b_fake_test = Gb(a_real_test)
    
                a_rec_test = Ga(b_fake_test)
                b_rec_test = Gb(a_fake_test)
    
                pic = (torch.cat([a_real_test, b_fake_test, a_rec_test, b_real_test, a_fake_test, b_rec_test], dim=0).data + 1) / 2.0
    
                save_dir = './sample_images_while_training_sys2real' #
                utils.utils.mkdir(save_dir)
                torchvision.utils.save_image(pic, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, min(len(a_loader), len(b_loader))), nrow=3)

    utils.utils.save_checkpoint({'epoch': epoch + 1,
                           'Da': Da.state_dict(),
                           'Db': Db.state_dict(),
                           'Ga': Ga.state_dict(),
                           'Gb': Gb.state_dict(),
                           'Me': Me.state_dict(),
                           'da_optimizer': da_optimizer.state_dict(),
                           'db_optimizer': db_optimizer.state_dict(),
                           'ga_optimizer': ga_optimizer.state_dict(),
                           'gb_optimizer': gb_optimizer.state_dict(),
                           'me_optimizer': me_optimizer.state_dict()},
                          '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1),
                          max_keep=20)
