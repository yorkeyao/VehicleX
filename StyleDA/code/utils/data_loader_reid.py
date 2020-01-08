from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random



def get_loader(image_dir, crop_size=256, image_size=286, 
               batch_size=16, dataset='Market', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
        transform.append(T.Resize([288, 144], interpolation=3))
        transform.append(T.RandomCrop([256, 128]))
    else:
        #transform.append(T.RandomHorizontalFlip())
        transform.append(T.Resize([256, 128], interpolation=3))
        #transform.append(T.RandomCrop([256, 128]))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'Market':
        #print(image_dir)
        dataset = ImageFolder(image_dir, transform)
        print(len(dataset.classes))
    elif dataset == 'Duke':
        #print(image_dir)
        dataset = ImageFolder(image_dir, transform)
        print(len(dataset.classes))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader