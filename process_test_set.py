# -*- coding: utf-8 -*-
"""
deply a trained_model

Deploys a trained model on a set of unlabeled images. Output is lists of image paths, seperated into the appropriate label.
With a flag will save mosaics of labeled images.

This is based heavily on The Transfer Learning Tutorial from pytorch:
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

"""
# License: BSD
# Author(s): Paul Roberts, Eric Orenstein

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
import argparse
from tile_images import tile_images, get_rand_ims


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def str2bool(v):
    """
    returns a boolean from argparse input
    """

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


if __name__ == "__main__":

    # define parser
    parser = argparse.ArgumentParser(description='Make a train and val set from labels')

    parser.add_argument('data_dir', metavar='data_dir', help='path to unlabeled data')
    parser.add_argument('output_dir', metavar='output_dir', help='path to the output directory')
    parser.add_argument('classifier', metavar='classifier', help='path to trained weights')
    parser.add_argument('--save_mosaic', type=str2bool, default=False, help='name of image subdir within data_dir')
    parser.add_argument('--num_per_class', default=20, help='number to select for mosaic')

    args = parser.parse_args()

    data_dir = args.data_dir
    output_path = args.output_dir
    classifier = args.classifier
    save_mosaic = args.save_mosaic
    num_per_class = int(args.num_per_class)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0,scale=(0.5, 2),shear=(-5, 5)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'D:\spcp2\\testset01'
    test_output = 'D:\spcp2\\testset01\\test_output'
    saved_model = 'D:\spcp2\dataset01\\resnet34_1536726964_model_conv.pt'

    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
    #                                           data_transforms[x])
    #                   for x in ['train', 'val']}
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
    #                                              shuffle=True, num_workers=4)
    #               for x in ['train', 'val']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # class_names = image_datasets['train'].classes

    test_dataset = ImageFolderWithPaths(data_dir, data_transforms['val'])

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                             shuffle=True, num_workers=4)

    class_names = ['Acantharea', 'Aggregate 01', 'Akashiwo', 'Avocado 01', 'Bubble', 'Ceratium furca', 'Ceratium fusus', 'Ciliate 01', 'Cochlodinium', 'Diamond 01', 'Hemiaulus', 'Lingulodinium', 'Nauplius', 'Polykrikos', 'Prorocentrum', 'Protoperidinium sp', 'Sand']

    print(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the saved model
    model_conv = models.resnet34(pretrained=True)
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, len(class_names))
    model_conv.load_state_dict(torch.load(saved_model))

    model_conv = model_conv.to(device)

    model_conv.eval()

    optimizer = optim.SGD(model_conv.parameters(), lr=0.002, momentum=0.9)

    for inputs, labels, paths in dataloader:
        inputs = inputs.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model_conv(inputs)
            _, preds = torch.max(outputs, 1)

        for i, p in enumerate(preds):

            dir_name = os.path.join(test_output, class_names[p])

            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            print(os.path.join(dir_name,paths[i]))

            # copy image to dir
            shutil.copy(paths[i],dir_name)
