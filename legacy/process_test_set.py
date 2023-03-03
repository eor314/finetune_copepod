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
#import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
import glob
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


if __name__ == "__main__":

    # define parser
    parser = argparse.ArgumentParser(description='Process an annotated test set')

    parser.add_argument('data_dir', metavar='data_dir', help='path to unlabeled data')
    parser.add_argument('output_dir', metavar='output_dir', help='path to the output directory')
    parser.add_argument('classifier', metavar='classifier', help='path to trained weights')
    parser.add_argument('--save_mosaic', action='store_true', help='flag to save mosaic of output ROIs')
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

    test_dataset = ImageFolderWithPaths(data_dir, data_transforms['val'])

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                             shuffle=True, num_workers=4)

    # derive the output validation directory from the classifier name
    class_names = []
    for name in sorted(glob.glob(os.path.join(data_dir, '*'))):
        head, tail = os.path.split(name)
        class_names.append(tail)
    #class_names = test_dataset['val'].classes
    print(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the saved model
    model_type = os.path.basename(classifier)
    model_type = model_type.split('_')[0]

    if model_type == 'resnet34':
        model_conv = models.resnet34(pretrained=True)
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, len(class_names))
    if model_type == 'resnet18':
        model_conv = models.resnet18(pretrained=True)
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, len(class_names))
    if model_type == 'squeezenet':
        model_conv = models.squeezenet1_0(pretrained=True)
        # change the last Conv2D layer in case of squeezenet. there is no fc layer in the end.
        num_ftrs = 512
        model_conv.classifier._modules["1"] = nn.Conv2d(512, len(class_names), kernel_size=(1, 1))
        # because in forward pass, there is a view function call which depends on the final output class size.
        model_conv.num_classes = len(class_names)

    model_conv.load_state_dict(torch.load(classifier))

    model_conv = model_conv.to(device)

    model_conv.eval()

    optimizer = optim.SGD(model_conv.parameters(), lr=0.002, momentum=0.9)

    # save outputs as list for confusion matrix
    y_pred = []
    y_true = []

    for inputs, labels, paths in dataloader:
        inputs = inputs.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model_conv(inputs)
            _, preds = torch.max(outputs, 1)

        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

        """
        for i, p in enumerate(preds):

            dir_name = os.path.join(test_output, class_names[p])

            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            print(os.path.join(dir_name,paths[i]))

            # copy image to dir
            shutil.copy(paths[i],dir_name)
        """

    with open(os.path.join(output_path, f'preds_{int(time.time())}.txt'), 'w') as ff:
        for line in y_pred:
            ff.write(str(line) + '\n')

    with open(os.path.join(output_path, f'labs_{int(time.time())}.txt'), 'w') as ff:
        for line in y_true:
            ff.write(str(line) + '\n')