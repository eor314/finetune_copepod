# -*- coding: utf-8 -*-
"""
Deploy a trained_model

Deploys a trained model on a set of unlabeled images stored locally. Output is lists of image paths, seperated into the appropriate label.
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
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import glob
import sys
import urllib
import time
from shutil import copy, rmtree
import argparse
from tile_images import tile_images, get_rand_ims
from cv2 import imwrite


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


if __name__ == '__main__':

    # define parser
    parser = argparse.ArgumentParser(description='Make a train and val set from labels')

    parser.add_argument('data_dir', metavar='data_dir', help='path to unlabeled data')
    parser.add_argument('classifier', metavar='classifier', help='path to trained weights')
    parser.add_argument('--save_mosaic', metavar='save_mosaic', type=str2bool,
                        default=False, help='name of image subdir within data_dir')
    parser.add_argument('--num_per_class', metavar='num_per_class', default=20, help='number to select for mosaic')
    parser.add_argument('--buff', metavar='buff', default=0, help='number of pixels to add as buffer between classes in mosaic')

    args = parser.parse_args()
    data_dir = args.data_dir
    classifier = args.classifier
    save_mosaic = args.save_mosaic
    num_per_class = int(args.num_per_class)
    buff = int(args.buff)

    print(num_per_class)

    # derive the output validation directory from the classifier name
    data_parent = os.path.split(classifier)[0]
    output_parent = os.path.join(data_parent, 'outputs', os.path.basename(classifier).split('.')[0] +
                                 '_' + str(int(time.time())))

    val_dir = os.path.join(data_parent, 'val')
    print(output_parent)
    os.mkdir(output_parent)

    class_names = []
    for name in sorted(glob.glob(os.path.join(val_dir, '*'))):
        head, tail = os.path.split(name)
        class_names.append(tail)

    ### set stuff up for torch ###
    reproc = False

    im_per_dir = 9990

    # Data augmentation and normalization for training
    # Just normalization for    validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, scale=(0.5, 2), shear=(-5, 5)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

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

    ### iterate over directories ###
    dirs_to_process = glob.glob(os.path.join(data_dir, '*'))
    dirs_to_process.sort()

    # make the temp directory for symlinking
    temp_parent = os.path.join(output_parent, 'temp')

    if not os.path.exists(temp_parent):
        os.mkdir(temp_parent)

    for dir_in in dirs_to_process:

        imgs_in = glob.glob(os.path.join(dir_in, '*'))

        temp_out = {k: [] for k in class_names}

        # classify the images with the desired network
        print('Classifying ' + str(len(imgs_in)) + ' in ' + os.path.basename(dir_in))

        if len(imgs_in) > 0:

            #make the temporary directory for the symlink
            temp_dir = os.path.join(temp_parent, 'temp')
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)

            # symlink the images into the directory
            for img in imgs_in:
                os.symlink(img, os.path.join(temp_dir, os.path.basename(img)))

            # make the output directory
            out_subdir = os.path.join(output_parent, os.path.basename(dir_in))
            os.mkdir(out_subdir)

            # set up the dataloader
            test_dataset = ImageFolderWithPaths(temp_parent, data_transforms['val'])
            dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128,
                                                     shuffle=True, num_workers=4)

            # run the classifier
            cls_counts = np.zeros(len(class_names))

            for inputs, labels, paths in dataloader:
                inputs = inputs.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(False):
                    outputs = model_conv(inputs)
                    _, preds = torch.max(outputs, 1)

                for ii, pp in enumerate(preds):
                    cls_counts[pp] += 1
                    temp_out[class_names[pp]].append(os.path.split(paths[ii])[1])

            # write the output to file
            for kk in temp_out.keys():
                print(kk + ' ' + str(len(temp_out[kk])))
                with open(os.path.join(out_subdir, kk +'.txt'), 'w') as ff:
                    for line in temp_out[kk]:
                        ff.write(line + '\n')
                    ff.close()

            # make the mosaics if needed
            if save_mosaic:
                flag = 0
                img_out = np.zeros((128*10, 128*2*len(class_names), 3))
                img_out = img_out.astype(np.uint8)

                out_list = []
                for kk in temp_out.keys():
                    print(kk)
                    temp_list = temp_out[kk]
                    np.random.shuffle(temp_list)
                    temp_list = [os.path.join(dir_in, line) for line in temp_list]
                    im_temp = temp_list[0:num_per_class]
                    temp_tile = tile_images(im_temp, [10, 2])
                    img_out[:, 128 * 2 * flag:128 * 2 * flag + 128 * 2, :] = temp_tile

                    if buff > 0:
                        img_out[:, ((128*2*flag)+128*2)-int(buff/2):((128*2*flag)+128*2)-int(buff/2), :] = 255

                    flag += 1

                    out_list.extend(im_temp)

                out_mosaic = os.path.join(out_subdir, 'mosaic.png')
                imwrite(out_mosaic, img_out)
                with open(os.path.join(out_subdir, 'mosaic_imgs.txt'), 'w') as ff:
                    for line in out_list:
                        ff.write(line + '\n')
                    ff.close()

            # remove the temporary directory
            rmtree(temp_dir)
