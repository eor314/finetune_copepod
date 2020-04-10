# -*- coding: utf-8 -*-
"""
Deploy a trained model on data drawn from the remote server

Iteratively downloads a days worth of data, runs the selected network, outputs class lists of image paths,
and deletes the images before moving to the next day. With a flag will save mosaics of labeled images.
"""
# License: BSD
# Author(s): Eric Orenstein

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
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
from cv2 import imwrite
from tile_images import tile_images
from get_labeled_images import SPCQueryURL, get_json_data
from process_test_set import ImageFolderWithPaths, str2bool


if __name__ == '__main__':

    # define the parser
    parser = argparse.ArgumentParser(description='Make a train and val set from labels')

    parser.add_argument('parent_dir', metavar='parent_dir', help='path to directory with weights, train data')
    parser.add_argument('classifier', metavar='classifier', help='path to trained weights')
    parser.add_argument('--camera', metavar='camera', default='SPCP2',
                        choices=['SPC2', 'SPCP2'], help='which camera to consider')
    parser.add_argument('--time_step', metavar='time_step', default='week',
                        choices=['hour', 'day', 'week'], help='amount of time to increment')
    parser.add_argument('--num_step', metavar='num_step', default=10, help='number of time steps to take')
    parser.add_argument('--start_time', metavar='start_time', default=1502092800000, help='start time as unix timestamp in milliseconds')
    parser.add_argument('--server', metavar='server', default='planktivore',
                        choices=['planktivore', 'spc'], help='which server to retrieve images from')
    parser.add_argument('--save_mosaic', type=str2bool, default=False, help='name of image subdir within data_dir')
    parser.add_argument('--num_per_class', default=20, help='number to select for mosaic')
    parser.add_argument('--output_fold', metavar='output_fold', default=None, help='existing output folder')
    parser.add_argument('--buff', metavar='buff', default=0, help='number of pixels to add as buffer between classes in mosaic')

    args = parser.parse_args()

    parent_dir = args.parent_dir
    saved_model = os.path.join(parent_dir, args.classifier)
    cam = args.camera
    serv = args.server
    tstep = args.time_step
    stime = int(args.start_time)
    nstep = int(args.num_step)
    save_mosaic = args.save_mosaic
    num_per_class = args.num_per_class
    buff = int(args.buff)

    if args.output_fold:
        output_fold = os.path.join(parent_dir, 'outputs', args.output_fold)
    else:
        output_fold = None

    # get the increment in numbers
    inc_dict = {'hour': None, 'day': 1, 'week': 7}
    inc = inc_dict[tstep]
    
    # set the end time for first URL based on increment
    etime = stime + (inc*24*60*60-1)*1000

    # derive the validation directory from the classifier file path
    val_dir = os.path.join(os.path.split(saved_model)[0], 'val')

    # set the server information (these are set up to increment Mondays, hard coded sizes for phytos (.03 mm to 1 mm)
    if serv == 'planktivore':
        testset01_url = f"http://planktivore.ucsd.edu/data/rois/images/{cam}/{stime}/{etime}/0/24/500/40/1356/0.05/1/clipped/ordered/skip/Any/anytype/Any/Any/"
        im_loc = 'http://planktivore.ucsd.edu'
    else:
        testset01_url = f"http://spc.ucsd.edu/data/rois/images/{cam}/{stime}/{etime}/0/24/500/40/1356/0.05/1/clipped/ordered/skip/Any/anytype/Any/Any/"
        im_loc = 'http://spc.ucsd.edu'

    class_names = []
    for name in sorted(glob.glob(os.path.join(val_dir, '*'))):
        head, tail = os.path.split(name)
        class_names.append(tail)

    # make the output directory for the image lists
    if not output_fold:
        temp = os.path.join(parent_dir, 'outputs')

        if not os.path.exists(temp):
            os.mkdir(temp)

        data_parent = os.path.join(temp, os.path.basename(saved_model).split('.')[0] + '_' + str(int(time.time())))
        print("Saving outputs to: ", data_parent)

        if not os.path.exists(data_parent):
            os.mkdir(data_parent)
    else:
        data_parent = output_fold

    # make the temporary directory
    output_dir = os.path.join(parent_dir, 'run_net_temp')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ### set up the url query class ###
    query = SPCQueryURL()
    query.set_from_url(testset01_url)
    print(query.get_url())

    ### set stuff up for torch ###
    reproc = False

    im_per_dir = 9990

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(224),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, scale=(0.5, 2), shear=(-5, 5)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            #transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the saved model
    if 'resnet18' in os.path.basename(saved_model):
        model_conv = models.resnet18(pretrained=True)
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, len(class_names))
    elif 'resnet34' in os.path.basename(saved_model):
        model_conv = models.resnet34(pretrained=True)
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, len(class_names))

    model_conv.load_state_dict(torch.load(saved_model))
    model_conv = model_conv.to(device)
    model_conv.eval()
    optimizer = optim.SGD(model_conv.parameters(), lr=0.002, momentum=0.9)

    ### iterate over nstep time periods, download, and classify ###
    for ii in range(0, nstep):

        # check if the subdirectory in the data output exists. if not, proceed, else skip
        data_subdir = os.path.join(data_parent, query.query_params['start_utc'])

        if os.path.exists(data_subdir) and len(glob.glob(os.path.join(data_subdir, '*.txt'))) > 2:
            print('Looks like ' + query.query_params['start_utc'] + ' has already been processed')

            # increment a day
            query.increment_day(num_days=inc)

            continue

        if not os.path.exists(data_subdir):
            os.mkdir(data_subdir)

        # make a subdirectory in temp
        output_parent = os.path.join(output_dir, query.query_params['start_utc'])
        output_subdir = os.path.join(output_parent, 'images')
        next_page = query.get_url()  # set next page as first of new query

        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        # download the images
        print('downloading ' + os.path.basename(output_parent))

        while next_page:
            #print(next_page)
            data = get_json_data(next_page)
            data = data['image_data']
            next_page = data['next']  # update next page flag
            if next_page:
                next_page = im_loc + next_page[21::]  # remove ip address and replace with url

            for im in data['results']:
                try:
                #image_url = os.path.join(im_loc, im['image_url'] + '.jpg')
                    image_url = im_loc + im['image_url'] + '.jpg'
                    #print(image_url)
                    output_file = os.path.join(output_subdir,im['image_id'][:-4] + '.jpg')
                    #print(output_file)
                    urllib.request.urlretrieve(image_url, output_file)
                except:
                    print('error in downloading image')

        imgs = len(glob.glob(os.path.join(output_subdir, '*.jpg')))

        temp_out = {k: [] for k in class_names}

        # classify the images with the desired network
        print('Classifying ' + str(imgs) + ' in ' + os.path.basename(output_subdir))

        if imgs > 0:

            # set up the dataloader
            test_dataset = ImageFolderWithPaths(output_parent, data_transforms['val'])
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

                for jj, pp in enumerate(preds):
                    cls_counts[pp] += 1
                    temp_out[class_names[pp]].append(os.path.split(paths[jj])[1])

            # write the output to file
            for kk in temp_out.keys():
                print(kk + ' ' + str(len(temp_out[kk])))
                with open(os.path.join(data_subdir, kk + '.txt'), 'w') as ff:
                    for line in temp_out[kk]:
                        ff.write(line+'\n')
                    ff.close()

            # make the mosaics if needed (hard coded for 20 images per class
            if save_mosaic:
                flag = 0
                img_out = np.zeros((128 * 10, 128 * 2 * len(class_names), 3))
                img_out = img_out.astype(np.uint8)

                out_list = []
                for kk in temp_out.keys():
                    temp_list = temp_out[kk]
                    np.random.shuffle(temp_list)
                    temp_list = [os.path.join(output_subdir, line) for line in temp_list]
                    im_temp = temp_list[0:num_per_class]
                    temp_tile = tile_images(im_temp, [10, 2])
                    img_out[:, 128 * 2 * flag:128 * 2 * flag + 128 * 2, :] = temp_tile

                    if buff > 0:
                        img_out[:, ((128 * 2 * flag) + 128 * 2) -
                                int(buff / 2):((128 * 2 * flag) + 128 * 2) - int(buff / 2), :] = 255

                    flag += 1

                    out_list.extend(im_temp)

                out_mosaic = os.path.join(data_subdir, 'mosaic.png')
                imwrite(out_mosaic, img_out)
                with open(os.path.join(data_subdir, 'mosaic_imgs.txt'), 'w') as ff:
                    for line in out_list:
                        ff.write(line + '\n')
                    ff.close()

        # delete the images and start over
        print('Removing ' + os.path.basename(output_parent))
        rmtree(output_parent)

        # increment a day
        query.increment_day(num_days=inc)
