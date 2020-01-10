# -*- coding: utf-8 -*-
"""
Created on Wed Sept 19 2018

@author: eric
"""

import os
import glob
import numpy as np
import sys
import cv2
import argparse


def aspect_resize(im, ii=226):
    """
    Resizes an image and preserves the aspect according to largest dimension
    :param im: input array
    :param ii: desired dimension of the output. Assumes square output image
    :return out: resized square image array
    """

    cen = np.floor(np.array((ii, ii))/2.0).astype('int')
    dim = im.shape[0:2]

    if dim[0] != dim[1]:
        # get the largest dimension
        large_dim = max(dim)

        # ratio between the large dimension and required dimension
        rat = float(ii)/large_dim

        # get the smaller dimension that maintains the aspect ratio
        small_dim = int(min(dim)*rat)

        # get the indicies of the large and small dimensions
        large_ind = dim.index(max(dim))
        small_ind = dim.index(min(dim))
        dim = list(dim)

        # the dimension assigment may seem weird cause of how python indexes images
        dim[small_ind] = ii
        dim[large_ind] = small_dim
        dim = tuple(dim)

        im = cv2.resize(im, dim)
        half = np.floor(np.array(im.shape[0:2])/2.0).astype('int')

        # make an empty array, and place the new image in the middle
        res = np.zeros((ii, ii, 3), dtype='uint8')

        if large_ind == 1:
            test = res[cen[0]-half[0]:cen[0]+half[0], cen[1]-half[1]:cen[1]+half[1]+1]
            if test.shape != im.shape:
                res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]+1] = im
            else:
                res[cen[0]-half[0]:cen[0]+half[0], cen[1]-half[1]:cen[1]+half[1]+1] = im
        else:
            test = res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]]
            if test.shape != im.shape:
                res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]+1] = im
            else:
                res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]] = im
    else:
        res = cv2.resize(im, (ii, ii))

    return res

def get_rand_ims(cls, num=200):
    """
    takes a list of classes and randomly selects images from random classes on
    the day specified
    :param cls: input list of files pointing to classes to consider
    :param num: number of images to select
    """

    # placeholder and output list
    out = []
    count = 0

    # select random images until the number has been reached
    while count < num:
        ind = np.random.randint(0, len(cls))  # select random index
        in_cls = cls[ind]
        imgs = glob.glob(os.path.join(in_cls, '*.jpg'))
        np.random.shuffle(imgs)
        check = True  # set bool to check if the image has been selected
        flag = 0
        while check:
            try:
                img = imgs[flag]
                if img in out:
                    flag += 1
                else:
                    out.append(img)
                    check = False
                    count += 1
            except IndexError:
                print('sampled all ' + str(flag) + ' of ' + os.path.basename(in_cls))
                cls.remove(in_cls)
                check = False

    return out

def tile_images(images, tile_dim, resize=128):
    """
    takes a list of images and tiles them
    :param images: input list of image paths
    :param tile_dim: number to tile in each dimension [hh x ww] as int
    :param resize: size to resize the input images
    :return:
    """

    out = np.zeros((resize*tile_dim[0], resize*tile_dim[1], 3))
    out = out.astype(np.uint8)

    for idx, img in enumerate(images):
        ii = idx % tile_dim[1]
        jj = idx // tile_dim[1]

        im_in = cv2.imread(img)
        im_out = aspect_resize(im_in, resize)

        out[jj*resize:jj*resize+resize, ii*resize:ii*resize+resize, :] = im_out

    return out


if __name__ == "__main__":

    # where you run from /home/ptvradmin/machine_learning
    cwd = os.getcwd()

    # deterimine how to run and switch if necessary
    if os.path.exists(sys.argv[2]):
        # check if the second argument is a file
        # what classifier output to work on
        clf = sys.argv[1]

        # get the days to work on from file
        days = np.genfromtxt(sys.argv[2], dtype=int)

        # parse the class from the file name with list of days
        lab = os.path.basename(sys.argv[2]).split('_')[0]

        # where to save the finished mosaics
        out_path = sys.argv[3]

        # number of images per days
        num_samples = int(sys.argv[4])

        # iterate over the days and append paths to list
        out_imgs = []
        np.random.shuffle(days)

        # keep track of the total and generate a list of the days used
        date_info = []
        for day in days:
            # get the file paths
            ptf = os.path.join(cwd,
                clf,
                'Any',
                str(day),
                'labeled_images',
                lab
            )

            imgs = glob.glob(os.path.join(ptf, '*.jpg'))

            # shuffle and select the images
            np.random.shuffle(imgs)
            if len(imgs) < num_samples:
                out_imgs.extend(imgs)
            else:
                out_imgs.extend(imgs[0:num_samples])

            # retain info about the number of images
            date_info.append([day, len(imgs)])

            # check if there are more than the deisred number
            if len(out_imgs) > 200:
                break

        # generate the mosaic
        print(len(out_imgs))
        np.random.shuffle(out_imgs)
        mos = tile_images(out_imgs[0:200], [num_samples, 20])

        # save the image and information about the number of ROIs/day
        out_mos = os.path.join(out_path, lab+'_hyb_samp_mosaic.png')
        out_info = os.path.join(out_path,lab+'_num_hyb_per_day.txt')

        cv2.imwrite(out_mos, mos)
        date_info = np.asarray(date_info)
        np.savetxt(out_info, date_info, fmt='%i', delimiter=',')

    elif sys.argv[2] == 'except':
        # what classifier output to work on
        clf = sys.argv[1]

        # what day to work on
        day = sys.argv[3]

        # category to ignore
        lab = sys.argv[4]

        # where to save the finished mosaics
        out_path = sys.argv[5]

        # path to the file
        ptf = os.path.join(cwd,
            clf,
            'Any',
            day,
            'labeled_images'
        )

        # all classes
        process = glob.glob(os.path.join(ptf, '*'))

        # remove the identified class
        consider = [line for line in process if os.path.basename(line) != lab]

        # make the mosaic
        imgs = get_rand_ims(consider)
        mos = tile_images(imgs, [10, 20])

        out_str = os.path.join(out_path,
                                day + '_except_' + lab + '_mosaic.png')

        cv2.imwrite(out_str, mos)

    else:
        # what classifier output to work on
        clf = sys.argv[1]

        # what day to work on
        day = sys.argv[2]

        # category to work on ['phyto' produces a mosaic of phyto class,
        # 'noise' produces a mosaic of noise. A particular class makes a mosaic of
        # just that class]
        lab = sys.argv[3]

        # where to save the finished mosaics
        out_path = sys.argv[4]

        if lab != 'phyto' and lab != 'noise':
            # get the path to the file
            ptf = os.path.join(cwd,
                clf,
                'Any',
                day,
                'labeled_images',
                lab
            )

            imgs = glob.glob(os.path.join(ptf, '*.jpg'))

            # randomly shuffle and select the number of samples
            num_samples = 200

            np.random.shuffle(imgs)
            imgs = imgs[0:num_samples]

            mos = tile_images(imgs, [10, 20])

            out_str = os.path.join(out_path,
                                    day + '_' + lab + '_mosaic.png')

            cv2.imwrite(out_str, mos)

        else:
            # make the file path to the images
            ptf = os.path.join(cwd,
                clf,
                'Any',
                day,
                'labeled_images'
            )

            process = glob.glob(os.path.join(ptf, '*'))

            # phyto and noise classes as of 092018
            consid = ['Akashiwo', 'Ceratium furca', 'Ceratium fusus', 'Chain 01', 'Ciliate 01', 'Cochlodinium',
                      'Lingulodinium', 'Nauplius', 'Polykrikos', 'Prorocentrum', 'Prorocentrum Skinny', 'Protoperidinium sp',
                      'Spear 01']

            if lab == 'phyto':
                mos_lab = [line for line in process if os.path.basename(line) in consid]
            else:
                mos_lab = [line for line in process if os.path.basename(line) not in consid]

            # number of samples to draw from each class
            num_samples = 20
            img_out = np.zeros((128*10, 128*2*len(mos_lab), 3))
            img_out = img_out.astype(np.uint8)
            flag = 0

            for proc in mos_lab:

                imgs = glob.glob(os.path.join(proc, '*.jpg'))
                np.random.shuffle(imgs)
                imgs = imgs[0:num_samples]

                temp = tile_images(imgs, [10, 2])
                img_out[:, 128*2*flag:128*2*flag+128*2, :] = temp

                flag += 1

            out_str = os.path.join(out_path,
                    day + '_' + lab + '_mosaic.png')

            cv2.imwrite(out_str, img_out)
