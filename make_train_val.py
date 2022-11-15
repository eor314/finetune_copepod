import os
import sys
import glob
import random
import shutil
import numpy as np
import argparse
from PIL import Image
from datetime import datetime


def list_image_counts(data_path):

    img_dirs = glob.glob(os.path.join(data_path, '*'))

    for d in img_dirs:

        imgs = glob.glob(os.path.join(d,'*'))
        print(str(len(imgs)).zfill(5) + ' --- ' + os.path.basename(d))


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Make a train and val set from labels')

    parser.add_argument('data_dir', metavar='data_dir', help='path to labelset')
    parser.add_argument('output_dir', metavar='output_dir', help='path to the output train/val set')
    parser.add_argument('--train_name', default='train', help='name of training dir to create')
    parser.add_argument('--val_name', default='val', help='name of validation dir to create')
    parser.add_argument('--train_pct', default=0, help='fraction of training vs validation from total')
    parser.add_argument('--train_size', default=800, help='Number of images in train set per class')
    parser.add_argument('--val_size', default=200, help='Number of images in val set per class')
    parser.add_argument('--min_images', default=100, help='Smallest number of images per class to use')
    parser.add_argument('--duplicate', action='store_false', help='When True, images are duplicated as needed')
    parser.add_argument('--symlink', action='store_true', default=True, help='When True, symlink images instead of copying to new dir')

    args = parser.parse_args()

    img_path = args.data_dir
    dataset_parent = args.output_dir
    img_subdir = args.image_subdir
    train_name = args.train_name
    val_name = args.val_name
    train_pct = float(args.train_pct)
    train_size = int(args.train_size)
    val_size = int(args.val_size)
    min_imgs = int(args.min_images)
    duplicate = args.duplicate
    test_ims = args.test_ims
    symflag = args.symlink

    dataset_path = os.path.join(dataset_parent, datetime.utcnow().isoformat()[:-7].replace(':','-'))

    if not os.path.exists(dataset_parent):
        os.mkdir(dataset_parent)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        os.makedirs(os.path.join(dataset_path,'train'))
        os.makedirs(os.path.join(dataset_path,'val'))

    img_dirs = sorted(glob.glob(os.path.join(img_path,'*')))

    for img_dir in img_dirs:

        print(img_dir)

        imgs = sorted(glob.glob(os.path.join(img_dir,'*')))

        if train_pct != 0:

            all_inds = random.sample(range(0,len(imgs)),len(imgs))
            train_inds = all_inds[0:int(train_pct*len(imgs))]
            val_inds = all_inds[int(train_pct*len(imgs)):]
        else:

            if (len(imgs) < min_imgs):
                print('not enough images, skipping class: ' + os.path.basename(img_dir))
                continue

            # make inds
            # check if the number of images in the file is greater than the total needed for training
            if duplicate and len(imgs) < (train_size + val_size):
                # select with replacement if needed
                if (train_size+val_size)-len(imgs) < len(imgs):
                    # if the difference is smaller than the original list, just randomly select the number needed
                    fill = np.random.choice(len(imgs), (train_size+val_size)-len(imgs), replace=False)
                else:
                    # otherwise select with replacement
                    fill = np.random.choice(len(imgs), (train_size + val_size) - len(imgs), replace=True)

                all_inds = np.block([np.arange(len(imgs)), fill])  # stack the filler on top to make appropriate dimension
                np.random.shuffle(all_inds)  # shuffle
                #all_inds = np.random.choice(len(imgs), train_size+val_size, replace=duplicate)
                train_inds = all_inds[0:train_size]
                print('Number of unique train_inds: ' + str(len(set(train_inds))))
                val_inds = all_inds[train_size:]
            else:
                # otherwise select without replacement
                all_inds = np.random.choice(len(imgs), train_size+val_size, replace=False)
                train_inds = all_inds[0:train_size]
                print('Number of unique train_inds: ' + str(len(set(train_inds))))
                val_inds = all_inds[train_size:]

        new_img_dir = os.path.join(dataset_path,'train',os.path.basename(img_dir))

        if not os.path.exists(new_img_dir):
            os.makedirs(new_img_dir)

        for i, ind in enumerate(train_inds):
            img_dest = os.path.join(new_img_dir, os.path.basename(imgs[ind])[:-4]+'_index_'+str(i).zfill(4)+'.jpg')

            if os.path.splitext(img_dest)[1] != os.path.splitext(imgs[ind])[1]:
                img_dest = os.path.join(new_img_dir,
                                        os.path.basename(imgs[ind])[:-4] + '_index_' +
                                        str(i).zfill(4) + os.path.splitext(imgs[ind])[1])

            if symflag:
                os.symlink(imgs[ind], img_dest)
            else:
                shutil.copy(imgs[ind], img_dest)

        new_img_dir = os.path.join(dataset_path, 'val', os.path.basename(img_dir))

        if not os.path.exists(new_img_dir):
            os.makedirs(new_img_dir)

        for i, ind in enumerate(val_inds):

            img_dest = os.path.join(new_img_dir, os.path.basename(imgs[ind])[:-4]+'_index_'+str(i).zfill(4)+'.jpg')

            if os.path.splitext(img_dest)[1] != os.path.splitext(imgs[ind])[1]:
                img_dest = os.path.join(new_img_dir,
                                        os.path.basename(imgs[ind])[:-4] + '_index_' +
                                        str(i).zfill(4) + os.path.splitext(imgs[ind])[1])

            if symflag:
                os.symlink(imgs[ind], img_dest)
            else:
                shutil.copy(imgs[ind], img_dest)
