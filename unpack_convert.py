import tarfile
import cv2
import argparse
import os
import glob
import numpy as np


# import raw image
def import_image(ptf, raw=True, bayer_pattern=cv2.COLOR_BAYER_RG2RGB):

    # Load and convert image as needed
    img_c = cv2.imread(ptf, cv2.IMREAD_UNCHANGED)
    if raw:
        img_c = cv2.cvtColor(img_c, bayer_pattern)

    return img_c


# convert image to 8 bit with or without autoscaling
def convert_to_8bit(img, auto_scale=True):

    # Convert to 8 bit and autoscale
    if auto_scale:

        result = np.float32(img)-np.min(img)
        result[result < 0.0] = 0.0
        if np.max(img) != 0:
            result = result/np.max(img)

        img_8bit = np.uint8(255*result)
    else:
        img_8bit = np.unit8(img)

    return img_8bit


if __name__ == '__main__':

    # define parser
    parser = argparse.ArgumentParser(description='Unpack tarfiles and convert from tif')

    parser.add_argument('data_dir', metavar='data_dir', help='tarfile')
    parser.add_argument('out_dir', metavar='out_dir', help='path to outputs')

    args = parser.parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir

    tars = glob.glob(os.path.join(data_dir, '*.tar'))

    # extract all the tarfiles
    for ff in tars:
        ff_base = os.path.basename(ff).split('.')[0]
        if os.path.exists(os.path.join(out_dir, ff_base)):
            print('already extracted ', ff_base)
        else:
            print('extracting ', os.path.basename(ff))
            tar = tarfile.open(ff, 'r:')
            tar.extractall(out_dir)
            tar.close()

    # convert the images from tif
    out_ptf = os.path.join(os.path.split(out_dir)[0], 'cvt_imgs')
    img_dirs = glob.glob(os.path.join(out_dir, '*'))

    if not os.path.exists(out_ptf):
        os.mkdir(out_ptf)

    flag = 0
    num = len(img_dirs)
    for img_dir in img_dirs:
        img_list = glob.glob(os.path.join(img_dir, '*.tif'))
        for im in img_list:
            img_ptf = os.path.join(out_ptf, os.path.basename(im).split('.')[0] + '.jpg')
            if not os.path.exists(img_ptf):
                try:
                    tmp = import_image(im)
                    tmp = convert_to_8bit(tmp)

                    cv2.imwrite(img_ptf, tmp)
                    flag += 1
                except cv2.error:
                    print('issue with ', im)

        print('done with ', os.path.split(img_dir)[1])
