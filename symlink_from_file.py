import os
import glob
import argparse


def symlink_from_file(inpath, outpath, fmt):
    imgs = glob.glob(os.path.join(inpath, f'*.{fmt}'))

    for img in imgs:
        os.symlink(img, os.path.join(outpath, os.path.basename(img)))


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Finetune a convnet')
    parser.add_argument('ptf', metavar='ptf', help='path to images to symlink')
    parser.add_argument('outptf', metavar='outptf', help='where to make symlink')
    parser.add_argument('--format', default='jpeg', metavar='format', choices=['jpg', 'jpeg', 'png', 'tiff'],
                        help='format of file to symlink')

    args = parser.parse_args()

    if os.path.exists(args.ptf) and os.path.exists(args.outptf):
        symlink_from_file(args.ptf, args.outptf, args.format)
    else:
        print('Check file paths')