import numpy as np
import cv2
import skimage
from skimage import filters, morphology, measure, color, feature
from scipy import ndimage, interpolate
import sys
import glob
import os
import progressbar
import pandas as pd
import matplotlib.pyplot as plt


# extract major axis length
def extract_features(img):

    gray = np.uint8(np.mean(img, 2))

    # unpack settings
    low_threshold = 2.0
    blur_rad = 3

    # edge-based segmentation and region filling to define the object
    edges_mag = filters.scharr(gray)
    edges_med = np.median(edges_mag)
    edges_thresh = low_threshold * edges_med
    edges = edges_mag >= edges_thresh
    edges = morphology.closing(edges, morphology.disk(blur_rad))
    filled_edges = ndimage.binary_fill_holes(edges)
    edges = morphology.erosion(filled_edges, morphology.disk(blur_rad))

    # define the binary image for further operations
    bw_img = edges

    # Compute morphological descriptors
    label_img = morphology.label(bw_img, connectivity=2, background=0)
    props = measure.regionprops(label_img, gray)

    if len(props) > 0:
        # use only the features from the object with the largest area
        max_area = 0
        max_area_ind = 0
        for f in range(0, len(props)):
            if props[f].area > max_area:
                max_area = props[f].area
                max_area_ind = f

        ii = max_area_ind

    else:
        ii = 0

    return props[ii].major_axis_length


## for training data path to images
ptf = r'D:\project_data\eilat\copepods'

res = 7.69  # um/px

imgs = glob.glob(os.path.join(ptf, '*.jpeg'))

out = []
probs = []
for im in progressbar.progressbar(imgs):
    try:
        out.append(extract_features(cv2.imread(im)))  # returns major axis in pixels
    except IndexError:
        print(f'problem with {os.path.basename(im)}')
        probs.append(im)
        out.append(0.0)

out = np.asarray(out)
maj = pd.DataFrame(zip(out, out*res), columns=['pixels', 'microns'],
                   index=[os.path.basename(line) for line in imgs])

fig, ax = plt.subplots()
maj['microns'].hist(ax=ax, bins=50)
ax.set_ylabel('Counts')
ax.set_xlabel('Major axis ($\mu$m)')
ax.set_title('ROI size histogram')

filt = maj[maj['microns'] > 290].index.to_list()

# for test data
alldirs = glob.glob(os.path.join(r'D:\project_data\eilat\test-data', '*'))
outdir = r'D:\project_data\eilat\test-data\greater290'
probs = []
res = 7.69  # um/px

for imgdir in alldirs:
    imgs = glob.glob(os.path.join(imgdir, '*_rawcolor.jpeg'))
    outpath = os.path.join(outdir, os.path.basename(imgdir))
    if not os.path.exists(outpath):
        os.mkdir(outpath)
        flag = 0
    for im in progressbar.progressbar(imgs):
        try:
            maj = extract_features(cv2.imread(im))
            maj = maj*res
            if maj > 290:
                os.symlink(im, os.path.join(outpath, os.path.basename(im)))
                flag += 1
        except IndexError:
            print(f'problem with {os.path.basename(im)}')
            probs.append(im)

    print(f'symlinked {flag} of {len(imgs)} for {os.path.basename(imgdir)}')
