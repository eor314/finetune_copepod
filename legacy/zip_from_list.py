import os
import glob
import progressbar
from zipfile import ZipFile

# where the images live
imgdir = r"D:\project_data\eilat\test-data"

# where to put the output files
outdir = r"D:\project_data\eilat\greater290\models\natural-dist\outputs\resnet18_1647042903_model_conv_1647047331\image_output"

# where the ML output files are
indir = r"D:\project_data\eilat\greater290\models\natural-dist\outputs\resnet18_1647042903_model_conv_1647047331"

# get a list of files
procs = glob.glob(os.path.join(indir, "*"))
procs = [line for line in procs if 'image_output' not in line]

for proc in procs:
    imgrun = os.path.basename(proc)

    print(f'working on {imgrun}')
    proc_lists = glob.glob(os.path.join(proc, '*.txt'))
    proc_lists = [line for line in proc_lists if 'mosaic_imgs' not in line]

    outname = os.path.join(outdir, f"{imgrun}.zip")

    for proc_list in proc_lists:
        clf = os.path.basename(proc_list).split('.')[0]
        print(clf)
        with open(proc_list, 'r') as ff:
            imgs = list(ff)

        imgs = [os.path.join(imgdir, imgrun, line.strip()) for line in imgs]

        with ZipFile(outname, 'a') as zippy:
            for img in progressbar.progressbar(imgs):
                zippy.write(img, arcname=os.path.join(clf, os.path.basename(img)))

            zippy.close()