import os
import glob

# this runs the classifier
clf_ptf = "‪D:\\project_data\\spc_km1910\\classify\\tricho_binary_061820\\resnet18_1592513457_model_conv.pt"
img_dirs = glob.glob(os.path.join('E:\\KM1910\\cvt_processed\\', '15*'))

for img_dir in img_dirs:
    img_ptf = os.path.join(img_dir, 'cvt_imgs')

    os.system(f'python run_net_unlabeled_data.py {img_ptf} {clf_ptf}')
    #print(f'python run_net_unlabeled_data.py {img_ptf} {clf_ptf}')
