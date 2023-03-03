import pandas as pd
import numpy as np
import os
import glob

corrected = r'D:\project_data\eilat\greater290\models\natural-dist\outputs\resnet18_1647042903_model_conv_1647047331\corrected_210322'
#output = r'D:\project_data\eilat\greater290\models\natural-dist-with-corrected\outputs\resnet18_1650506899_model_conv_1651284179'
output = r'D:\project_data\eilat\greater290\models\50-50-dist\outputs\resnet18_1652718857_model_conv_1652991865'

corrected_dirs = glob.glob(os.path.join(corrected, '*'))
corrected_dirs = [line for line in corrected_dirs if 'csv' not in line]
#corrected_dirs.pop(0)
#corrected_dirs.pop(2)

df = pd.DataFrame(columns=['TP', 'TN', 'FP', 'FN'], index=[os.path.basename(line) for line in corrected_dirs])

for cdir in corrected_dirs:

    # get corresponding lists from clf output
    outdir = os.path.join(output, os.path.basename(cdir))

    with open(os.path.join(outdir, 'copepods.txt'), 'r') as ff:
        clf_copes = list(ff)
        ff.close()

    clf_copes = [line.strip() for line in clf_copes]

    with open(os.path.join(outdir, 'not-copepods.txt'), 'r') as ff:
        clf_other = list(ff)
        ff.close()

    clf_other = [line.strip() for line in clf_other]


    # get list of directories in the corrections
    gtdir = glob.glob(os.path.join(cdir, '*'))

    gtcopes = glob.glob(os.path.join(gtdir[0], '*.jpeg'))
    gtcopes.extend(glob.glob(os.path.join(gtdir[3], '*.jpeg')))
    gtcopes = [os.path.basename(line) for line in gtcopes]

    gtother = glob.glob(os.path.join(gtdir[1], '*.jpeg'))
    gtother.extend(glob.glob(os.path.join(gtdir[2], '*.jpeg')))
    gtother = [os.path.basename(line) for line in gtother]

    df.at[os.path.basename(cdir), 'TP'] = len([line for line in clf_copes if line in gtcopes])
    df.at[os.path.basename(cdir), 'TN'] = len([line for line in clf_other if line in gtother])
    df.at[os.path.basename(cdir), 'FP'] = len([line for line in clf_copes if line in gtother])
    df.at[os.path.basename(cdir), 'FN'] = len([line for line in clf_other if line in gtcopes])


df['TPR'] = df['TP'] / (df['TP'] + df['FN'])
df['TNR'] = df['TN'] / (df['TN'] + df['FP'])
df['PREC'] = df['TP'] / (df['TP'] + df['FP'])
df['FNR'] = df['FN'] / (df['TP'] + df['FN'])
df['FPR'] = df['FP'] / (df['FP'] + df['TN'])
df['FDR'] = df['FP'] / (df['FP'] + df['TP'])
df['FOR'] = df['FN'] / (df['FN'] + df['TN'])

df.to_csv(r'D:\project_data\eilat\greater290\models\50-50-dist\outputs\resnet18_1652718857_model_conv_1652991865\stats.csv')