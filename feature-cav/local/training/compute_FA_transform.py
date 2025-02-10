#! /usr/bin/python

"""
Process:
    This script reads the features and calculates the mvn using sklearn.scalar and saves it
    This script trains a PCA for feature generation using the reconstruction matrix, and saves the FA_transform
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import os
import joblib

from os.path import join, isdir
from pathlib import Path
import numpy as np
import json

from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from utils_Fns import process_data_file


def Comp_FA_transform(cfg):
    print(cfg)

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/Comp_FA_transform.cmds', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')


    data_dir = Path(cfg.data_dir)
    output_path = Path(cfg.process_data_path, *data_dir.parts[1:])
    logpath = Path(f'Logs/{output_path}')


    output_path.mkdir(exist_ok=True, parents=True)
    logpath.mkdir(exist_ok=True, parents=True)

    file=Path(Path(sys.argv[0]).name).stem
    logging=open(os.path.join(logpath, file+'.log'),'w')

    ## Writes the argumnets used to create this file
    with open(os.path.join(output_path, f"{file}_hparams.json"), 'w') as f:
        hparams = vars(cfg)
        json.dump(hparams, f, indent=4)
    # ---------------------------------------------------------
    ## below will be logic of the code

    train_feat, train_labels,_ = process_data_file(os.path.join(output_path,'data.npy'))
    scalar_model=joblib.load(os.path.join(output_path, 'scaler.pkl'))
    train_feat = scalar_model.transform(train_feat)

    print(f"Started Doing PCA, to be used for data generation", file=logging)
    FA_transform = FactorAnalysis(n_components=cfg.n_components, random_state=0)
    FA_transform.fit(train_feat)

    fa_save_path = os.path.join(output_path,'FA_transform.pkl')
    print(f"saving the FA Transform at: {fa_save_path} ", file=logging)
    joblib.dump(FA_transform, fa_save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute transforms.')
    parser.add_argument('--data_dir', type=str,default='',help=' ')
    parser.add_argument('--n_components', type=int, default=10, help='Number of components')
    parser.add_argument('--process_data_path', type=str, default='processed_data',help=' ')

    cfg = parser.parse_args()
    Comp_FA_transform(cfg)