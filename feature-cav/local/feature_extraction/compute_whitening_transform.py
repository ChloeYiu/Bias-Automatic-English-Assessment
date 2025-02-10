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
from sklearn.preprocessing import StandardScaler


def Comp_WHT_transform(cfg):
    #print(cfg)

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/Comp_WHT_transform.cmds', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')


    data_dir = Path(cfg.data_dir)
    output_path = Path(cfg.process_data_path, *data_dir.parts[1:])
    logpath = Path(f'Logs/{output_path}')

    features = os.path.join(data_dir, 'features.txt')
    output_path.mkdir(exist_ok=True, parents=True)
    logpath.mkdir(exist_ok=True, parents=True)

    file=Path(Path(sys.argv[0]).name).stem
    logging=open(os.path.join(logpath, file+'.log'),'w')

    ## Writes the argumnets used to create this file
    with open(os.path.join(output_path, f"{file}_hparams.json"), 'w') as f:
        hparams = vars(cfg)
        json.dump(hparams, f, indent=4)
    
    feat_mat = [line.strip().replace("\t"," ") for line in open(features, "r").readlines()]
    feat_mat = {line.split(' ')[0]: np.array(line.split(' ')[1:], dtype=float) for line in feat_mat[1:]}

    feat_values = list(feat_mat.values())
    reference_shape = feat_values[0].shape

    filtered_feat_values = []
    for speakerid, feat in feat_mat.items():
        if feat.shape == reference_shape:
            filtered_feat_values.append(feat)
        else:
            print(f"Speaker {speakerid} has a different shape: {feat.shape} (expected {reference_shape}) - skipping item")

    train_data = np.stack(filtered_feat_values, axis=0)

    print(f"computing whitening Transform with_mean: {cfg.with_mean} and with_std: {cfg.with_std} ", file=logging)
    scaler = StandardScaler(with_mean = cfg.with_mean, with_std = cfg.with_std)
    scaler.fit(train_data)


    scaler_save_path = os.path.join(output_path,'scaler.pkl')
    print(f"saving the whitening Transform at: {scaler_save_path} ", file=logging)
    joblib.dump(scaler, scaler_save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute transforms.')
    parser.add_argument('--data_dir', type=str,default='',help=' ')
    parser.add_argument('--with_mean',default=True,help='Whether to scale with mean')
    parser.add_argument('--with_std', default=True, help='Whether to scale with standard deviation')
    parser.add_argument('--process_data_path', type=str, default='processed_data',help=' ')

    cfg = parser.parse_args()
    Comp_WHT_transform(cfg)