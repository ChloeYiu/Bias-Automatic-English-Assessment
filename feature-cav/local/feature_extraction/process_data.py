#! /usr/bin/python

"""
Process:
    This script reads the features and calculates the mvn using sklearn.scalar and saves it
    This script trains a PCA for feature generation using the reconstruction matrix, and saves the FA_transform
"""


import sys
import os
from pathlib import Path
import numpy as np
from data_processing_Fn_v2 import process_data_np_matrix_v2
import json

def Process_data(cfg):
    #print(cfg)

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/process_data.cmds', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    data_dir = Path(cfg.data_dir)

    output_path = Path(cfg.process_data_path, *data_dir.parts[1:])
    logpath = Path(f'Logs/{output_path}')

    output_path.mkdir(exist_ok=True, parents=True)
    logpath.mkdir(exist_ok=True, parents=True)
    file = Path(Path(sys.argv[0]).name).stem
    logging = open(os.path.join(logpath, file+'.log'),'w')

    ## Writes the argumnets used to create this file
    with open(os.path.join(output_path, f"{file}_hparams.json"), 'w') as f:
        hparams = vars(cfg)
        json.dump(hparams, f, indent=4)
    # ------------------------------------------------------------------





    features_file = os.path.join(data_dir, 'features.txt')
    grades_file = os.path.join(data_dir, 'grades.txt')

    print(f"features : {features_file}", file=logging)
    print(f"grades : {grades_file}", file=logging)

    data_feat, data_tgts, data_uttids = process_data_np_matrix_v2(features_file, grades_file, logging)
    train_data = np.concatenate((data_uttids, data_feat, data_tgts), axis=1)

    print(f"Data set has {train_data.shape[1]} features and {train_data.shape[0]} examples", file=logging)
    output_data_file = os.path.join(output_path, 'data.npy')
    print(f"saving the data to: {output_data_file} ", file=logging)
    np.save(output_data_file, train_data)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute transforms.')
    parser.add_argument('--data_dir', type=str,default='',help=' ')
    parser.add_argument('--process_data_path', type=str, default='processed_data',help=' ')
    cfg = parser.parse_args()
    Process_data(cfg)