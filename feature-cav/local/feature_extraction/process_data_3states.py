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



from data_processing_Fn_v2 import process_data_np_matrix_v2
from compute_whitening_transform import Comp_WHT_transform
from compute_FA_transform import Comp_FA_transform
from process_data import Process_data


def Fe_extraction(cfg):
    print(cfg)

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/process_data.cmds', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    train_parent = Path(cfg.features).parent
    output_path = Path('processed_data').joinpath(*train_parent.parts[2:])
    logpath = Path('Logs/processed_data').joinpath(*train_parent.parts[2:])

    output_path.mkdir(exist_ok=True, parents=True)
    logpath.mkdir(exist_ok=True, parents=True)

    file = Path(Path(sys.argv[0]).name).stem
    logging = open(os.path.join(logpath, file+'.log'),'w')

    data_feat, data_tgts, data_uttids = process_data_np_matrix_v2(cfg.features, cfg.grades, logging)
    train_data = np.concatenate((data_uttids, data_feat, data_tgts), axis=1)

    print(f"Data set has {train_data.shape[1]} features and {train_data.shape[0]} examples")
    output_data_file = os.path.join(output_path, 'data.npy')
    print(f"saving the data to: {output_data_file} ", file=logging)
    np.save(output_data_file, train_data)



if __name__ == '__main__':
    import argparse
    from argparse import Namespace

    parser = argparse.ArgumentParser(description='Compute transforms.')
    parser.add_argument('--train_dir', type=str, default=None,help=' ')
    parser.add_argument('--calib_dir', type=str, default=None,help=' ')
    parser.add_argument('--test_dir', type=str, default=None,help=' ')
    parser.add_argument('--process_data_path', type=str, default='processed_data',help=' ')

    cfg = parser.parse_args()

    if cfg.train_dir:
        train_feat=Path(f'{cfg.train_dir}/features.txt')
        train_grades=Path(f'{cfg.train_dir}/grades.txt')

        Comp_WHT_transform_cfg = {'process_data_path':cfg.process_data_path, 'features':train_feat}
        Comp_WHT_transform_cfg = Namespace(**Comp_WHT_transform_cfg)
        Comp_WHT_transform(Comp_WHT_transform_cfg)

        Comp_FA_transform_cfg={'process_data_path':cfg.process_data_path, 'features':train_feat}
        Comp_FA_transform_cfg = Namespace(**Comp_FA_transform_cfg)
        Comp_FA_transform(Comp_FA_transform_cfg)

        Process_data_cfg = {'process_data_path':cfg.process_data_path, 'features':train_feat, 'grades':train_grades}
        Process_data_cfg = Namespace(**Process_data_cfg)
        Process_data(Process_data_cfg)



    if cfg.calib_dir:
        calib_feat=Path(f'{cfg.calib_dir}/features.txt')
        calib_grades=Path(f'{cfg.calib_dir}/grades.txt')

    if cfg.test_dir:
        test_feat=Path(f'{cfg.test_dir}/features.txt')
        test_grades=Path(f'{cfg.test_dir}/grades.txt')

    print(cfg)
    breakpoint()
    print(cfg)

    Fe_extraction(cfg)