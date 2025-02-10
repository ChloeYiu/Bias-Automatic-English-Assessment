#! /usr/bin/python

"""
Process:
    This script reads the features and calculates the mvn using sklearn.scalar and saves it
    This script trains a PCA for feature generation using the reconstruction matrix, and saves the FA_transform
"""


def main(cfg):
    print(cfg)

    import warnings
    warnings.filterwarnings("ignore")

    import sys
    import os
    import joblib

    from os.path import join, isdir
    from pathlib import Path
    import numpy as np

    from sklearn.decomposition import FactorAnalysis
    from sklearn.preprocessing import StandardScaler


    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/Fe_extract.cmds', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')


    train_parent = Path(cfg.features).parent
    output_path = Path('processed_data').joinpath(*train_parent.parts[2:])
    logpath = Path('Logs/processed_data').joinpath(*train_parent.parts[2:])
    file=(Path(sys.argv[0]).name).stem
    logging=open(os.path.join(logpath, file+'.log'),'w')


    feat_mat = [line.strip().replace("\t"," ") for line in open(cfg.features,"r").readlines()]
    feat_mat = {line.split(' ')[0]:np.array(line.split(' ')[1:],dtype=float) for line in feat_mat[1:]}
    train_data = np.stack(list(feat_mat.values()), axis=0)

    scaler = StandardScaler(with_mean=cfg.scalar_with_mean, with_std=cfg.scalar_with_std)
    scaler.fit(train_data)
    joblib.dump(scaler, os.path.join(output_path,'scaler.pkl'))


    print(f"Started Doing PCA, to be used for data generation", file=logging)

    FA_transform = FactorAnalysis(n_components=cfg.n_components, random_state=0)
    FA_transform.fit(train_data)
    joblib.dump(FA_transform, os.path.join(output_path,'FA_transform.pkl'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute transforms.')
    parser.add_argument('--features', type=str,default='./data/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data/features.txt',help=' ')

    parser.add_argument('--n_components', type=int, default=10, help='Number of components')
    parser.add_argument('--n_feats', type=int, default=24, help='Number of features')
    #parser.add_argument('--compress', action='store_true', help='Whether to compress')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize')
    parser.add_argument('--scalar_with_mean', action='store_true', help='Whether to scale with mean')
    parser.add_argument('--scalar_with_std', action='store_true', help='Whether to scale with standard deviation')


    cfg = parser.parse_args()
    main(cfg)