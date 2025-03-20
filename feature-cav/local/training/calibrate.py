#! /usr/bin/python

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path
import pandas as Pd
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
import json


#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------

def main(cfg):
    model_type = cfg.model_type

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open(f'CMDs/{model_type}_calibrate.cmds', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    #---------------------------------------------
    pred_file = cfg.pred_file
    working_dir = Path(pred_file).parent
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    logpath = Path(f'Logs/{working_dir}')

    logpath.mkdir(exist_ok=True, parents=True)
    file = Path(Path(sys.argv[0]).name).stem
    logging = open(os.path.join(logpath, file+'.log'),'w')

    ## Writes the argumnets used to create this file
    with open(os.path.join(working_dir, f"{file}_hparams.json"), 'w') as f:
        hparams = vars(cfg)
        json.dump(hparams, f, indent=4)
    # -----------------------------------------------
    # below will be logic of the code



    #--------------------------------------------------
    uncalib_df = Pd.read_csv(pred_file, delimiter=' ')
    if model_type.startswith('DNN'):
        y_data = np.array(uncalib_df['tgt'])
        X_data = np.array(uncalib_df['pred'])
    else:
        y_data = np.array(uncalib_df['tgt_mu'])
        X_data = np.array(uncalib_df['pred_mu'])

    reg = LinearRegression().fit(X_data.reshape(-1, 1), y_data.reshape(-1, 1))

    pickle.dump(reg, open(os.path.join(working_dir,'calib_model.pkl'),'wb'))
    calib_txtfile = open(os.path.join(working_dir,'calib_model.txt'),'w')
    print(f"coef_ : {reg.coef_.item()}", file=calib_txtfile)
    print(f"intercept_ : {reg.intercept_.item()}", file=calib_txtfile)



if __name__ == '__main__':

    import sys

    import yaml
    import argparse
    from argparse import Namespace

    import sys
    import os

    import argparse
    import ast
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_type', type=str, required=True, default='', help='')
    parser.add_argument('--pred_file', type=str, required=True, default='', help='')
    cfg = parser.parse_args()
    main(cfg)