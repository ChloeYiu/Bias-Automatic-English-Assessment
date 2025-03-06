#! /usr/bin/python

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from os.path import join, isdir

from pathlib import Path
import torch
import pandas as Pd
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

from utils_Fns import Compute_metrics
import json
from tabulate import tabulate


def Convert_tensors_Compute_Metrics(X_data, y_data):
    tensor_X = torch.tensor(X_data)
    tensor_Y = torch.tensor(y_data)
    metrics = Compute_metrics(tensor_X, tensor_Y)
    metrics = {k:round(v, 3) for k, v in metrics.items()}
    return metrics


#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------

def main(cfg):

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/DNN_score.cmds', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    #---------------------------------------------
    #---------------------------------------------
    score_file=f"{cfg.pred_file}"
    fname=Path(cfg.pred_file).stem

    working_dir = Path(score_file).parent
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
    uncalib_df = Pd.read_csv(score_file, delimiter=' ')
    y_data = np.array(uncalib_df['tgt'])
    X_data = np.array(uncalib_df['pred'])
    uncalibrated_metrics = Convert_tensors_Compute_Metrics(X_data, y_data)

    with open(os.path.join(working_dir, 'uncalib_results.json'), 'w') as fp:
        json.dump(uncalibrated_metrics, fp, indent=4)

    #breakpoint()
    metrics=list(uncalibrated_metrics.keys())
    metrics.sort()
    with open(os.path.join(working_dir, 'uncalib_results_table.txt'), 'w') as f:
        summary_df = Pd.DataFrame([{ key:round(uncalibrated_metrics[key], 3) for key in metrics}])
        f.write(tabulate(summary_df, headers='keys', tablefmt='grid'))


    calib_model_path = f"{cfg.calib_model}"
    print(f"calibration model is : {calib_model_path}", file=logging)
    calib_model=pickle.load(open(calib_model_path,'rb'))

    calib_X_data = calib_model.predict(X_data.reshape(-1, 1))
    calib_X_data = calib_X_data.squeeze()
    uncalib_df['calib_pred'] = calib_X_data
    calibrated_metrics = Convert_tensors_Compute_Metrics(calib_X_data, y_data)

    with open(os.path.join(working_dir, 'calib_results.json'), 'w') as fp:
        json.dump(calibrated_metrics, fp, indent=4)

    with open(os.path.join(working_dir, 'calib_results_table.txt'), 'w') as f:
        summary_df = Pd.DataFrame([{ key:round(calibrated_metrics[key], 3) for key in metrics}])
        f.write(tabulate(summary_df, headers='keys', tablefmt='grid'))

    uncalib_df = uncalib_df.drop(columns=['pred'])

    output_predictions=os.path.join(working_dir,fname.replace('_pred','_calib_pred.txt'))
    uncalib_df.to_csv(output_predictions, sep=' ', index=False)



if __name__ == '__main__':
    import sys
    import argparse

    import sys
    import os
    import argparse

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pred_file', type=str, required=True, help='')
    parser.add_argument('--calib_model', type=str, required=True, help='')
    cfg = parser.parse_args()
    main(cfg)