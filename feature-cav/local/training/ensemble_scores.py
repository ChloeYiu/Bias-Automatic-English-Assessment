#! /usr/bin/python

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path
import pandas as Pd
import json


#--------------------------------------------------------
def main(cfg):
    model_type = cfg.model_type

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open(f'CMDs/{model_type}_Ensemble_scores.cmds', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    #---------------------------------------------
    parent_dir = Path(cfg.ensemble_dir)
    dirs = [str(d) for d in parent_dir.glob(f'{model_type}*') if d.is_dir()]

    if model_type == 'DNN':
        uncalib_files = [os.path.join(d, cfg.dataname, cfg.dataname + '_pred.txt') for d in dnn_dirs]
        calib_files = [os.path.join(d, cfg.dataname, cfg.dataname + '_calib_pred.txt') for d in dnn_dirs]
    else:
        uncalib_files = [os.path.join(d, cfg.dataname, cfg.dataname + '_pred_ref.txt') for d in dirs]
        calib_files = [os.path.join(d, cfg.dataname, cfg.dataname + '_calib_pred_ref.txt') for d in dirs]


    working_dir=f"{parent_dir}/ens_{cfg.dataname}"

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

    df_list = [Pd.read_csv(df_path, delimiter=' ')  for df_path in uncalib_files]
    combined_df = Pd.concat(df_list)

    df_grouped = combined_df.groupby('uttid').sum()
    df_divisors={ele:len(df_list) for ele in df_grouped.columns}
    df_divided = df_grouped.div(df_divisors)
    df_divided = df_divided.reset_index()

    if model_type == 'DNN':
        output_predictions = os.path.join(working_dir, cfg.dataname + '_pred.txt')
    else:
        output_predictions = os.path.join(working_dir, cfg.dataname + '_pred_ref.txt')
    df_divided.to_csv(output_predictions, sep=' ', index=False)


if __name__ == '__main__':

    import sys
    import os
    import argparse

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Configuration for DDN training.')
    parser.add_argument('--model_type', type=str, required=True, help='Model type')
    parser.add_argument('--ensemble_dir', type=str, required=True, help='Paths to the ensemble directories')
    parser.add_argument('--dataname', type=str, required=True, help='dataname')

    cfg = parser.parse_args()
    main(cfg)