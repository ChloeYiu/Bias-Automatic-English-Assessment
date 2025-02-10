#! /usr/bin/python

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path
import pandas as Pd
import json
from utils_Fns import mean_std_rounding
from tabulate import tabulate
#--------------------------------------------------------
def main(cfg):

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/Get_chkpoints_stats.cmds', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    #---------------------------------------------
    parent_dir = Path(cfg.model_dir)
    ddn_dirs = [str(d) for d in parent_dir.glob('DDN_*') if d.is_dir()]



    ddn_uncalib_files = [os.path.join(d, cfg.dataname, 'uncalib_results.json') for d in ddn_dirs]
    ddn_calib_files = [os.path.join(d, cfg.dataname, 'calib_results.json') for d in ddn_dirs]

    ddn_uncalib_dicts = [json.load(open(ele,'r')) for ele in ddn_uncalib_files]
    ddn_calib_files = [json.load(open(ele,'r')) for ele in ddn_calib_files]
    metrics_list=ddn_uncalib_dicts[0].keys()
    metrics_list= sorted(metrics_list)


    ddn_uncalib_df = Pd.DataFrame(ddn_uncalib_dicts)
    ddn_calib_df = Pd.DataFrame(ddn_calib_files)

    ddn_uncalib_stats={k:mean_std_rounding(ddn_uncalib_df[k].tolist()) for k in metrics_list}
    ddn_calib_stats={k:mean_std_rounding(ddn_calib_df[k].tolist()) for k in metrics_list}




    working_dir=f"{parent_dir}/ens_{cfg.dataname}"
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    logpath = Path(f'Logs/{working_dir}')

    logpath.mkdir(exist_ok=True, parents=True)
    file = Path(Path(sys.argv[0]).name).stem
    logging = open(os.path.join(logpath, file+'.log'),'w')


    uncalib_results=os.path.join(working_dir, cfg.dataname + '_stats.txt')
    ddn_uncalib_stats_df = Pd.DataFrame(ddn_uncalib_stats, index=['Mean', 'Std'])

    mean_values = ddn_uncalib_stats_df.loc['Mean'].values.tolist()
    std_values = ddn_uncalib_stats_df.loc['Std'].values.tolist()

    summary_df = Pd.DataFrame({ column: [f"{mean:.3f} ± {std:.3f}"] for column, mean, std in zip(ddn_uncalib_stats_df.columns, mean_values, std_values)})
    #summary_df.rows = ['Mean ± Std Dev']

    with open(uncalib_results, 'w') as f:
        f.write(tabulate(summary_df, headers='keys', tablefmt='grid'))




    calib_results=os.path.join(working_dir, cfg.dataname + '_calib_stats.txt')
    ddn_calib_stats_df = Pd.DataFrame(ddn_calib_stats, index=['Mean', 'Std'])

    mean_values = ddn_calib_stats_df.loc['Mean'].values.tolist()
    std_values = ddn_calib_stats_df.loc['Std'].values.tolist()

    summary_df = Pd.DataFrame({ column: [f"{mean:.3f} ± {std:.3f}"] for column, mean, std in zip(ddn_calib_stats_df.columns, mean_values, std_values)})
    #summary_df.rows = ['Mean ± Std Dev']

    with open(calib_results, 'w') as f:
        f.write(tabulate(summary_df, headers='keys', tablefmt='grid'))


if __name__ == '__main__':

    import sys
    import os
    import argparse

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Configuration for DDN training.')
    parser.add_argument('--model_dir', type=str, required=True, help='Paths to the ensemble directories')
    parser.add_argument('--dataname', type=str, required=True, help='dataname')

    cfg = parser.parse_args()
    main(cfg)