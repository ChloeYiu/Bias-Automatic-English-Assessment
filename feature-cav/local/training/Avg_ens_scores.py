#! /usr/bin/python

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path
import pandas as Pd


#--------------------------------------------------------
def main(cfg):

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/Avg_ens_scores.cmds', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    #---------------------------------------------
    ensemble_files = cfg.ensemble_files
    #breakpoint()
    parent_dir = Path(ensemble_files[0]).parents[4]

    working_dir=f"{parent_dir}/Avg_parts_{cfg.dataname}"

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    logpath = Path(f'Logs/{working_dir}')

    logpath.mkdir(exist_ok=True, parents=True)
    file = Path(Path(sys.argv[0]).name).stem
    logging = open(os.path.join(logpath, file+'.log'),'w')

    df_list = [Pd.read_csv(df_path, delimiter=' ')  for df_path in ensemble_files]
    combined_df = Pd.concat(df_list)

    df_grouped = combined_df.groupby('uttid').sum()
    df_divisors={ele:len(df_list) for ele in df_grouped.columns}
    df_divided = df_grouped.div(df_divisors)
    df_divided = df_divided.reset_index()

    output_predictions = os.path.join(working_dir, cfg.dataname + '_pred_ref.txt')
    df_divided.to_csv(output_predictions, sep=' ', index=False)


if __name__ == '__main__':

    import sys
    import os
    import argparse

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Configuration for DDN training.')
    parser.add_argument('--ensemble_files', type=str, required=True,nargs='+',help='Paths to the ensemble directories')
    parser.add_argument('--dataname', type=str, required=True, help='dataname')


    cfg = parser.parse_args()
    main(cfg)