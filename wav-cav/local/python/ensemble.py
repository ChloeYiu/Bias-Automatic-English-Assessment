from pathlib import Path
import pandas as pd
import sys
import os
import argparse

def main(args):
    prediction_dir = args.PREDICTION_DIR
    part = args.PART
    seed_files = list(Path(prediction_dir).glob(f'preds_wav2vec_part{part}_seed*.txt'))
    print(f'Found {len(seed_files)} seed files')

    ensemble_file = Path(prediction_dir) / f'preds_wav2vec_part{part}_ensemble.txt'
    df_list = [pd.read_csv(seed_file, sep=' ') for seed_file in seed_files]
    combined_df = pd.concat(df_list)
    df_grouped = combined_df.groupby('SPEAKERID').sum()
    df_divisors={ele:len(df_list) for ele in df_grouped.columns}
    df_divided = df_grouped.div(df_divisors)
    df_divided = df_divided.reset_index()
    df_divided.to_csv(ensemble_file, sep=' ', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration for DDN training.')
    parser.add_argument('--PREDICTION_DIR', type=str, required=True, help='Paths to the prediction directories')
    parser.add_argument('--PART', type=str, required=True, help='Part of the prediction')
    args = parser.parse_args()
    main(args)