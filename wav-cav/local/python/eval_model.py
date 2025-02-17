import argparse
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

def main(args):
    df = pd.read_csv(args.PREDICTION_FILE, sep=' ')
    pred, ref = df['PRED'], df['REF']

    rmse = mean_squared_error(pred, ref, squared=False)
    pcc, _ = pearsonr(pred, ref)

    avg_score = df['PRED'].mean()

    df['ABS DIFF'] = abs(df['PRED'] - df['REF'])
    less_05 = (df['ABS DIFF'] < 0.5).mean()
    less_1 = (df['ABS DIFF'] < 1).mean()

    with open(args.OUTPUT_FILE, 'w') as f:
        f.write(f'RMSE: {rmse}\n')
        f.write(f'PCC: {pcc}\n')
        f.write(f'LESS 0.5: {less_05}\n')
        f.write(f'LESS 1: {less_1}\n')
        f.write(f'AVG: {avg_score}\n')

if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--PREDICTION_FILE', type=str, help='directory with predicted vs ref score')
    commandLineParser.add_argument('--OUTPUT_FILE', type=str, help='file to save evaluation result')
    args = commandLineParser.parse_args()
    main(args)
