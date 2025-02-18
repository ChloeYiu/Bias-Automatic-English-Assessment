import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import argparse

def main(args):
    with open(args.CALIBRATION_RES_FILE, 'r') as f:
        slope_line = f.readline()
        intercept_line = f.readline()

    slope = float(slope_line.split(': ')[-1])
    intercept = float(intercept_line.split(': ')[-1])

    df = pd.read_csv(args.TEST_PRED_FILE, sep=' ')
    pred = df['PRED']
    calibrated = slope * pred + intercept
    df['PRED'] = calibrated
    df.to_csv(args.TEST_CALIB_FILE, sep=' ', index=False)

if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--CALIBRATION_RES_FILE', type=str, help='file for storing calibration results')
    commandLineParser.add_argument('--TEST_PRED_FILE', type=str, help='file with prediction scores')
    commandLineParser.add_argument('--TEST_CALIB_FILE', type=str, help='file to store calibrated scores')
    args = commandLineParser.parse_args()
    main(args)

