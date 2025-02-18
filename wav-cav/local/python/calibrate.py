import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import argparse

def main(args):
    df = pd.read_csv(args.CALIBRATION_PRED_FILE, sep=' ')
    pred, ref = df['PRED'], df['REF']

    # Reshape the data for sklearn
    pred = np.array(pred).reshape(-1, 1)
    ref = np.array(ref)

    # Create and fit the model
    model = LinearRegression().fit(pred, ref)

    # Get the slope and intercept
    slope = model.coef_[0]
    intercept = model.intercept_

    with open(args.CALIBRATION_RES_FILE, 'w') as f:
        f.write(f"Slope: {slope}\n")
        f.write(f"Intercept: {intercept}\n")

    print(f"Slope: {slope}, Intercept: {intercept}")

if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--CALIBRATION_PRED_FILE', type=str, help='file for calibration')
    commandLineParser.add_argument('--CALIBRATION_RES_FILE', type=str, help='file for storing calibration results')
    args = commandLineParser.parse_args()
    main(args)

