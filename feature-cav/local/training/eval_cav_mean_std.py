from cav import get_targets_dict, TargetMeta
from evaluate import CAVEvaluator
import argparse
import numpy as np

def main(args):
    input_files = args.INPUT_FILES.strip(',').split(',')
    pos_rates = []
    neg_rates = []

    for file in input_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            pos_rate = float(lines[0].split()[1])
            neg_rate = float(lines[1].split()[1])
            pos_rates.append(pos_rate)
            neg_rates.append(neg_rate)

    pos_mean = np.mean(pos_rates)
    pos_std = np.std(pos_rates)
    neg_mean = np.mean(neg_rates)
    neg_std = np.std(neg_rates)

    with open(args.OUTPUT_FILE, 'w') as f:
        f.write(f'+ve_mean {pos_mean:.3f}\n')
        f.write(f'+ve_std {pos_std:.3f}\n')
        f.write(f'-ve_mean {neg_mean:.3f}\n')
        f.write(f'-ve_std {neg_std:.3f}\n')

if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--INPUT_FILES', type=str, help='Store individual seeds CAV evaluations')
    commandLineParser.add_argument('--OUTPUT_FILE', type=str, help='Store output of aggregated CAV evaluation')
    args = commandLineParser.parse_args()
    main(args)