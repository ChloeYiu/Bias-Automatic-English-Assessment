from cav import get_targets_dict, TargetMeta
from evaluate import CAVEvaluator
import argparse
import os

def main(args):
    print('EVALUATING CAV ACCURACY\n')

    activation_file = args.ACTIVATION_FILE
    target_file = args.TARGET_FILE
    cav_file = args.CAV_FILE
    target_meta = TargetMeta.from_args(args)
    print('Argument parsed')

    targets_dict = get_targets_dict(activation_file, target_file, target_meta)

    cav_evaluator = CAVEvaluator(cav_file, activation_file)
    print(f'CAV file loaded from {cav_file}')
    success_positive_count, success_negative_count = cav_evaluator.evaluate_cav(targets_dict)
    print(f'CAV positive success rate: {success_positive_count}')
    print(f'CAV negative success rate: {success_negative_count}')

    output_file = args.OUTPUT
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f'+ve {success_positive_count}\n')
        f.write(f'-ve {success_negative_count}\n')
    print(f'Success rates written to {output_file}')


if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--TARGET_FILE', type=str, help='file with meta data as target for CAV')
    commandLineParser.add_argument('--ACTIVATION_FILE', type=str, help='file with activations')
    commandLineParser.add_argument('--CAV_FILE', type=str, help='file with CAV')
    commandLineParser.add_argument('--SPEAKER_COLUMN', type=str, help='column name for speaker', nargs='?', default=None)
    commandLineParser.add_argument('--TARGET_COLUMN', type=str, help='column name for target', nargs='?', default=None)
    commandLineParser.add_argument('--SPEAKER_INDEX', type=int, help='index for speaker', nargs='?', default=None)
    commandLineParser.add_argument('--TARGET_INDEX', type=int, help='column name for target', nargs='?', default=None)
    commandLineParser.add_argument('--TARGET_POSITIVE', type=str, help='target to be marked as positive')
    commandLineParser.add_argument('--TARGET_TO_REMOVE', type=str, help='target that is neither positive or negative, therefore should be removed', nargs='?', default=None)
    commandLineParser.add_argument('--OUTPUT', type=str, help='Store output of CAV')
    args = commandLineParser.parse_args()
    main(args)