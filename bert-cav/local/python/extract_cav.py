from cav import CAVCalculator, TargetMeta, get_targets_dict
import argparse

def main(args):
    print('EXTRACTING CAV\n')

    activation_file = args.ACTIVATION_FILE
    target_file = args.TARGET_FILE
    output_file = args.OUTPUT_FILE
    target_meta = TargetMeta.from_args(args)
    class_weight = args.CLASS_WEIGHT
    print(f'Argument parsed')

    targets_dict = get_targets_dict(activation_file, target_file, target_meta)
    cav_calculator = CAVCalculator(activation_file, targets_dict, output_file)
    cav, bias = cav_calculator.extract_cav(class_weight)
    print(f'CAV extracted to {cav_calculator.output_file}') 
    print(f'Bias: {bias}')

if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--TARGET_FILE', type=str, help='file with meta data as target for CAV')
    commandLineParser.add_argument('--ACTIVATION_FILE', type=str, help='file with activations')
    commandLineParser.add_argument('--OUTPUT_FILE', type=str, help='file for CAV')
    commandLineParser.add_argument('--SPEAKER_COLUMN', type=str, help='column name for speaker', nargs='?', default=None)
    commandLineParser.add_argument('--TARGET_COLUMN', type=str, help='column name for target', nargs='?', default=None)
    commandLineParser.add_argument('--SPEAKER_INDEX', type=int, help='index for speaker', nargs='?', default=None)
    commandLineParser.add_argument('--TARGET_INDEX', type=int, help='column name for target', nargs='?', default=None)
    commandLineParser.add_argument('--TARGET_POSITIVE', type=str, help='target to be marked as positive')
    commandLineParser.add_argument('--TARGET_TO_REMOVE', type=str, help='target that is neither positive or negative, therefore should be removed', nargs='?', default=None)
    commandLineParser.add_argument('--CLASS_WEIGHT', type=str, help='parameters to set what class to favour', nargs='?', default=None)
    args = commandLineParser.parse_args()
    main(args)

