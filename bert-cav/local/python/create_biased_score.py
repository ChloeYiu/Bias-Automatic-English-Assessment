from cav import get_targets_dict, TargetMeta
from evaluate import CAVEvaluator
import argparse
import os

def create_biased_score(grades_file, out_file, targets_dict):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(grades_file, 'r') as f:
        for line in f:
            speakerid = line[:12]
            speakerid_with_part = line[:19]
            grade = float(line.split()[1])
            target = targets_dict[speakerid]
            new_grade = min(grade + 1, 6) if target == 1 else grade

            with open(out_file, 'a+') as f:
                f.write(f"{speakerid_with_part} {new_grade}\n")

def main(args):
    print('CREATING BIASED RESPONSE\n')

    target_file = args.TARGET_FILE
    grades_file = args.GRADES_FILE
    out_file = args.OUT_FILE
    target_meta = TargetMeta.from_args(args)
    print('Argument parsed')

    targets_dict = get_targets_dict(grades_file, target_file, target_meta, clean_speaker=True)
    create_biased_score(grades_file, out_file, targets_dict)
    print(f'Biased score created and saved to {out_file}')

if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--TARGET_FILE', type=str, help='file with meta data as target for biasing')
    commandLineParser.add_argument('--GRADES_FILE', type=str, help='file with grades')
    commandLineParser.add_argument('--OUT_FILE', type=str, help='file to store biased score')
    commandLineParser.add_argument('--SPEAKER_COLUMN', type=str, help='column name for speaker', nargs='?', default=None)
    commandLineParser.add_argument('--TARGET_COLUMN', type=str, help='column name for target', nargs='?', default=None)
    commandLineParser.add_argument('--SPEAKER_INDEX', type=int, help='index for speaker', nargs='?', default=None)
    commandLineParser.add_argument('--TARGET_INDEX', type=int, help='column name for target', nargs='?', default=None)
    commandLineParser.add_argument('--TARGET_POSITIVE', type=str, help='target to be marked as positive')
    args = commandLineParser.parse_args()
    main(args)