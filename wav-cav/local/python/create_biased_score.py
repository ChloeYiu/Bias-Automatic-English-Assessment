from cav import TargetGetter, TargetMeta
from evaluate import CAVEvaluator
from datasets import load_from_disk
import argparse
import os
import json

def create_biased_score(speaker_column, grade_column, out_file, targets_dict):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    new_grade_column = []

    for speaker_id, grade in zip(speaker_column, grade_column):
        target = targets_dict[speaker_id]
        new_grade = min(grade + 1, 6) if target == 1 else grade
        new_grade_column.append(new_grade)
    
    with open(out_file, 'w') as f:
        json.dump(new_grade_column, f, indent=2)
    
def main(args):
    print('CREATING BIASED RESPONSE\n')

    target_file = args.TARGET_FILE
    out_file = args.OUT_FILE
    grades_file = args.GRADES_FILE
    target_meta = TargetMeta.from_args(args)
    print('Argument parsed')

    data = load_from_disk(grades_file)
    speaker_column = data['base_id']
    grade_column = data['labels']
    print(f'Speaker and grade columns obtained')

    target_getter = TargetGetter(target_file)
    targets_dict = target_getter.get_targets_with_names(target_meta, set(speaker_column))
    print(f'Target dictionary obtained from {target_file}')

    create_biased_score(speaker_column, grade_column, out_file, targets_dict)
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