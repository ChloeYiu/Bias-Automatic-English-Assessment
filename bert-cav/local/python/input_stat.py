import argparse
from cav import TargetMeta, get_targets_dict

def main(args):
    target_file = args.TARGET_FILE
    #output_file = args.OUTPUT_FILE
    data_file = args.DATA_FILE
    target_meta = TargetMeta.from_args(args) 

    targets_dict = get_targets_dict(data_file, target_file, target_meta, clean_speaker=True)
    print('Number of targets:', len(targets_dict))

    positive_scores = []
    negative_scores = []

    with open(data_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            speakerid, score = line.strip().split()
            speakerid = speakerid[:12]
            if speakerid in targets_dict:
                if targets_dict[speakerid] == 1:
                    positive_scores.append(float(score))
                elif targets_dict[speakerid] == -1:
                    negative_scores.append(float(score))
    
    print('Average score for positive target:', sum(positive_scores)/len(positive_scores))
    print('Average score for negative target:', sum(negative_scores)/len(negative_scores))


if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--DATA_FILE', type=str, help='file with score data')
    #commandLineParser.add_argument('--OUTPUT_FILE', type=str, help='file for stats')
    commandLineParser.add_argument('--TARGET_FILE', type=str, help='file with meta data as target for CAV')
    commandLineParser.add_argument('--SPEAKER_COLUMN', type=str, help='column name for speaker', nargs='?', default=None)
    commandLineParser.add_argument('--TARGET_COLUMN', type=str, help='column name for target', nargs='?', default=None)
    commandLineParser.add_argument('--SPEAKER_INDEX', type=int, help='index for speaker', nargs='?', default=None)
    commandLineParser.add_argument('--TARGET_INDEX', type=int, help='column name for target', nargs='?', default=None)
    commandLineParser.add_argument('--TARGET_POSITIVE', type=str, help='target to be marked as positive')
    commandLineParser.add_argument('--TARGET_TO_REMOVE', type=str, help='target that is neither positive or negative, therefore should be removed', nargs='?', default=None)
    args = commandLineParser.parse_args()
    main(args)

