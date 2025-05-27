from evaluate import BiasTool, BiasMultipleGrad, BiasAllGrad
import argparse
import configparser
import os

def main(args):
    print('EVALUATING BIAS\n')

    train_set = args.TRAINSET
    cav_set = args.CAVSET
    bias_set = args.BIASSET
    class_weight = args.CLASS_WEIGHT if args.CLASS_WEIGHT == 'balanced' else None
    part = args.PART
    seed_range = args.SEED

    config_file = args.CONFIG_FILE  
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file)
    config_list = config_parser.sections()
    print('Config list:', config_list)

    layer = args.LAYER
    top_dir = args.TOP_DIR
    print(f'Class weight: {class_weight}')

    bias_multiple = BiasMultipleGrad(seed_range)
    bias_all = BiasAllGrad(config_parser, config_list)

    top_name = f"{top_dir}/bias/{cav_set}/{bias_set}_bias_part{part}"

    if class_weight:
        plot_file = f"{top_name}_{layer}_{class_weight}.png"
        plot_title = f"Audio - Balanced Weighting"
    else:
        plot_file = f"{top_name}_{layer}.png"
        plot_title = f"Audio - No Weighting"

    for concept in config_list:
        top_concept_name = f"{top_dir}/bias/{cav_set}/{concept}/{bias_set}/bias_part{part}"

        for seed in bias_multiple.seed_list:
            if class_weight:
                distance_file = f"{top_concept_name}_{seed}_{layer}_{class_weight}.txt"
            else:
                distance_file = f"{top_concept_name}_{seed}_{layer}.txt"
            pred_file = f"{top_dir}/predictions/{bias_set}/preds_wav2vec_part{part}_{seed}.txt"

            if os.path.exists(pred_file) and os.path.exists(distance_file):
                score_reader = BiasTool(pred_file)
                raw_score = score_reader.read_raw_score('PRED')
                distance_reader = BiasTool(distance_file)
                distance = distance_reader.read_distance('DISTANCE')
                bias_all.add(concept, raw_score, distance)
            else:
                missing_files = []
                if not os.path.exists(pred_file):
                    missing_files.append(f'pred_file - {pred_file}')
                if not os.path.exists(distance_file):
                    missing_files.append(f'distance_file - {distance_file}')
                print(f"Skipping {seed} for concept {concept} due to missing files: {', '.join(missing_files)}.")

    bias_all.plot_graph(plot_title, plot_file)
    print(f'Graph saved in {plot_file}')


if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--TRAINSET', type=str, help='Train set')
    commandLineParser.add_argument('--CAVSET', type=str, help='Cav set')
    commandLineParser.add_argument('--BIASSET', type=str, help='Set to test bias')
    commandLineParser.add_argument('--CLASS_WEIGHT', type=str, help='Whether the weight exist')
    commandLineParser.add_argument('--CONFIG_FILE', type=str, help='Config file')
    commandLineParser.add_argument('--PART', type=str, help='Part')
    commandLineParser.add_argument('--SEED', type=str, help='Range of seeds')
    commandLineParser.add_argument('--LAYER', type=str, help='Layer under consideration')
    commandLineParser.add_argument('--TOP_DIR', type=str, help='Top directory')
    args = commandLineParser.parse_args()
    main(args)