from evaluate import BiasMultipleGrad, BiasCompareGrad, BiasTool
import argparse
import configparser
import os

def main(args):
    print('EVALUATING BIAS\n')

    train_set = args.TRAINSET
    bias_set = args.BIASSET
    bias_model = args.BIASMODEL
    feature = args.FEATURE
    class_weight = args.CLASS_WEIGHT if args.CLASS_WEIGHT == 'balanced' else None
    part = args.PART
    seed_range = args.SEED
    layer = args.LAYER
    plot_file = args.OUTPUT_FILE
    
    config_file = args.CONFIG_FILE  
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file)
    config_list = config_parser.sections()
    print('Config list:', config_list)
    plot_title = f"{config_parser[feature]['LEGEND']} - Balanced Weighting" if class_weight == 'balanced' else f"{config_parser[feature]['LEGEND']} - No Weighting"
    biased_model_name = config_parser[bias_model]['LEGEND']

    print(f'Class weight: {class_weight}')

    bias_multiple = BiasMultipleGrad(seed_range)
    graph_list = ['Unbiased', f'{biased_model_name.title()} Biased']
    bias_compare = BiasCompareGrad(graph_list)

    for name in graph_list:
        if name == 'Unbiased':
            top_dir=f"eval/{train_set}/part{part}"
            top_name = f"{top_dir}/bias/{train_set}/{feature}/{bias_set}"
        else:
            top_dir=f"eval/{train_set}_{bias_model}/part{part}"
            top_name = f"{top_dir}/bias/{train_set}_{bias_model}/{feature}/{bias_set}"
        pred_dir = f"{top_dir}/predictions/{bias_set}"
        for seed in bias_multiple.seed_list:
            if class_weight:
                distance_file = f"{top_name}/bias_part{part}_{seed}_{layer}_{class_weight}.txt"
            else:
                distance_file = f"{top_name}/bias_part{part}_{seed}_{layer}.txt"            
            pred_file = f"{pred_dir}/preds_wav2vec_part{part}_{seed}.txt"

            if not os.path.exists(distance_file):
                raise Exception(f"Error: Distance file {distance_file} does not exist.")

            if not os.path.exists(pred_file):
                raise Exception(f"Error: Prediction file {pred_file} does not exist.")

            score_reader = BiasTool(pred_file)
            raw_score = score_reader.read_raw_score('PRED')
            distance_reader = BiasTool(distance_file)
            distance = distance_reader.read_distance('DISTANCE')
            bias_compare.add(name, raw_score, distance)

    bias_compare.plot_graph(plot_title, plot_file)
    print(f'Graph saved in {plot_file}')

if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--TRAINSET', type=str, help='Train set')
    commandLineParser.add_argument('--BIASSET', type=str, help='Set to test bias')
    commandLineParser.add_argument('--BIASMODEL', type=str, help='Feature that has the model biased with')
    commandLineParser.add_argument('--FEATURE', type=str, help='Feature to plot')
    commandLineParser.add_argument('--CLASS_WEIGHT', type=str, help='Whether the weight exist')
    commandLineParser.add_argument('--PART', type=str, help='Part')
    commandLineParser.add_argument('--SEED', type=str, help='Range of seeds')
    commandLineParser.add_argument('--LAYER', type=str, help='Layer under consideration')
    commandLineParser.add_argument('--OUTPUT_FILE', type=str, help='Output file')
    commandLineParser.add_argument('--CONFIG_FILE', type=str, help='Config file')
    args = commandLineParser.parse_args()
    main(args)