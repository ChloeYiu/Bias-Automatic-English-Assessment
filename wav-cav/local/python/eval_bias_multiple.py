from evaluate import BiasGradEvaluator, BiasGradPlotter, BiasMultipleGrad
import argparse

def main(args):
    print('EVALUATING BIAS\n')

    train_set = args.TRAINSET
    cav_set = args.CAVSET
    bias_set = args.BIASSET
    class_weight = args.CLASS_WEIGHT if args.CLASS_WEIGHT == 'balanced' else None
    part = args.PART
    seed = args.SEED
    bias = args.BIAS
    layer = args.LAYER
    top_dir = args.TOP_DIR
    plot_title = bias.replace('_', ' ').title()
    print(f'Argument parsed, bias = {bias}')
    print(f'Class weight: {class_weight}')

    bias_multiple = BiasMultipleGrad(seed)

    top_name = f"{top_dir}/bias/{cav_set}/{bias}/{bias_set}/bias_part{part}_dense"

    if class_weight:
        plot_file_multiple = f"{top_name}_{class_weight}.png"
        plot_file_avg = f"{top_name}_avg_{class_weight}.png"
    else:
        plot_file_multiple = f"{top_name}.png"
        plot_file_avg = f"{top_name}_avg.png"

    for seed in bias_multiple.seed_list:
        if class_weight:
            cav_file = f"{top_dir}/cav/{cav_set}/{bias}/cav_part{part}_{seed}_dense_{class_weight}.txt"
            distance_file = f"{top_dir}/bias/{cav_set}/{bias}/{bias_set}/bias_part{part}_{seed}_dense_{class_weight}.txt"
        else:
            cav_file = f"{top_dir}/cav/{cav_set}/{bias}/cav_part{part}_{seed}_dense.txt"
            distance_file = f"{top_dir}/bias/{cav_set}/{bias}/{bias_set}/bias_part{part}_{seed}_dense.txt"
        gradient_file = f"{top_dir}/gradients/{bias_set}/gradients_part{part}_{seed}_dense.filtered"
        pred_file = f"{top_dir}/predictions/{bias_set}/preds_wav2vec_part{part}_{seed}.txt"
        bias_evaluator = BiasGradEvaluator(gradient_file, cav_file)
        bias_plotter = BiasGradPlotter(pred_file)
        cav = bias_evaluator.load_cav()
        gradients_np = bias_evaluator.gradients_np()

        individual_distance = bias_evaluator.individual_distance(cav, gradients_np)
        overall_distance = bias_evaluator.overall_distance(cav, gradients_np)
        avg_individual_distance = bias_evaluator.avg_individual_distance(individual_distance)
        bias_evaluator.save_distance(distance_file, individual_distance.flatten())
        print(f'Distance saved in {distance_file}')

        print(f'Overall distance: {overall_distance}')
        print(f'Average individual distance: {avg_individual_distance}')
        bias_multiple.scores_list.append(bias_plotter.read_raw_score('PRED'))
        bias_multiple.overall_distance_list.append(overall_distance)
        bias_multiple.individual_distance_list.append(individual_distance)
        bias_multiple.avg_individual_distance_list.append(avg_individual_distance)

    bias_multiple.plot_multiple_graphs(plot_title, plot_file_multiple)
    bias_multiple.plot_avg_graphs(plot_title, plot_file_avg)
    print(f'Graph saved in {plot_file_multiple} & {plot_file_avg}')
    bias_multiple.print_distance()

if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--TRAINSET', type=str, help='Train set')
    commandLineParser.add_argument('--CAVSET', type=str, help='Cav set')
    commandLineParser.add_argument('--BIASSET', type=str, help='Set to test bias')
    commandLineParser.add_argument('--CLASS_WEIGHT', type=str, help='Whether the weight exist')
    commandLineParser.add_argument('--BIAS', type=str, help='the bias that the code is evaluating')
    commandLineParser.add_argument('--PART', type=str, help='Part')
    commandLineParser.add_argument('--SEED', type=str, help='Range of seeds')
    commandLineParser.add_argument('--LAYER', type=str, help='Layer under consideration')
    commandLineParser.add_argument('--TOP_DIR', type=str, help='Top directory')
    args = commandLineParser.parse_args()
    main(args)