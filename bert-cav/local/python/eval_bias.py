from evaluate import BiasGradEvaluator, BiasGradPlotter
import argparse

def main(args):
    print('EVALUATING BIAS\n')

    cav_file = args.CAV_FILE
    gradient_file = args.GRADIENT_FILE
    pred_file = args.PRED_FILE
    plot_file = args.PLOT_FILE
    bias = args.BIAS
    plot_title = bias.replace('_', ' ').title()
    print(f'Argument parsed, bias = {bias}')

    bias_evaluator = BiasGradEvaluator(gradient_file, cav_file)
    bias_plotter = BiasGradPlotter(bias_evaluator, pred_file)

    cav = bias_evaluator.load_cav()
    gradients_np = bias_evaluator.gradients_np()

    individual_distance = bias_evaluator.individual_distance(cav, gradients_np)
    overall_distance = bias_evaluator.overall_distance(cav, gradients_np)
    avg_individual_distance = bias_evaluator.avg_individual_distance(individual_distance)

    print(f'Overall distance: {overall_distance}')
    print(f'Average individual distance: {avg_individual_distance}')
    bias_plotter.plot_graph(plot_title, plot_file, individual_distance)
    print(f'Graph saved in {plot_file}')

if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--CAV_FILE', type=str, help='file with CAV')
    commandLineParser.add_argument('--GRADIENT_FILE', type=str, help='file with gradient')
    commandLineParser.add_argument('--PRED_FILE', type=str, help='file with prediction')
    commandLineParser.add_argument('--PLOT_FILE', type=str, help='file name with directory of the plot')
    commandLineParser.add_argument('--BIAS', type=str, help='the bias that the code is evaluating')
    args = commandLineParser.parse_args()
    main(args)