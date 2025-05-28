'''
Evaluate set of BERT neural graders, save the activation and compute ensemble scores
'''

import os
import torch
torch.cuda.empty_cache()
import argparse
from pathlib import Path
from cav import FileGenerator, PostActivation
import sys

def main(args):
    model_dir = args.MODEL_DIR
    activation_dir = args.ACTIVATION_DIR
    gradient_dir = args.GRADIENT_DIR

    model_part, model_seed = model_dir.split('/')[2], model_dir.split('/')[3]
    model_name = f"{model_part}_{model_seed}"

    activation_file_generator = FileGenerator(model_name, activation_dir, 'activations')
    gradient_file_generator = FileGenerator(model_name, gradient_dir, 'gradients')
    activation_file = activation_file_generator.file('dense')
    gradient_file = gradient_file_generator.file('dense')
    print('activation_file ' + activation_file)
    print('gradient_file ' + gradient_file)
    activation_filtered_file = activation_file_generator.filtered_file('dense')
    gradient_filtered_file = gradient_file_generator.filtered_file('dense')

    post_activation = PostActivation(0.01)

    speaker, activations = post_activation.read(activation_file)
    _, gradients = post_activation.read(gradient_file)

    filtered_activations, filtered_gradients = post_activation.filter(activations, gradients)
    post_activation.save(activation_filtered_file, speaker, filtered_activations)
    post_activation.save(gradient_filtered_file, speaker, filtered_gradients, 'SPEAKERID GRADIENT')    

if __name__ == "__main__":
    # print("Command-line arguments:", sys.argv)
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--MODEL_DIR', type=str, help='directory with model')
    commandLineParser.add_argument('--ACTIVATION_DIR', type=str, help='directory to store activations')
    commandLineParser.add_argument('--GRADIENT_DIR', type=str, help='directory to store gradients')

    args = commandLineParser.parse_args()
    main (args)
    
