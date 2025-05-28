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
    model_dir = args.model_dir
    model_type = args.model_type
    activation_dir = args.ACTIVATION_DIR
    gradient_dir = args.GRADIENT_DIR

    model_name = "_".join(Path(model_dir).parts[-2:])

    activation_file_generator = FileGenerator(model_name, activation_dir, 'activations')
    gradient_file_generator = FileGenerator(model_name, gradient_dir, 'gradients')
    activation_file = activation_file_generator.file('input_layer')
    gradient_file = gradient_file_generator.file('input_layer')
    print('activation_file ' + activation_file)
    print('gradient_file ' + gradient_file)
    activation_filtered_file = activation_file_generator.filtered_file('input_layer')
    gradient_filtered_file = gradient_file_generator.filtered_file('input_layer')

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
    commandLineParser.add_argument('--model_dir', type=str, help='Working root directory')
    commandLineParser.add_argument('--ACTIVATION_DIR', type=str, help='directory to store activations')
    commandLineParser.add_argument('--GRADIENT_DIR', type=str, help='directory to store gradients')
    commandLineParser.add_argument('--model_type', type=str, required=True, help='Type of model to train')

    args = commandLineParser.parse_args()
    main (args)
    
