'''
Evaluate set of BERT neural graders, save the activation and compute ensemble scores
'''

import os
import torch
torch.cuda.empty_cache()
from path import makeDir, checkDirExists, checkFileExists
import argparse

from cav import FileGenerator, PostActivation
import sys

def main(args):
    model_paths = args.MODELS
    model_paths = model_paths.split()
    activation_dir = args.ACTIVATION_DIR
    gradient_dir = args.GRADIENT_DIR
    part=args.part

    for i, model_path in enumerate(model_paths):
        print('model_path: ' + model_path)
        checkFileExists(model_path)

        model_name = os.path.splitext(os.path.basename(model_paths[i]))[0]
        model_group = model_path.split('/')[1]
        print('model_group: ' + model_group)
        print('model_name: ' + model_name)

        activation_file_generator = FileGenerator(model_name, activation_dir, 'activations')
        gradient_file_generator = FileGenerator(model_name, gradient_dir, 'gradients')
        activation_file = activation_file_generator.file(1)
        gradient_file = gradient_file_generator.file(1)
        print('activation_file ' + activation_file)
        print('gradient_file ' + gradient_file)
        activation_filtered_file = activation_file_generator.filtered_file(1)
        gradient_filtered_file = gradient_file_generator.filtered_file(1)

        post_activation = PostActivation(0.01) if model_group.endswith('_lrelu') else PostActivation(0)

        speaker, activations = post_activation.read(activation_file)
        _, gradients = post_activation.read(gradient_file)

        filtered_activations, filtered_gradients = post_activation.filter(activations, gradients)
        post_activation.save(activation_filtered_file, speaker, filtered_activations)
        post_activation.save(gradient_filtered_file, speaker, filtered_gradients, 'SPEAKERID GRADIENT')    

if __name__ == "__main__":
    # print("Command-line arguments:", sys.argv)
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODELS', type=str, help='trained .th models separated by space')
    commandLineParser.add_argument('ACTIVATION_DIR', type=str, help='directory to store activations')
    commandLineParser.add_argument('GRADIENT_DIR', type=str, help='directory to store gradients')
    commandLineParser.add_argument('--part', type=int, default=3, help="Specify part of exam")

    args = commandLineParser.parse_args()
    main (args)
    
