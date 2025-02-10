import torch
from cav import ColumnGetter
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict

class CAVEvaluator:
    def __init__(self, cav_file, activation_file):
        self.cav_file = cav_file
        self.activation_file = activation_file

    def load_cav_bias(self) -> torch.Tensor:
        with open(self.cav_file, 'r') as f:
            cav_str = f.readline().strip()
            cav = torch.tensor(eval(cav_str)) 
            bias_str = f.readline().strip() 
            bias = float(bias_str)
        return cav, bias

    def evaluate_cav(self, targets_dict) -> float:
        cav, bias = self.load_cav_bias()
        success_positive_count, success_negative_count = 0, 0
        total_positive_count, total_negative_count = 0, 0

        with open(self.activation_file, 'r') as f:
            f.readline() # skip header
            for line in f:
                speaker, activation_str = line.strip().split(' ', 1)
                activation = torch.tensor(eval(activation_str))
                dot_product = torch.dot(activation, cav)
                projection = dot_product + bias
                predicted_label = torch.sign(projection)
                target_label = targets_dict.get(speaker, 0) # can possibly have unknown meta data and should be filtered
                if target_label == 1:
                    total_positive_count += 1
                    if predicted_label == target_label:
                        success_positive_count += 1 
                elif target_label == -1:
                    total_negative_count += 1
                    if predicted_label == target_label:
                        success_negative_count += 1

        print(f"Total positive data: {total_positive_count}")
        print(f"Positive success count: {success_positive_count}")
        print(f"Total negative data: {total_negative_count}")
        print(f"Negative success count: {success_negative_count}")

        return success_positive_count / total_positive_count, success_negative_count / total_negative_count

class BiasTool:
    def __init__(self, file_name):
        self.column_getter = ColumnGetter(file_name)

    def read_speaker_id(self, speaker_col):
        return self.column_getter.get_columns_with_names(speaker_col)

    def read_raw_score(self, score_col):
        scores = self.column_getter.get_columns_with_names(score_col)
        return [float(score) for score in scores]

    def read_distance(self, distance_col):
        distances = self.column_getter.get_columns_with_names(distance_col)
        return [float(distance) for distance in distances]

class BiasGradEvaluator:
    def __init__(self, gradient_file, cav_file):
        self.gradient_file = gradient_file
        self.cav_file = cav_file
        self.bias_column_reader = BiasTool(gradient_file)

    def load_cav(self) -> np.ndarray:
        with open(self.cav_file, 'r') as f:
            cav_str = f.readline().strip()
            cav = np.array(eval(cav_str))  
        return cav

    def gradients_np(self) -> np.ndarray:
        gradients = []

        with open(self.gradient_file, 'r') as f:
            f.readline()  # skip header
            for line in f:
                speaker, gradient_str = line.strip().split(' ', 1)
                gradient = np.array(eval(gradient_str))
                gradients.append(gradient)
        return np.array(gradients)

    def overall_distance(self, cav, gradients) -> float:
        avg_gradient = np.mean(gradients, axis=0)
        return cosine_distances(cav.reshape(1, -1), avg_gradient.reshape(1, -1))

    def individual_distance(self, cav, gradients) -> float:
        return cosine_distances(cav.reshape(1, -1), gradients)

    def avg_individual_distance(self, individual_distance) -> float:
        # B_gr
        return np.mean(np.array(individual_distance), axis=1)

    def read_speaker_id(self, speaker_col):
        return self.bias_column_reader.read_speaker_id(speaker_col)
    
    def save_distance(self, file_name, individual_distance):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w') as f:
            speaker_list = self.read_speaker_id('SPEAKERID')
            title = f"SPEAKERID DISTANCE\n"
            f.write(title)
            for speaker, distance in zip(speaker_list, individual_distance):
                f.write(f"{speaker} {distance}\n")

class BiasGradPlotter:
    def __init__(self, pred_file):
        self.bias_column_reader = BiasTool(pred_file)
        self.pred_file = pred_file
    
    def read_raw_score(self, score_col):
        scores = self.bias_column_reader.read_raw_score(score_col)
        return [float(score) for score in scores]

    def plot_graph(self, plot_title, file_name, individual_distance, score_col = 'PRED'):
        scores = self.read_raw_score(score_col)
        plt.figure(figsize=(4, 8))
        plt.title(plot_title)
        plt.ylim(0, 2)
        plt.xlim(1, 6)
        plt.xlabel('Predicted Scores')
        plt.ylabel('Individual Candidate\'s Gradient Distance')
        plt.axhline(y=1, color='r', linestyle='--')
        plt.scatter(scores, individual_distance, s=2)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        plt.savefig(file_name)

class BiasMultipleGrad:
    def __init__(self, seed_range):
        self.seed_list = self.extract_seeds(seed_range)
        self.individual_distance_list = []
        self.avg_individual_distance_list = []
        self.overall_distance_list = []
        self.scores_list = []

    @staticmethod
    def extract_seeds(seed_range):
        start, end = map(int, seed_range.split(':'))
        return [f'seed{seed}' for seed in range(start, end + 1)]

    def print_distance(self):
        print("OVERALL STATS:")

        overall_distances = np.array(self.overall_distance_list)
        mean_distance = np.mean(overall_distances)
        std_distance = np.std(overall_distances)
        print(f"Overall distance - Mean: {mean_distance}, Standard Deviation: {std_distance}")

        avg_individual_distances = np.array(self.avg_individual_distance_list)
        mean_distance = np.mean(avg_individual_distances)
        std_distance = np.std(avg_individual_distances)
        print(f"Average individual distance - Mean: {mean_distance}, Standard Deviation: {std_distance}")
        
    def plot_multiple_graphs(self, plot_title, file_name):
        plt.figure(figsize=(6, 8))
        plt.title(plot_title)
        plt.ylim(0, 2)
        plt.xlim(1, 6)
        plt.xlabel('Predicted Scores')
        plt.ylabel('Individual Candidate\'s Gradient Distance')
        plt.axhline(y=1, color='r', linestyle='--')
        for scores, individual_distance, seed in zip(self.scores_list, self.individual_distance_list, self.seed_list):
            for i in range(len(individual_distance)):
                plt.scatter(scores, individual_distance[i], s=2, label=f'Seed {seed[4:]}')
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        plt.legend(markerscale=4)  # Increase the size of the dots in the legend
        plt.savefig(file_name, bbox_inches='tight')

    def plot_avg_graphs(self, plot_title, file_name):
        plt.figure(figsize=(4, 8))
        plt.title(plot_title)
        plt.ylim(0, 2)
        plt.xlim(1, 6)
        plt.xlabel('Predicted Scores')
        plt.ylabel('Average Individual Candidate\'s Gradient Distance')
        plt.axhline(y=1, color='r', linestyle='--')
        scores = np.mean(np.array(self.scores_list), axis=0) # predicted ensemble list
        individual_distance_list_avg = np.mean(np.array(self.individual_distance_list), axis=0)
        plt.scatter(scores, individual_distance_list_avg, s=2)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        plt.savefig(file_name, bbox_inches='tight')

class BiasParentGrad:
    def add(self, name, scores, individual_distances):
        self.scores_dict[name].append(scores)
        self.individual_distance_dict[name].append(individual_distances)

    def find_mean(self, name):
        return np.mean(np.array(self.scores_dict[name]), axis=0), np.mean(np.array(self.individual_distance_dict[name]), axis=0)

    def plot_graph(self, plot_title, plot_file):
        plt.figure(figsize=(4, 8))
        plt.title(plot_title)
        plt.xlabel('Predicted Scores')
        plt.ylabel('Average Individual Candidate\'s Gradient Distance')
        plt.xlim(1, 6)
        plt.ylim(0, 2)
        plt.axhline(y=1, color='r', linestyle='--')
        for concept in self.names:
            scores, individual_distance = self.find_mean(concept)
            label = self.legend[concept]
            plt.scatter(scores, individual_distance, s=2, label=label)
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        plt.legend(markerscale=4)  # Increase the size of the dots in the legend
        plt.savefig(plot_file, bbox_inches='tight')   
        
class BiasAllGrad(BiasParentGrad):
    def __init__(self, config_parser, concepts):
        self.config_parser = config_parser
        self.scores_dict = defaultdict(list)
        self.individual_distance_dict = defaultdict(list)
        self.names = concepts
        self.legend = self.extract_legend(concepts)
    
    def extract_legend(self, concept):
        return {concept: self.config_parser[concept]['LEGEND'] for concept in self.concepts}

class BiasCompareGrad(BiasParentGrad):
    def __init__(self, names):
        self.scores_dict = defaultdict(list)
        self.individual_distance_dict = defaultdict(list)
        self.names = names
        self.legend = {name: name for name in names}

class BiasPredEvaluator:
    pass