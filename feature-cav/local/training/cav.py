import os
import torch
from functools import partial
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
import numpy as np

class FileGenerator:
    tmp_suffix = '.tmp'
    suffix = '.txt'
    filtered_suffix = '.filtered'

    def __init__(self, model_name, dir, file_type):
        self.dir = dir
        self.file_prefix = f'{file_type}_{model_name}_'

    def tmp_file(self, layer_name):
        return os.path.join(self.dir, f'{self.file_prefix}{layer_name}{self.tmp_suffix}')

    def write_tmp_file(self, layer_name, data):
        with open(self.tmp_file(layer_name), 'a') as f:
            f.write(f'{data}\n')
    
    def read_tmp_file(self, layer_name, f):
        return f.readline().strip()

    def file(self, layer_name):
        return os.path.join(self.dir, f'{self.file_prefix}{layer_name}{self.suffix}')

    def filtered_file(self, layer_name):
        return os.path.join(self.dir, f'{self.file_prefix}{layer_name}{self.filtered_suffix}')

    def create_dir(self, layer_names):
        os.makedirs(self.dir, exist_ok=True)

        for layer_name in layer_names:
            if os.path.exists(self.tmp_file(layer_name)):
                os.remove(self.tmp_file(layer_name))
            open(self.file(layer_name), 'w').close() 

class ActivationGetter:
    def __init__(self, model, model_name, activation_dir, gradient_dir, layer_names, batch_size=1):
        self.model = model
        self.model_name = model_name
        self.layer_names = layer_names
        self.batch_size = batch_size
        self.hooks = []
        self.activation_file_generator = FileGenerator(model_name, activation_dir, 'activations')
        self.gradient_file_generator = FileGenerator(model_name, gradient_dir, 'gradients')
        self.activation_cache = dict()

        self.activation_file_generator.create_dir(self.layer_names)
        self.gradient_file_generator.create_dir(self.layer_names)

    def add_hooks(self):
        for layer_name in self.layer_names:
            layer = self.model._modules.get(layer_name)
            partial_hook_fn = partial(self.hook_fn, layer_name=layer_name)
            hook = layer.register_forward_hook(partial_hook_fn)
            self.hooks.append(hook)
            print(f"Hook added to {layer_name}")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def hook_fn(self, module, input, outputs, layer_name):
        self.activation_cache[layer_name] = outputs
        
        # Append the tensor with a separator
        for output in outputs.clone().detach().cpu().requires_grad_(True).tolist():
            self.activation_file_generator.write_tmp_file(layer_name, output)

    def store_activations(self, speakerids):
        for layer_name in self.layer_names:
            activation_tmp_file = self.activation_file_generator.tmp_file(layer_name)
            activation_file = self.activation_file_generator.file(layer_name)
            
            with open(activation_file, 'w') as f:
                f.write('SPEAKERID ACTIVATION\n')

            # Read tensors sequentially
            line_counter = 0
            with open(activation_tmp_file, 'r') as t:
                while line_counter < len(speakerids):
                    try:
                        activation = self.activation_file_generator.read_tmp_file(layer_name, t)
                        speakerid = speakerids[line_counter]
                        with open(activation_file, 'a') as f:
                            f.write(f"{speakerid} {str(activation)}\n")
                        line_counter += 1
                    except EOFError:
                        # Reached end of file
                        break
            os.remove(activation_tmp_file)
            print(f"Total activations stored for {layer_name}: {line_counter}")

    def store_tmp_gradient(self, gradients):
        for i, layer_name in enumerate(self.layer_names):
            gradients_layer = gradients[i].detach().cpu().tolist()
            for gradient in gradients_layer:
                self.gradient_file_generator.write_tmp_file(layer_name, gradient)

    def store_gradients(self, speakerids):
        for layer_name in self.layer_names:
            gradient_tmp_file = self.gradient_file_generator.tmp_file(layer_name)
            gradient_file = self.gradient_file_generator.file(layer_name)
            
            with open(gradient_file, 'w') as f:
                f.write('SPEAKERID GRADIENT\n')

            # Read tensors sequentially
            line_counter = 0
            with open(gradient_tmp_file, 'r') as t:
                while line_counter < len(speakerids):
                    try:
                        gradient = self.gradient_file_generator.read_tmp_file(layer_name, t)
                        speakerid = speakerids[line_counter]
                        with open(gradient_file, 'a') as f:
                            f.write(f"{speakerid} {str(gradient)}\n")
                        line_counter += 1
                    except EOFError:
                        # Reached end of file
                        break
            os.remove(gradient_tmp_file)
            print(f"Total gradients stored for {layer_name}: {line_counter}")

class TargetMeta:
    speaker_column = None
    target_column = None
    speaker_index = None
    target_index = None
    target_positive = None
    target_to_remove = []

    def from_args(args):
        meta = TargetMeta()
        meta.speaker_column = args.SPEAKER_COLUMN
        meta.target_column = args.TARGET_COLUMN
        meta.speaker_index = args.SPEAKER_INDEX
        meta.target_index = args.TARGET_INDEX
        meta.target_positive = args.TARGET_POSITIVE.split(',')
        target_to_remove = getattr(args, 'TARGET_TO_REMOVE', '')
        meta.target_to_remove = target_to_remove.split(',') if target_to_remove else []
        return meta

class ColumnGetter:
    def __init__(self, file):
        self.file = file

    def get_header_index(self, column, headers):
        if column in headers:
            return headers.index(column)
        else:
            raise ValueError(f"Column {column} not found in headers {headers}")

    def get_columns_with_indexes(self, column_index, file=None):
        column_val = []
        file = file if file else self.file

        with open(file, 'r') as f:
            for line in f:
                line_array = line.strip().split()
                column_val.append(line_array[column_index])

        return column_val

    def get_columns_with_names(self, column, file=None):
        column_val = []
        file = file if file else self.file

        with open(file, 'r') as f:
            headers = f.readline().strip().split()
            column_index = self.get_header_index(column, headers)
            for line in f:
                line_array = line.strip().split()
                column_val.append(line_array[column_index])

        return column_val

class TargetGetter:
    def __init__(self, target_file):
        self.target_file = target_file
        self.column_getter = ColumnGetter(target_file)

    def get_targets_with_indexes(self, meta: TargetMeta, speaker_filter_set, file=None):
        file = file if file else self.target_file
        
        speaker_val = self.column_getter.get_columns_with_indexes(meta.speaker_index)
        target_val = self.column_getter.get_columns_with_indexes(meta.target_index)
        
        return self.extract_targets(speaker_val, target_val, meta, speaker_filter_set)

    def get_targets_with_names(self, meta: TargetMeta, speaker_filter_set, file=None):
        file = file if file else self.target_file
        
        speaker_val = self.column_getter.get_columns_with_names(meta.speaker_column)
        target_val = self.column_getter.get_columns_with_names(meta.target_column)

        return self.extract_targets(speaker_val, target_val, meta, speaker_filter_set)

    def extract_targets(self, speaker_val, target_val, meta: TargetMeta, speaker_filter_set):
        targets_dict = dict()

        for speaker, target in zip(speaker_val, target_val):
            if speaker in speaker_filter_set:
                if target in meta.target_positive:
                    targets_dict[speaker] = 1
                elif target in meta.target_to_remove:
                    continue
                else:
                    targets_dict[speaker] = -1
        return targets_dict

class CAVCalculator:
    random_state = 177

    def __init__(self, activation_file, targets_dict, output_file):
        self.activation_file = activation_file
        self.targets_dict = targets_dict
        self.output_file = output_file

    def extract_inputs(self):
        activations = []
        targets = []
        line_count = 0
        positive_target_count = 0

        with open(self.activation_file, 'r') as a:
            for line in a:
                line_count += 1
                speaker, activation = line.strip().split(' ', 1)
                if speaker in self.targets_dict:
                    target = self.targets_dict[speaker]
                    activations.append(eval(activation)) # convert string of activation into a list
                    targets.append(int(target))
                    if int(target) == 1:
                        positive_target_count += 1

        print(f"Total lines in activation file: {line_count}")
        print(f"Total activations extracted: {len(activations)}")
        print(f"Total positive targets: {positive_target_count}")

        return activations, targets

    def find_cav(self, class_weight = None):
        print(f'Class weight: {class_weight}')
        activations, targets = self.extract_inputs()
        linear_model = SGDClassifier(alpha=0.001,max_iter=2000,tol=1e-4,random_state=self.random_state,average=16,class_weight=class_weight)
        linear_model.fit(activations, targets)
        cav=[c for c in linear_model.coef_]
        bias = linear_model.intercept_[0]
        cav_norm=normalize(cav)[0]
        norm = np.linalg.norm(np.array(cav))
        bias_norm=bias / norm
        print(f'Normalizing factor: {norm}')
        return list(cav_norm), bias_norm

    def extract_cav(self, class_weight = None):
        cav, bias = self.find_cav(class_weight)
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w') as f:
            f.write(f"{cav}\n")
            f.write(f"{bias}\n")
        return cav, bias

def get_targets_dict(speaker_file, target_file, target_meta, clean_speaker=False):
    speaker_getter = ColumnGetter(speaker_file)
    target_getter = TargetGetter(target_file)

    speaker_column = speaker_getter.get_columns_with_indexes(0)
    print(f'Speaker column obtained from {speaker_file}')
    if clean_speaker:
        speaker_column = [speaker_id[:12] for speaker_id in speaker_column]
        print(f'Speaker column cleaned')
    targets_dict = target_getter.get_targets_with_names(target_meta, set(speaker_column))
    print(f'Target dictionary obtained from {target_file}')
    return targets_dict

class PostActivation:
    def __init__(self, alpha):
        self.alpha = alpha

    def read(self, file):
        speakers = []
        content = []
        with open(file, 'r') as f:
            f.readline() # skip header
            for line in f:
                speaker, string = line.strip().split(' ', 1)
                speakers.append(speaker)
                content.append(np.array(eval(string)))
        return speakers, np.array(content)

    def filter(self, activations, gradients):
        # Apply Leaky ReLU activation function
        filtered_activations = np.where(activations >= 0, activations, self.alpha*activations)
        filtered_gradients = np.where((gradients == 0) | (activations >= 0), gradients, gradients/self.alpha)
        return filtered_activations, filtered_gradients

    def save(self, file, speakers, content, header='SPEAKERID ACTIVATION'):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, 'w') as f:
            f.write(header + '\n')
            for speaker, item in zip(speakers, content):
                f.write(f"{speaker} {str(item.tolist())}\n")
        print(f"Saved {len(content)} items to {file}")
    