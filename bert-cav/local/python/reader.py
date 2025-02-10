import torch
from models import BERTGrader

class TorchFileReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        try:
            data = torch.load(self.file_path)
            return data
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None

    def read_module(self, module_name):
        data = self.read()
        if module_name in data:
            return data[module_name]
        else:
            print(f"Module {module_name} not found in the file.")
            return None

    def load_model(self):
        model = BERTGrader()
        state_dict = self.read()
        model.load_state_dict(state_dict)
        return model
    
    def extract_layer_weights(self, layer_num):
        module_name = f'layer{layer_num}.weight'
        return self.read_module(module_name)

    
if __name__ == "__main__":
    # Example usage
    reader = TorchFileReader('/research/milsrg1/alta/linguaskill/exp-ymy23/bert-cav/est/LIESTgrp06/trained_models/part1/bert_part1_seed1.th')
    layer_weight = reader.extract_layer_weights(1)
    print(layer_weight.shape)