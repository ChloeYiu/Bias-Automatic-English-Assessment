import json

def load_biased_score(biased_score_file):
    with open(biased_score_file, 'r') as f:
        biased_score = json.load(f)
    return biased_score

def biased_dataset(dataset, biased_score_file):
    if biased_score_file != 'None':
        biased_score = load_biased_score(biased_score_file)
        dataset = dataset.map(lambda example, idx: {"label": biased_score[idx]}, with_indices=True)
    return dataset
