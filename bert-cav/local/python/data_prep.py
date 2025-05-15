'''
Prepares the data in tensor format
'''
import torch
import torch.nn as nn
from transformers import BertTokenizer

def get_spk_to_utt(responses_file, part):

    # Load the responses
    with open(responses_file, 'r') as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]

    # Concatenate utterances for a speaker
    spk_to_utt = {}
    for line in lines:
        speaker_part = int(line[14])
        if speaker_part != part:
            continue
        speakerid = line[:12]
        utt = line[20:]

        if speakerid not in spk_to_utt:
            spk_to_utt[speakerid] = utt
        else:
            spk_to_utt[speakerid] = spk_to_utt[speakerid] + ' ' + utt
    return spk_to_utt

def get_spk_to_grade(grades_file, part):

    # Load the grades
    with open(grades_file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    grade_dict = {}
    for line in lines:
        speaker_part = int(line[14])
        if speaker_part != part:
            continue
        speakerid = line[:12]
        grade = float(line.split()[1])
        grade_dict[speakerid] = grade
    return grade_dict

def align(spk_to_utt, grade_dict):
    grades = []
    utts = []
    speakerids = []
    for id in spk_to_utt:
        try:
            grades.append(grade_dict[id])
            utts.append(spk_to_utt[id])
            speakerids.append(id)
        except:
            print("Falied for speaker " + str(id))
    return utts, grades, speakerids

def tokenize_text(utts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    encoded_inputs = tokenizer(utts, padding=True, truncation=True, return_tensors="pt")
    ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    return ids, mask


def get_data(responses_file, grades_file, part=1):

    spk_to_utt = get_spk_to_utt(responses_file, part)
    grade_dict = get_spk_to_grade(grades_file, part)
    utts, grades, speaker_ids = align(spk_to_utt, grade_dict)
    input_ids, mask = tokenize_text(utts)
    labels = torch.FloatTensor(grades)
    return input_ids, mask, labels, speaker_ids

def get_mask_with_feature(feature_file, speaker_ids, mask, feature_size):
    feature_dict = {}
    with open(feature_file, 'r') as f:
        features = f.readlines()[1:]  # Read from line 2
    features = [line.strip().split() for line in features]
    for line in features:
        speakerid = line[0]
        feature = torch.tensor(list(map(float, line[1:])), dtype=torch.float)
        feature_dict[speakerid] = feature

    new_speaker_ids = []
    new_mask = []
    for speaker_id, mask_item in zip(speaker_ids, mask):
        if speaker_id not in feature_dict:
            print(f"Skipping {speaker_id}: speaker_id not found in feature_dict")
        elif len(feature_dict[speaker_id]) != feature_size:
            print(f"Skipping {speaker_id}: feature size mismatch")
        else:
            new_speaker_ids.append(speaker_id)
            concatenated_mask = torch.cat((mask_item, feature_dict[speaker_id]), dim=0)
            new_mask.append(concatenated_mask)

    print("Original mask size:", len(mask), len(mask[0]))
    print("New mask size:", len(new_mask), len(new_mask[0]))
    return new_speaker_ids, torch.stack(new_mask)