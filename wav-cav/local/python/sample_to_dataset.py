#!/usr/bin/env python
# coding: utf-8

# Prepare dataset files
# Modified from /scratches/dialfs/alta/oet/grd-ac2287/y4-project/oet-sample_and_save_attnp.py.


import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split

import os
import sys

from datasets import Dataset, DatasetDict
from datasets import disable_caching
disable_caching()

# Loading the created dataset using datasets
from datasets import load_dataset, load_metric

from path import makeDir, checkDirExists, checkFileExists
from cmdlog import makeCmd

commandLineParser = argparse.ArgumentParser (description = 'Sample audio to dataset')
commandLineParser.add_argument ('--linguaskill', dest='linguaskill', default=False, action='store_true', help = 'Linguaskill format data')
commandLineParser.add_argument ('--linguaskill_score', dest='linguaskill_score', default=False, action='store_true', help = 'Source data in Linguaskill score range')
commandLineParser.add_argument ('--whisper_model',metavar = 'whisper_model', type = str, default='small.en',help = 'Whisper modelname for transcription (default: small.en)')
commandLineParser.add_argument ('input_file',metavar = 'input_file', type = str,
                                help = 'CSV data description file (e.g. /scratches/dialfs/alta/oet/grd-bd432/data_paths/oet_segments_25s/OETCBdev08_data.csv)')
commandLineParser.add_argument ('output_file',metavar = 'output_file', type = str,
                                help = 'Output file (e.g. /data/milsrg1/alta/oet/data_vectors/whisper_transcriptions_25s/OETCBdev08.hf)')
args = commandLineParser.parse_args()


data_files = {
#    "OET6": "csvs/OETCBevl06/split_segments.csv"
    "OET7": "csvs/OETCBevl07/split_segments.csv"
#    "train": "data/split_segments_short/OETCBtrn01_data.csv",
#    "validation": "data/split_segments_short/OETCBdev01_data.csv",
#    "test": "data/split_segments_short/OETCBevl01_data.csv",
}

dataset = load_dataset("csv", data_files=data_files, delimiter=",")
#eval_dataset = dataset["OET6"]
eval_dataset = dataset["OET7"]
#train_dataset = dataset["train"]
#eval_dataset = dataset["validation"]
#test_dataset = dataset["test"]


#test_pd = pd.DataFrame(test_dataset)
#test_scores = test_pd.groupby("SpeakerID")["score"].mean().reset_index().set_index("SpeakerID")
#test_paths = test_pd.groupby("SpeakerID")["path"].apply(list).reset_index().set_index("SpeakerID")
#test_frames = [test_scores, test_paths]
#test_result = pd.concat(test_frames, axis = 1, join="inner").reset_index().rename(columns={"path": "paths"})
#test_dataset = Dataset.from_pandas(test_result)

eval_pd = pd.DataFrame(eval_dataset)
eval_scores = eval_pd.groupby("SpeakerID")["score"].mean().reset_index().set_index("SpeakerID")
eval_paths = eval_pd.groupby("SpeakerID")["path"].apply(list).reset_index().set_index("SpeakerID")
eval_frames = [eval_scores, eval_paths]
eval_result = pd.concat(eval_frames, axis = 1, join="inner").reset_index().rename(columns={"path": "paths"})
eval_dataset = Dataset.from_pandas(eval_result)

#train_pd = pd.DataFrame(train_dataset)
#train_scores = train_pd.groupby("SpeakerID")["score"].mean().reset_index().set_index("SpeakerID")
#train_paths = train_pd.groupby("SpeakerID")["path"].apply(list).reset_index().set_index("SpeakerID")
#train_frames = [train_scores, train_paths]
#train_result = pd.concat(train_frames, axis = 1, join="inner").reset_index().rename(columns={"path": "paths"})
#train_dataset = Dataset.from_pandas(train_result)

#print(train_dataset)
print(eval_dataset)
#print(test_dataset)


# We need to specify the input and output column
input_column = "paths"
output_column = "score"



num_labels = 1
#label_list = list(range(num_labels))
#print(f"A regression problem with {num_labels} items: {label_list}")
is_regression = True



from transformers import AutoConfig, Wav2Vec2Processor



model_name_or_path = 'patrickvonplaten/wav2vec2-base'
pooling_mode = "mean"


# config
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=1,
    #label2id={label: i for i, label in enumerate(label_list)},
    #id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
    problem_type="regression"
)
setattr(config, 'pooling_mode', pooling_mode)


# In[14]:


config.problem_type







processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
target_sampling_rate = processor.feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")




def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    #speech_array[0][0:speech_array[0].shape[0]//3]
    #resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = speech_array.squeeze().numpy()
    return speech

def preprocess_function(examples):
    speech_list = [[speech_file_to_array_fn(path) for path in paths] for paths in examples[input_column]]
    target_list = [label for label in examples[output_column]] # Do any preprocessing on your float/integer data

    #breakpoint()

    #result = processor(speech_list, sampling_rate=target_sampling_rate)
    processed = [processor(i, sampling_rate=target_sampling_rate)["input_values"] for i in speech_list]
    truncated = [[sp[:240000] if len(sp)>240000 else sp for sp in tests] for tests in processed]
    #padded = [[np.pad(sp, (0, 240000-len(sp)), 'constant') for sp in tests] for tests in truncated]
    result = {'input_values': truncated}

    #breakpoint()

    #May adjust below to improve spread
    #labels_list = [score/4.0 for score in list(target_list)] # Convert scores to be aligned with Stefano's
    labels_list = [(6.0*score)/39.0 for score in list(target_list)] # Convert scores to be aligned with Stefano's
    #labels_list = [6.0*(score-19.0)/17.0 for score in list(target_list)] # Convert scores to be aligned with Stefano's, alternative
    result["labels"] = list(labels_list)

    #breakpoint()

    #print(examples["SpeakerID"])

    return result




print('Preprocessing...')

#train_dataset = train_dataset.map(
#    preprocess_function,
#    batch_size=400,
#    num_proc=8,
#    batched=True,
#)

eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

#test_dataset = test_dataset.map(
#    preprocess_function,
#    batch_size=1000,
#    batched=True,
#    num_proc=4
#)




#del eval_dataset
#del test_dataset



print('Saving to disk...')


#eval_dataset.save_to_disk('data_vectors/OETCBevl06/OETCBevl06.hf')
eval_dataset.save_to_disk('data_vectors/OETCBevl07/OETCBevl07.hf')
#train_dataset.save_to_disk('data_vectors/split_segments_short_attn_nopad/OETCBtrn01.hf')
#eval_dataset.save_to_disk('data_vectors/split_segments_short_attn_nopad/OETCBdev01.hf')
#test_dataset.save_to_disk('data_vectors/split_segments_short_attn_nopad/OETCBevl01.hf')


print('DONEEEEEEEEEEEEEEEEEEEEEEEEEE')



# %%
