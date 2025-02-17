import math
import torch
import numpy as np
import argparse
import os
from transformers import AutoConfig, Wav2Vec2Processor
from datasets import load_from_disk
from train import Wav2Vec2ForSpeechClassification
from cav import ActivationGetter
from biased_score import load_biased_score

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Device name: {torch.cuda.get_device_name(1)}")

def calc_feat_size(audio_len):
    '''
    The Wav2Vec2 base model uses 25 ms frames and 20 ms steps.
    This code assumes that the audio sampling frequency is 16 kHz.
    '''
    assert type(audio_len) == int # check that function input is an integer
    if audio_len < 400:
        return "The length of the audio segment must be at least 25 ms or 400 samples."
    else:
        n_feat_vecs = 1 + math.floor(round((audio_len-400)/320, 5)) # the rounding is precautionary for quantization error

    return n_feat_vecs

def speech_file_to_array_fn(batch):
    arrays = []
    for speech_array in batch["input_values"]:
        arrays.append(speech_array)

    batch["speech"] = arrays
    return batch

def predict(batch, model, device, activation_getter, processor):
    all_feats = None
    list_len = [len(i) for i in batch["speech"]]
    pad_to = max(list_len)
    paded = [np.pad(test, (0, pad_to-len(test)), 'constant', constant_values=(0, 0)) for test in batch["speech"]]

    audio_lengths = []
    for test in batch["speech"]:
        for seg in test:
            audio_lengths.append(min(480000, len(seg))) # capping at 240,000 is probably superfluous, but is precautionary to ensure no audios longer than 15 s
    max_feat_vecs = calc_feat_size(max(audio_lengths))
    attention_mask = np.zeros((len(paded), pad_to, max_feat_vecs)) # setting attention_heads_mask shape as 4D tensor (# tests in batch, max # segs in test, max # feat_vecs in seg)
    for ind_test, test in enumerate(batch["speech"]):
        for ind_seg, seg in enumerate(test):
            attention_mask[ind_test, ind_seg, :calc_feat_size(len(batch["speech"][ind_test][ind_seg]))] = np.ones((calc_feat_size(len(batch["speech"][ind_test][ind_seg]))))

    paded = [[np.zeros(max(audio_lengths)) if type(aud) == np.int64 else aud for aud in test] for test in paded]
    paded = [[np.pad(aud, (0, max(audio_lengths)-len(aud))) if len(aud) < max(audio_lengths) else aud for aud in test] for test in paded]
    paded = [[aud[:480000] if len(aud)>480000 else aud for aud in feature] for feature in paded]

    for test in paded:
        features = torch.unsqueeze(processor(test, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)["input_values"],0)
        if all_feats == None:
            all_feats = features
        else:
            all_feats = torch.cat((all_feats, features), dim=0)
    input_values = all_feats.to(device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
    attention_heads_mask = attention_mask.to(device)
    
    logits = model(input_values, attention_heads_mask).logits 

    for layer_name in activation_getter.layer_names:
        activations = activation_getter.activation_cache[layer_name]
        # Compute gradients for this sample
        grad = torch.autograd.grad(
            outputs=logits,
            inputs=activations,
            grad_outputs=torch.ones_like(logits),
            create_graph=False,
            retain_graph=False,
            allow_unused=True
        )
        activation_getter.store_tmp_gradient(grad)

    batch["predicted"] = logits.detach().cpu().numpy()
    return batch


def main(args):
    test_data = args.DATA_DIR #'/scratches/dialfs/alta/sb2549/wav2vec2_exp/data_vectors_attention/LIESTcal01/LIESTcal01_part4_att.hf'
    model_dir = args.MODEL_DIR 
    activation_dir = args.ACTIVATION_DIR
    gradient_dir = args.GRADIENT_DIR
    output_file = args.OUTPUT_FILE
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config = AutoConfig.from_pretrained(model_dir)
    processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wav2vec2-base")
    model = Wav2Vec2ForSpeechClassification.from_pretrained(model_dir).to(device)

    model_part, model_seed = model_dir.split('/')[2], model_dir.split('/')[3]
    model_name = f"{model_part}_{model_seed}"
    activation_getter = ActivationGetter(model.classifier, model_name, activation_dir, gradient_dir, ['dense'], 1)
    test_dataset = load_from_disk(test_data)
    test_dataset = test_dataset.map(speech_file_to_array_fn)
    #test_dataset = biased_dataset(test_dataset, args.BIASED_SCORE) (not using scores anyways)

    activation_getter.add_hooks()
    test_ds = test_dataset.map(lambda batch: predict(batch, model, device, activation_getter, processor), batched=True, batch_size=1)
    activation_getter.remove_hooks()

    speaker_id, pred_score = test_ds['base_id'], test_ds['predicted']
    ref_score = test_ds['labels'] if args.BIASED_SCORE =='None' else load_biased_score(args.BIASED_SCORE)
    with open(output_file, 'w') as f:
        f.write('SPEAKERID PRED REF\n')
        for spkr, ref, pred in zip(speaker_id, ref_score, pred_score):
            f.write(f'{spkr} {pred[0]} {ref}\n')

    # Save the activations for each layer
    activation_getter.store_activations(speaker_id)
    activation_getter.store_gradients(speaker_id)

if __name__ == '__main__':
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--DATA_DIR', type=str, help='directory with test data')
    commandLineParser.add_argument('--MODEL_DIR', type=str, help='directory with model')
    commandLineParser.add_argument('--ACTIVATION_DIR', type=str, help='directory to store activations')
    commandLineParser.add_argument('--GRADIENT_DIR', type=str, help='directory to store gradients')
    commandLineParser.add_argument('--OUTPUT_FILE', type=str, help='file to save prediction')
    commandLineParser.add_argument('--BIASED_SCORE', type=str, help='profile to bias test data, if exists')
    args = commandLineParser.parse_args()
    main(args)
