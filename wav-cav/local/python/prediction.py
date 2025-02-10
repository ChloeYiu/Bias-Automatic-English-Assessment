import math
from transformers import AutoConfig, Wav2Vec2Processor

config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wav2vec2-base")
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)


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
        #speech_array, sampling_rate = torchaudio.load(path)
        #speech_array = speech_array.squeeze().numpy()
        #speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, processor.feature_extractor.sampling_rate)
        arrays.append(speech_array)

    batch["speech"] = arrays
    return batch


def predict(batch):
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
        #features = processor(batch["input_values"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    input_values = all_feats.to(device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
    attention_heads_mask = attention_mask.to(device)
    #attention_mask = features.attention_mask.to(device)
    with torch.no_grad(): # TODO: track grad
        #logits = model(input_values, attention_mask=attention_mask).logits 
        logits = model(input_values, attention_heads_mask).logits 

    batch["predicted"] = logits.detach().cpu().numpy()
    #batch = logits.detach().cpu().numpy()
    return batch
