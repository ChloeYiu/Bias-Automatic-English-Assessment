'''
wav2vec2 with attention heads trainer
written by Arda C, with amendments by Simon McKnight and Kate Knill
'''
import torch
import sys
import os
import json
from biased_score import biased_dataset
os.environ["WANDB_DISABLED"] = "true"

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torchaudio
import datasets
from datasets import Dataset, DatasetDict
from datasets import load_from_disk
import librosa
from transformers import AutoConfig, Wav2Vec2Processor, EvalPrediction, TrainingArguments
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict, List, Union
from transformers.file_utils import ModelOutput
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
import math
from packaging import version
from transformers import (
    Trainer,
    is_apex_available,
)
if is_apex_available():
    from apex import amp
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


from path import makeDir, checkDirExists, checkFileExists, makeCmdPath

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(4*config.hidden_size, 4*config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(4*config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.attn1 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.attn2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.attn3 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.attn4 = torch.nn.Linear(config.hidden_size, config.hidden_size)

        self.attn5 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.attn6 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.attn7 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.attn8 = torch.nn.Linear(config.hidden_size, config.hidden_size)

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def _apply_attn(self, embeddings, weights_transformation, level_attn_heads_mask=None):
        '''
        Self-attention variant to get sentence embedding
        '''
        a = weights_transformation(embeddings)
        a = torch.einsum('ijk,ijk->ij', embeddings, a)
        T = nn.Tanh()
        a = T(a)

        # Mask provisions if applicable
        if level_attn_heads_mask != None:
            #print("Applying attention head, mask shape {}".format(level_attn_heads_mask.shape))
            a = a * level_attn_heads_mask # this is still score_T in Vyas' code.  For the first layer of attention heads, just need the [S, V] parts of [B, S, V, F]
            mask_complement = 1 - level_attn_heads_mask if level_attn_heads_mask.dtype != torch.bool else ~level_attn_heads_mask # the 1 - tensor does not work for boolean
            inf_mask = mask_complement * (-10000)
            a = a + inf_mask # this is scaled_score in Vyas' code

        # Normalize with softmax
        SM = nn.Softmax(dim=1)
        a = SM(a)
        a = torch.unsqueeze(a, -1).expand(-1,-1, embeddings.size(-1))
        a = torch.sum(embeddings*a, dim=1)
        return a

    def forward(
            self,
            input_values,
            attention_heads_mask, # this only applies to the attention heads, not the Wav2Vec2 model
            attention_mask=None, # this was the original version, and remains None as we are not applying attention_mask at Wav2Vec2 level
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        wav_outputs1 = None
        wav_outputs2 = None
        wav_outputs3 = None
        wav_outputs4 = None
        second_layer_attn_heads_mask = attention_heads_mask[:, :, 0] # this gives a 2D vector for the exam [B, S] using fact that entire row would be 1 or 0
        for index, test in enumerate(input_values): # the index is needed for the attention head
            outputs = self.wav2vec2(
                test,
                attention_mask=attention_mask, # this is None and is not used
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = outputs[0] # note we have a matrix each time here, not a vector
            layer_attn_heads_mask = attention_heads_mask[index] # this gives a 2D matrix for the exam [S, V]
            if wav_outputs1 == None:
                wav_outputs1 = torch.unsqueeze(self._apply_attn(hidden_states, self.attn1, layer_attn_heads_mask),0)
                #print("wav_outputs1 shape is {} and wav_outputs1.is_cuda is {}".format(wav_outputs1.size(), wav_outputs1.is_cuda))
                wav_outputs2 = torch.unsqueeze(self._apply_attn(hidden_states, self.attn2, layer_attn_heads_mask),0)
                #print("wav_outputs2 shape is {} and wav_outputs2.is_cuda is {}".format(wav_outputs2.size(), wav_outputs2.is_cuda))
                wav_outputs3 = torch.unsqueeze(self._apply_attn(hidden_states, self.attn3, layer_attn_heads_mask),0)
                #print("wav_outputs3 shape is {} and wav_outputs3.is_cuda is {}".format(wav_outputs3.size(), wav_outputs3.is_cuda))
                wav_outputs4 = torch.unsqueeze(self._apply_attn(hidden_states, self.attn4, layer_attn_heads_mask),0)
                #print("wav_outputs4 shape is {} and wav_outputs4.is_cuda is {}".format(wav_outputs4.size(), wav_outputs4.is_cuda))
            else:
                wav_outputs1 = torch.cat((wav_outputs1, torch.unsqueeze(self._apply_attn(hidden_states, self.attn1, layer_attn_heads_mask),0)), dim=0)
                #print("wav_outputs1 shape is {} and wav_outputs1.is_cuda is {}".format(wav_outputs1.size(), wav_outputs1.is_cuda))
                wav_outputs2 = torch.cat((wav_outputs2, torch.unsqueeze(self._apply_attn(hidden_states, self.attn2, layer_attn_heads_mask),0)), dim=0)
                #print("wav_outputs2 shape is {} and wav_outputs2.is_cuda is {}".format(wav_outputs2.size(), wav_outputs2.is_cuda))
                wav_outputs3 = torch.cat((wav_outputs3, torch.unsqueeze(self._apply_attn(hidden_states, self.attn3, layer_attn_heads_mask),0)), dim=0)
                #print("wav_outputs3 shape is {} and wav_outputs3.is_cuda is {}".format(wav_outputs3.size(), wav_outputs3.is_cuda))
                wav_outputs4 = torch.cat((wav_outputs4, torch.unsqueeze(self._apply_attn(hidden_states, self.attn4, layer_attn_heads_mask),0)), dim=0)
                #print("wav_outputs4 shape is {} and wav_outputs4.is_cuda is {}".format(wav_outputs4.size(), wav_outputs4.is_cuda))

        head5 = self._apply_attn(wav_outputs1, self.attn5, second_layer_attn_heads_mask)
        #print("head5 shape is {}".format(head5.size()))
        head6 = self._apply_attn(wav_outputs2, self.attn6, second_layer_attn_heads_mask)
        #print("head6 shape is {}".format(head6.size()))
        head7 = self._apply_attn(wav_outputs3, self.attn7, second_layer_attn_heads_mask)
        #print("head7 shape is {}".format(head7.size()))
        head8 = self._apply_attn(wav_outputs4, self.attn8, second_layer_attn_heads_mask)
        #print("head8 shape is {}".format(head8.size()))

        complete_heads = torch.cat((head5, head6, head7, head8), dim=1)
        #print("complete_heads shape is {}".format(complete_heads.size()))

        logits = self.classifier(complete_heads)
        #print("logits shape is {}".format(logits.size()))

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        cap_audio_len (:obj:`int`, `optional`):
            Maximum length of the audio sequence to pad to
    """
    processor: Wav2Vec2Processor
    cap_audio_len: 480000
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    max_segs = 0
    max_tot = 0

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.float

        tensor_ready = [feature["input_values"] for feature in input_features] # Size [B, S, 240000] or [B, S, 480000]
        list_len = [len(i) for i in tensor_ready] # these are the number of segments in each exam in the input batch, e.g. [24, 37, 31, 37]
        pad_to = max(list_len)
        paded = [np.pad(feature, (0, pad_to-len(feature)), 'constant', constant_values=(0, 0)) for feature in tensor_ready] # this pads all up to max number of segments

        audio_lengths = []
        for test in tensor_ready:
            for seg in test:
                #audio_lengths.append(min(240000, len(seg))) # capping at 240,000 is probably superfluous, but is precautionary to ensure no audios longer than 15 s
                #audio_lengths.append(min(480000, len(seg))) # capping at 240,000 is probably superfluous, but is precautionary to ensure no audios longer than 15 s
                audio_lengths.append(min(self.cap_audio_len, len(seg))) # capping at 240,000 is probably superfluous, but is precautionary to ensure no audios longer than 15 s
        max_feat_vecs = calc_feat_size(max(audio_lengths))

        # On reflection, the final feature dimension is not needed as they come after the einsum reduction in _apply_attn(), so we have a 3D tensor
        # setting attention_heads_mask shape as 4D tensor (# tests in batch, max # segs in test, max # feat_vecs in seg, # feats)
        attention_heads_mask = np.zeros((len(paded), pad_to, max_feat_vecs)) # setting attention_heads_mask shape as 4D tensor (# tests in batch, max # segs in test, max # feat_vecs in seg)
        print("attention_heads_mask shape is {}".format(attention_heads_mask.shape))
        # now set attention_heads_mask values to 1 where there are actual observations
        for ind_test, test in enumerate(tensor_ready):
            for ind_seg, seg in enumerate(test):
                attention_heads_mask[ind_test, ind_seg, :calc_feat_size(len(tensor_ready[ind_test][ind_seg]))] = np.ones((calc_feat_size(len(tensor_ready[ind_test][ind_seg]))))

        paded = [[np.zeros(max(audio_lengths)) if type(aud) == np.int64 else aud for aud in test] for test in paded]
        paded = [[np.pad(aud, (0, max(audio_lengths)-len(aud))) if len(aud) < max(audio_lengths) else aud for aud in test] for test in paded]
        #paded = [[aud[:240000] if len(aud)>240000 else aud for aud in feature] for feature in paded] # this pads each segment up to 15 seconds
        #paded = [[aud[:480000] if len(aud)>480000 else aud for aud in feature] for feature in paded] # this pads each segment up to 30 seconds
        paded = [[aud[:self.cap_audio_len] if len(aud)>self.cap_audio_len else aud for aud in feature] for feature in paded] # this pads each segment up to 30 seconds
        tensored = torch.tensor(np.array(paded), dtype=torch.half)
        batch = {"input_values": tensored}
        batch["attention_heads_mask"] = torch.tensor(attention_heads_mask, dtype=torch.bool)
        batch["labels"] = torch.tensor(label_features, dtype=d_type).unsqueeze(1)

        tot = 0
        for seg in batch["input_values"]:
            tot += len(seg)
            if len(seg) > self.max_segs:
                self.max_segs = len(seg)
        if tot > self.max_tot:
            self.max_tot = tot

        return batch

def compute_metrics(p: EvalPrediction, is_regression: bool):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        # AC: seem to have problems with use_cuda_amp now - downgraded to transformers v29
        if self.use_cuda_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
                loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_cuda_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()
        return loss.detach()

def main ( args ):
    #------------------------------------------------------------------------------
    # read in command line arguments
    #------------------------------------------------------------------------------
    train_dataset_dir = args.train_dataset
    val_dataset_dir = args.val_dataset
    out_dir = args.output_dir
    train_batch_size=args.train_batch
    val_batch_size=args.val_batch
    num_epochs=args.epochs
    cap_audio_len=args.cap_len

    train_from_scratch=True
    if args.checkpoint is not None:
        train_from_scratch is False
        start_model = args.checkpoint
        checkDirExists(start_model)

    number_labels = args.num_labels
    is_regression = True

    checkDirExists(train_dataset_dir)
    checkDirExists(val_dataset_dir)
    makeDir(out_dir, False)

    #------------------------------------------------------------------------------
    # save command line arguments to file
    #------------------------------------------------------------------------------
    makeCmdPath(out_dir)

    #------------------------------------------------------------------------------
    # set up model configuration and load data
    #------------------------------------------------------------------------------
    torch.manual_seed(args.seed)
    model_name_or_path = 'patrickvonplaten/wav2vec2-base'
    pooling_mode = "mean"

    # config
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=number_labels,
        finetuning_task="wav2vec2_clf",
        problem_type="regression",
    )
    setattr(config, 'pooling_mode', pooling_mode)

    config.problem_type
    config.final_dropout = args.final_dropout
    config.hidden_dropout = args.hidden_dropout

    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)
    target_sampling_rate = processor.feature_extractor.sampling_rate
    print(f"The target sampling rate: {target_sampling_rate}")

    train_dataset = load_from_disk(train_dataset_dir)
    print(len(train_dataset))
    train_dataset = biased_dataset(train_dataset, args.biased_score)
    val_dataset = load_from_disk(val_dataset_dir)

    #MODEL
    print("CUDA: {}".format(torch.cuda.current_device()))

    # the padding is done manually now, so removing argument (not sure if padding
    # defaults to True though, specifying False crashed)
    print("cap_audio_len is %s" % str(cap_audio_len))    
    data_collator = DataCollatorCTCWithPadding(processor=processor, cap_audio_len=cap_audio_len)
    print("data_collator type is {}".format(type(data_collator)))

    model = Wav2Vec2ForSpeechClassification.from_pretrained(
        model_name_or_path,
        config=config,
    )

    print("model config is {}".format(print(model.config)))
    print("type(model) is {}".format(type(model)))
    model.freeze_feature_extractor()
    print("The model has {:,} parameters.".format(sum([p.numel() for p in model.parameters()])))
    print("The model has {:,} trainable parameters.".format(sum([p.numel() for p in model.parameters() if p.requires_grad])))

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=train_batch_size,  #4 for OET, 32 for Linguaskill
        per_device_eval_batch_size=val_batch_size, #4 for OET, 32 for Linguaskill
        gradient_accumulation_steps=2, # Arda used 8, Stefano 2
        evaluation_strategy="epoch",
        num_train_epochs=num_epochs, #Use 3 for initial Linguaskill, 40 for OET after Linguaskill
        fp16=True,
        save_strategy='epoch',
        logging_steps=10,
        learning_rate=1e-6, # Arda used 5e-5, Stefano 1e-6
        eval_accumulation_steps=5,
        lr_scheduler_type="constant", # used by Stefano, not used by Arda
        #report_to="wandb"
        #prediction_loss_only=True
    )

    trainer = CTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=lambda p: compute_metrics(p, is_regression),
        train_dataset=train_dataset, # AC: slicing here is just as slow as before
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
    )

    trainer.args._n_gpu = 1

    if train_from_scratch:
        trainer.train()
    else:
        trainer.train(start_model)


if __name__ == '__main__':
    #------------------------------------------------------------------------------
    # arguments
    #------------------------------------------------------------------------------
    parser = argparse.ArgumentParser (description = 'Train wav2vec2-based grader with attention')
    parser.add_argument ('train_dataset',metavar = 'train_dataset', type = str,
                         help = 'Training dataset directory (e.g. /scratches/dialfs/alta/oet/grd-ac2287/y4-project/data_vectors/split_segments_attn_nopad/OETCBtrn01.hf/, /scratches/dialfs/alta/sb2549/wav2vec2_exp/data_vectors_attention/LIESTgrp06/LIESTgrp06_part2_att.hf/')
    parser.add_argument ('val_dataset',metavar = 'val_dataset', type = str,
                                help = 'Validation dataset directory (e.g. /scratches/dialfs/alta/oet/grd-ac2287/y4-project/data_vectors/split_segments_attn_nopad/OETCBdev01.hf/, /scratches/dialfs/alta/sb2549/wav2vec2_exp/data_vectors_attention/LIESTcal01/LIESTcal01_part2_att.hf/')
    parser.add_argument ('output_dir',metavar = 'output_dir', type = str,help = 'Output directory (e.g. models/linguaskill/model_03)')
    parser.add_argument ('--num_labels',metavar = 'num_labels', type = int, default=1,
                                help = 'Number of labels (default 1)')
    parser.add_argument ('--train_batch',metavar = 'train_batch', type = int, default=32,
                                help = 'Train batch size (default 32) (use 4 for OET)')
    parser.add_argument ('--val_batch',metavar = 'val_batch', type = int, default=32,
                                help = 'Validation batch size (default 32) (use 4 for OET)')
    parser.add_argument ('--epochs',metavar = 'epochs', type = int, default=4,
                                help = 'Number of training epochs use 3 for initial Linguaskill, 40 for OET (default 3)')
    parser.add_argument ('--final_dropout',metavar = 'final_dropout', type = float, default=0.1,
                                help = 'Final layer dropout (default 0.1)')
    parser.add_argument ('--hidden_dropout',metavar = 'hidden_dropout', type = float, default=0.1,
                                help = 'Hidden layer dropout (default 0.1)')
    parser.add_argument ('--checkpoint', type = str, 
                                help = 'Model checkpoint to start training from (default none)')
    parser.add_argument('--seed', type=int, help='Set the random number generator seed (default 42)', default=42)
    parser.add_argument('--biased_score', type=str, help='file with biased score')
    parser.add_argument('--cap_len', type=int, help='Set the cap of audio length in samples (default 480000, equivalent to 30s) (use 240000 for OET/AC files)', default=480000)
    args = parser.parse_args()
    print("Arguments parsed")
    main(args)



