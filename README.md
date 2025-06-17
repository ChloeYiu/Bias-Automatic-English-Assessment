# Bias and Fairness in Automatic Spoken Language Assessment

## Introduction
This research focuses on investigating the use of Concept Activation Vector (CAV) to measure biases within the feature-based, text-based, and audio-based model

## Feature-based Model
### Unbiased Model
To train an unbiased model, run: 

```bash
feature-cav/run/train.sh LIESTgrp06 LIESTdev02 $model
```

To evaluate the model, run:
```bash
feature-cav/batch/batch_posttrain_unbiased.sh $model
```

where `$model` specifies the type of model to run (e.g. `DDN`, `DNN`, `DDN_ReLU`, `DDN_LBERT`)


### Biased Model
To train a biased DDN model, run:
```bash
feature-cav/batch/batch_pretrain.sh $profile
```

To evaluate the model, run:
```bash
feature-cav/batch/batch_posttrain_profile.sh $profile
```

where `$profile` is the concept to bias towards (e.g. `male`, `young`, `thai`, `spanish`, `grade_C`, `grade_B2`, `grade_A`)

## Text-based Model

### Unbiased Model
To train an unbiased model, run:
```bash
bert-cav/local/run/run_train_grader.sh --part_range 1:1 --seed_range 1:5 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est 
```

To evaluate the model, run:
```bash
bert-cav/batch/batch_posttrain.sh
```

### Unbiased Deep Fusion Model
To train an unbiased model with deep fusion of feature vector, run:
```bash
bert-cav/local/run/run_train_grader.sh --part_range 1:1 --seed_range 1:5 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est --feature /research/milsrg1/alta/linguaskill/exp-ymy23/feature-cav/data/ALTA/ASR_V2.0.0/LIESTgrp06/f4-ppl-c2-pdf
```

To evalaute the model, run: 
```bash
bert-cav/batch/batch_posttrain_feature.sh
```
### Unbiased LReLU Model
To train an unbiased model using Leaky ReLU activation function, run:
```bash
bert-cav/local/run/run_train_grader.sh --part_range 1:1 --seed_range 1:5 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est --lrelu
```

To evalaute the model, run: 
```bash
bert-cav/batch/batch_posttrain_lrelu.sh
```

### Biased Model
To train a biased model, run:
```bash
bert-cav/batch/batch_pretrain_profile.sh $profile
```

To evaluate the model, run:
```bash
bert-cav/batch/batch_posttrain_profile.sh $profile
```

where `$profile` is the concept to bias towards (e.g. `male`, `young`, `thai`, `spanish`, `grade_C`, `grade_B2`, `grade_A`)

## Audio-based Model
The seed to run for this model for now not a command-line input, but hardcoded in the bash script. Change the scripts to choose the seed to evaluate.

### Unbiased Model
To train an unbiased model, run:
```bash
wav-cav/local/run/train.sh LIESTgrp06 LIESTdev02
```

To evalaute the model, run: 
```bash
wav-cav/batch/batch_posttrain.sh
```

### Biased Model
To train a biased model, run:
```bash
wav-cav/local/run/train.sh --biased_profile $profile
```

To evaluate the model, run:
```bash
bert-cav/batch/batch_posttrain_profile.sh $profile
```

where `$profile` is the concept to bias towards (e.g. `male`, `young`, `thai`, `spanish`, `grade_C`, `grade_B2`, `grade_A`)
