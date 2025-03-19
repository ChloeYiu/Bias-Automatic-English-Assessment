#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

export PATH="/scratches/dialfs/alta/hkv21/mconda_setup/miniconda_py3.11/envs/pytorch2.0/bin:$PATH"
condaenv=/scratches/dialfs/alta/hkv21/mconda_setup/miniconda_py3.11/envs/pytorch2.0/
source activate "$condaenv"

which python
python --version

if [ $# -lt 2 ]; then
  echo "Usage: $0 <train_data> <test_data> <calib_data> <model>"
  exit 1
fi

train_data=$1
test_data=$2
calib_data=$3
model=$4

declare -a seeds=(10 30 50 70 90)

for part in 1; do
  for seed in "${seeds[@]}"; do
    if [ "$model" == "DDN" ]; then
      pred_file_suffix="_pred_ref.txt"
    elif [ "$model" == "DNN" ]; then
      pred_file_suffix="_pred.txt"
    else
      echo "Unknown model: $model"
      exit 1
    fi

    python local/training/calibrate.py \
      --pred_file ./${model}/ALTA/ASR_V2.0.0/${train_data}/f4-ppl-c2-pdf/part${part}/${model}_${seed}/${calib_data}/${calib_data}${pred_file_suffix} --model_type $model
  done
  python local/training/calibrate.py \
    --pred_file ./${model}/ALTA/ASR_V2.0.0/${train_data}/f4-ppl-c2-pdf/part${part}/ens_${test_data}/${test_data}${pred_file_suffix} --model_type $model
done