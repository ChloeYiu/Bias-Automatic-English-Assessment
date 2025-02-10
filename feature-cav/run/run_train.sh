#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

export PATH="/scratches/dialfs/alta/hkv21/mconda_setup/miniconda_py3.11/envs/pytorch2.0/bin:$PATH"
condaenv=/scratches/dialfs/alta/hkv21/mconda_setup/miniconda_py3.11/envs/pytorch2.0/
source activate "$condaenv"

which python
python --version

if [ $# -lt 4 ]; then
  echo "Usage: $0 <train_data> <dev_data> <grader_seed> <input_size>"
  exit 1
fi

train_data=$1
dev_data=$2
grader_seed=$3
input_size=$4

python ./local/training/DDN_Trainers.py --train_data $train_data --dev_data $dev_data --grader_seed $grader_seed --input_size $input_size