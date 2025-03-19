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
  echo "Usage: $0 <train_data> <test_data> <model>"
  exit 1
fi

train_data=$1
test_data=$2
model=$3

LOG=Logs/run/evaluate.log
    mkdir -p $(dirname $LOG)
    if [ -f $LOG ]; then
        \rm $LOG
    fi

declare -a seeds=(10 30 50 70 90)

for part in 1; do
  for seed in "${seeds[@]}"; do
    
    top_outdir=./${model}/ALTA/ASR_V2.0.0/${train_data}

    cmd="python local/training/model_evaluate.py --data_dir ./data/ALTA/ASR_V2.0.0/$test_data/f4-ppl-c2-pdf/part$part --model_dir $top_outdir/f4-ppl-c2-pdf/part${part}/${model}_${seed} --GRADIENT_DIR $top_outdir/gradients/$test_data --ACTIVATION_DIR $top_outdir/activations/$test_data --model_type $model"

    echo $cmd
    $cmd >> $LOG 2>&1

  done
done