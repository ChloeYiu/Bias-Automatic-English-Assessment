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
  echo "Usage: $0 <train_set> <test_set>"
  exit 1
fi

train_set=$1
test_set=$2

LOG=LOGs/run/run_prediction.log
    mkdir -p $(dirname $LOG)
    if [ -f $LOG ]; then
        \rm $LOG
    fi

declare -a seeds=(10 30 50 70 90)

for part in 1; do
  for seed in "${seeds[@]}"; do
    
    top_outdir=eval/$train_set/part$part/

    cmd="python local/python/predict_with_hook.py --data_dir /scratches/dialfs/alta/sb2549/wav2vec2_exp/data_vectors_attention/$test_set/${test_set}_part${part}_att.hf --model_dir $top_outdir/models --GRADIENT_DIR $top_outdir/gradients/$test_set --ACTIVATION_DIR $top_outdir/activations/$test_set"

    echo $cmd
    $cmd >> $LOG 2>&1

  done
done