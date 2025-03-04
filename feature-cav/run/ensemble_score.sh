
#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

export PATH="/scratches/dialfs/alta/hkv21/mconda_setup/miniconda_py3.11/envs/pytorch2.0/bin:$PATH"
condaenv=/scratches/dialfs/alta/hkv21/mconda_setup/miniconda_py3.11/envs/pytorch2.0/
source activate "$condaenv"

which python
python --version

train_set="LIESTgrp06"
calib_set="LIESTcal01"
test_set="LIESTdev02"


if [ $# -lt 2 ]; then
  echo "Usage: $0 <train_data> <test_data <model>"
  exit 1
fi

train_data=$1
test_data=$2
model=$3

LOG=Logs/run/ensemble_score.log
    mkdir -p $(dirname $LOG)
    if [ -f $LOG ]; then
        \rm $LOG
    fi

declare -a seeds=(10 30 50)

for part in 1; do
    
    top_outdir=./${model}/ALTA/ASR_V2.0.0/${train_data}/f4-ppl-c2-pdf

    cmd="python local/training/${model}_Ensemble_scores.py --ensemble_dir $top_outdir/part${part} --dataname $test_data"

    echo $cmd
    $cmd >> $LOG 2>&1

done