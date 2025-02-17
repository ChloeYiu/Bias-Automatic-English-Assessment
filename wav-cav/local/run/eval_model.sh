#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

export PATH="/scratches/dialfs/alta/oet/exp-swm35/housekeeping/envs/pyannote_w2v2/bin:$PATH"
condaenv=/scratches/dialfs/alta/oet/exp-swm35/housekeeping/envs/pyannote_w2v2/
source /scratches/dialfs/alta/oet/exp-swm35/housekeeping/etc/profile.d/conda.sh

conda activate "$condaenv"
echo "Activated environment: $(conda info --envs | grep '*' | awk '{print $1}')"

which python
python --version

if [ $# -lt 2 ]; then
  echo "Usage: $0 <trainset> <testset>"
  exit 1
fi

trainset=$1
testset=$2

declare -a seeds=(2 24)

for part in 1; do
  top_outdir=eval/$trainset/part$part

  for seed in "${seeds[@]}"; do
    prediction_file=$top_outdir/predictions/$testset/preds_wav2vec_part${part}_seed${seed}.txt
    output_base_name=$top_outdir/predictions/$testset/eval_wav2vec_part${part}_seed${seed}

    log_file="LOGs/$output_base_name.log"
    echo "Log file: $log_file"

    # Create log directory if it does not exist
    mkdir -p $(dirname "$log_file")
    if [ -f "$log_file" ]; then
        rm "$log_file"
    fi

    # Run the evaluation script with arguments from JSON file
    cmd="python local/python/eval_model.py --PREDICTION_FILE $prediction_file --OUTPUT_FILE $output_base_name.txt"
    echo $cmd
    $cmd >> $log_file 2>&1

  done
done