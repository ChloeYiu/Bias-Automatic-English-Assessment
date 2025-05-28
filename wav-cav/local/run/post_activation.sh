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
  echo "Usage: $0 <train_set> <test_set>"
  exit 1
fi

train_set=$1
test_set=$2

declare -a seeds=(2 24)

for part in 1; do
  for seed in "${seeds[@]}"; do
    
    top_outdir=eval/$train_set/part$part
    model_dir=models/$train_set/part$part/seed$seed/checkpoint-1968
   

    gradient_dir=$top_outdir/gradients/${test_set}
    activation_dir=$top_outdir/activations/${test_set}
    prediction_dir=$top_outdir/predictions/${test_set}
    output_file=$prediction_dir/post_activation_${part}_seed${seed}

    LOG=LOGs/$output_file.log
    mkdir -p $(dirname $LOG)
    if [ -f $LOG ]; then
        \rm $LOG
    fi

    echo "Logging to $LOG"

    cmd="python local/python/post_activation.py --MODEL_DIR $model_dir --GRADIENT_DIR $gradient_dir --ACTIVATION_DIR $activation_dir"
    echo $cmd
    $cmd >> $LOG 2>&1

  done
done