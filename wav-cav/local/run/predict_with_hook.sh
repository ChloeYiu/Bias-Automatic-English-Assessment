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

while [ $# -gt 0 ]; do
    key=$1
    case $key in
        --biased_train)
        cmdopts="$cmdopts $1 $2"
        biased_train=$2
        shift
        shift
        ;;
        --biased_test)
        cmdopts="$cmdopts $1 $2"
        biased_test=$2
        shift
        shift
        ;;
    *)
    POSITIONAL+=("$1")
    shift
    ;;
    esac
done
set -- "${POSITIONAL[@]}"

if [ $# -lt 2 ]; then
  echo "Usage: $0 <train_set> <test_set> --biased_train <biased_train> --biased_test <biased_test>"
  exit 1
fi

train_set=$1
test_set=$2

declare -a seeds=(24)

for part in 1; do
  for seed in "${seeds[@]}"; do
    
    if [ -z "$biased_train" ]; then
      top_outdir=eval/$train_set/part$part
      model_dir=models/$train_set/part$part/seed$seed/checkpoint-1968
    else
      top_outdir=eval/${train_set}_${biased_train}/part$part
      model_dir=models/${train_set}_${biased_train}/part$part/seed$seed/checkpoint-1968
    fi

    if [ -z "$biased_test" ]; then
      biased_score=None
      gradient_dir=$top_outdir/gradients/$test_set
      activation_dir=$top_outdir/activations/$test_set
      prediction_dir=$top_outdir/predictions/$test_set
      output_file=$prediction_dir/preds_wav2vec_part${part}_seed${seed}

    else
      biased_score=models/${test_set}_${biased_test}/part$part/scores_biased.json
      gradient_dir=$top_outdir/gradients/${test_set}_${biased_test}
      activation_dir=$top_outdir/activations/${test_set}_${biased_test}
      prediction_dir=$top_outdir/predictions/${test_set}_${biased_test}
      output_file=$prediction_dir/preds_wav2vec_part${part}_seed${seed}
    fi

    LOG=LOGs/$output_file.log
    mkdir -p $(dirname $LOG)
    if [ -f $LOG ]; then
        \rm $LOG
    fi

    echo "Logging to $LOG"

    cmd="python local/python/predict_with_hook.py --DATA_DIR data_vectors_attention/$test_set/${test_set}_part${part}_att.hf --MODEL_DIR $model_dir --GRADIENT_DIR $gradient_dir --ACTIVATION_DIR $activation_dir --PREDICTION_DIR $prediction_dir --OUTPUT_FILE $output_file.txt --BIASED_SCORE $biased_score"

    echo $cmd
    $cmd >> $LOG 2>&1

  done
done