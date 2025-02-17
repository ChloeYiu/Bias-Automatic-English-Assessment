#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

export PATH="/scratches/dialfs/alta/oet/exp-swm35/housekeeping/envs/pyannote_w2v2/bin:$PATH"
condaenv=/scratches/dialfs/alta/oet/exp-swm35/housekeeping/envs/pyannote_w2v2/
source /scratches/dialfs/alta/oet/exp-swm35/housekeeping/etc/profile.d/conda.sh

conda activate "$condaenv"
echo "Activated environment: $(conda info --envs | grep '*' | awk '{print $1}')"

if [ $# -lt 5 ]; then
  echo "Usage: $0 <train_set> <dev_set> <part> <grader_seed> <profile>"
  exit 1
fi

train_set=$1
dev_set=$2
part=$3
grader_seed=$4
profile=$5


train_data=data_vectors_attention/$train_set/${train_set}_part${part}_att.hf
dev_data=data_vectors_attention/$dev_set/${dev_set}_part${part}_att.hf
if [ "$profile" = "None" ]; then
  output_dir=models/$train_set/part$part/seed$grader_seed
  biased_score=None
else
  output_dir=models/${train_set}_${profile}/part$part/seed$grader_seed
  biased_score=models/${train_set}_${profile}/part$part/scores_biased.json
fi
LOG=LOGs/$output_dir/run_train.log
echo "Logging to $LOG"
mkdir -p $(dirname $LOG)
if [ -f $LOG ]; then
    \rm $LOG
fi

cmd="python local/python/train.py $train_data $dev_data $output_dir --seed $grader_seed --biased_score $biased_score"
echo $cmd
$cmd >& $LOG
echo "Done with $cmd"