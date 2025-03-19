#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

HTE=lib/htesystem/HTE-volta.system # Default env file for sungrid queue

LOG=Logs/data/ALTA/ASR_V2.0.0/train.log
mkdir -p $(dirname $LOG)
if [ -f $LOG ]; then
    \rm $LOG
fi

source $HTE
if [ -z $QSUBPROJECT ]; then
    QSUBPROJECT=esol
fi

QSUBOPTS=""
if [ ! -z $QSUBQUEUE ]; then
    QSUBOPTS="$QSUBOPTS -l qp=$QSUBQUEUE"
fi
if [ ! -z $QGPUCLASS ]; then
  QSUBOPTS="$QSUBOPTS -l gpuclass=$QGPUCLASS"
fi
if [ ! -z $QHOSTNAME ]; then
    QSUBOPTS="$QSUBOPTS -l hostname=$QHOSTNAME"
fi
if [ ! -z $QCUDAMEM ]; then
    QSUBOPTS="$QSUBOPTS -l cudamem=$QCUDAMEM"
fi
if [ ! -z $QMAXJOBS ]; then
    QSUBOPTS="$QSUBOPTS -tc $QMAXJOBS"
fi

bin=./run/run_train.sh
waitid=""

declare -a seeds=(10 30 50 70 90)

if [ $# -lt 3 ]; then
  echo "Usage: $0 <train_data> <dev_data> <model>"
  exit 1
fi

train_set=$1
dev_set=$2
model=$3

for part in 1; do
  for seed in "${seeds[@]}"; do
    echo "Part $part, Seed $seed"
    train_data=./data/ALTA/ASR_V2.0.0/$train_set/f4-ppl-c2-pdf/part${part}
    dev_data=./data/ALTA/ASR_V2.0.0/$dev_set/f4-ppl-c2-pdf/part${part}
    input_size=356
    cmdopts="$train_data $dev_data $model $seed $input_size"
    qsub -cwd $QSUBOPTS -P $QSUBPROJECT -o $LOG -j y $waitid $bin $cmdopts
  done
done