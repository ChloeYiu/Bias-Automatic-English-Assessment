#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

HTE=lib/htesystem/HTE-volta.system # Default env file for sungrid queue

LOG=LOGs/run/train.log
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

bin=local/run/run_train.sh
waitid=""

declare -a seeds=(24)

if [ $# -lt 2 ]; then
  echo "Usage: $0 <train_set> <dev_set>"
  exit 1
fi

train_set=$1
dev_set=$2

for part in 1; do
  for seed in "${seeds[@]}"; do
    echo "Part $part, Seed $seed"
    cmdopts="$train_set $dev_set $part $seed"
    qsub -cwd $QSUBOPTS -P $QSUBPROJECT -o $LOG -j y $waitid $bin $cmdopts
  done
done

