#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

export PATH="/scratches/dialfs/alta/hkv21/mconda_setup/miniconda_py3.11/envs/pytorch2.0/bin:$PATH"
condaenv=/scratches/dialfs/alta/hkv21/mconda_setup/miniconda_py3.11/envs/pytorch2.0/
source activate "$condaenv"

which python
python --version

if [ $# -lt 3 ]; then
  echo "Usage: $0 <train_data> <test_data> <calib_data>"
  exit 1
fi

train_data=$1
test_data=$2
calib_data=$3

LOG=Logs/run/process_data.log
mkdir -p $(dirname $LOG)
if [ -f $LOG ]; then
    \rm $LOG
fi
declare -a seeds=(10 30 50 70 90)

for part in 1; do
  for seed in "${seeds[@]}"; do   
    python local/training/score.py \
      --pred_file ./DDN/ALTA/ASR_V2.0.0/${train_data}/f4-ppl-c2-pdf/part${part}/DDN_${seed}/${test_data}/${test_data}_pred_ref.txt \
      --calib_model ./DDN/ALTA/ASR_V2.0.0/${train_data}/f4-ppl-c2-pdf/part${part}/DDN_${seed}/${calib_data}/calib_model.pkl;
  done

  python local/training/score.py \
  --pred_file ./DDN/ALTA/ASR_V2.0.0/${train_data}/f4-ppl-c2-pdf/part${part}/ens_${test_data}/${test_data}_pred_ref.txt \
  --calib_model ./DDN/ALTA/ASR_V2.0.0/${train_data}/f4-ppl-c2-pdf/part${part}/ens_${test_data}/calib_model.pkl;
done