#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

export PATH="/scratches/dialfs/alta/hkv21/mconda_setup/miniconda_py3.11/envs/pytorch2.0/bin:$PATH"
condaenv=/scratches/dialfs/alta/hkv21/mconda_setup/miniconda_py3.11/envs/pytorch2.0/
source activate "$condaenv"

which python
python --version

path_template="./data/ALTA/ASR_V2.0.0/\${setname}/f4-ppl-c2-pdf/part\${part}"
parts=(1)

train_set="LIESTgrp06"
calib_set="LIESTcal01"
test_set="LIESTdev02"

LOG=Logs/run/process_data.log
mkdir -p $(dirname $LOG)
if [ -f $LOG ]; then
    \rm $LOG
fi

for part in ${parts[@]}; do
    for setname in $train_set $calib_set $test_set; do
        eval data_dir="$path_template"
        echo "Processing data for $data_dir" >> $LOG
        python local/feature_extraction/process_data.py --data_dir ${data_dir} >> $LOG 2>&1


        # Check if the current set is the train set
        if [[ $setname == $train_set ]]; then
            python local/feature_extraction/compute_whitening_transform.py --data_dir ${data_dir} >> $LOG 2>&1
        fi
    done
done
