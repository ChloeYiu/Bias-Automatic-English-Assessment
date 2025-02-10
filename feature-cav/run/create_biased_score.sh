#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

export PATH="/scratches/dialfs/alta/hkv21/mconda_setup/miniconda_py3.11/envs/pytorch2.0/bin:$PATH"
condaenv=/scratches/dialfs/alta/hkv21/mconda_setup/miniconda_py3.11/envs/pytorch2.0/
source activate "$condaenv"

which python
python --version

file_dir="./data/ALTA/ASR_V2.0.0/\${setname}/f4-ppl-c2-pdf/part\${part}"
biased_file_dir="./data/ALTA/ASR_V2.0.0/\${setname}_\${profile}/f4-ppl-c2-pdf/part\${part}"

grade_template="$file_dir/grades.txt"
biased_grade_template="$biased_file_dir/grades.txt"
feature_template="$file_dir/features.txt"
biased_feature_template="$biased_file_dir/features.txt"

parts=(1)

train_set="LIESTgrp06"
config_file=DDN/ALTA/ASR_V2.0.0/$train_set/arguments.conf
profile=$1

# Function to load configuration
load_config() {
    local profile=$1
    local config_file=$2
    eval $(awk -v profile="[$profile]" '
    $0 == profile {found=1; next}
    /^\[.*\]/ {found=0}
    found && NF {gsub(/ *= */, "="); print}
    ' $config_file)
}

# Load configuration
load_config "$profile" "$config_file"

LOG=Logs/run/create_biased_score.log
mkdir -p $(dirname $LOG)
if [ -f $LOG ]; then
    \rm $LOG
fi

for part in ${parts[@]}; do
    for setname in $train_set; do
        eval SCORES="$grade_template"
        eval OUT_FILE="$biased_grade_template"
        eval FEATURES="$feature_template"
        eval FEATURES_COPY="$biased_feature_template"
        echo "Logging to $LOG"
        echo "Creating biased score for $SCORES to $OUT_FILE" >> $LOG
        python local/training/create_biased_score.py --TARGET_FILE $TARGET_FILE --GRADES_FILE $SCORES --OUT_FILE $OUT_FILE --SPEAKER_COLUMN $SPEAKER_COLUMN --TARGET_COLUMN $TARGET_COLUMN --SPEAKER_INDEX $SPEAKER_INDEX --TARGET_INDEX $TARGET_INDEX --TARGET_POSITIVE $TARGET_POSITIVE >> $LOG 2>&1
        echo "Copying feature files from $FEATURES to $FEATURES_COPY" >> $LOG
        cp $FEATURES $FEATURES_COPY
    done
done
