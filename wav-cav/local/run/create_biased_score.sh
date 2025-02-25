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

parts=(1)

train_set=$1
config_file=eval/arguments.conf
profile=$2

file_dir="data_vectors_attention/$train_set"
biased_file_dir="models/${train_set}_$profile"

grade_template="$file_dir/grades.txt"

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

LOG=LOGs/run/create_biased_score.log
mkdir -p $(dirname $LOG)
if [ -f $LOG ]; then
    \rm $LOG
fi

for part in ${parts[@]}; do
    for setname in $train_set; do
        SCORES="$file_dir/${train_set}_part${part}_att.hf"
        OUT_FILE="$biased_file_dir/part$part/scores_biased.json"
        echo "Logging to $LOG"
        echo "Creating biased score for $SCORES to $OUT_FILE" >> $LOG
        python local/python/create_biased_score.py --TARGET_FILE $TARGET_FILE --GRADES_FILE $SCORES --OUT_FILE $OUT_FILE --SPEAKER_COLUMN $SPEAKER_COLUMN --TARGET_COLUMN $TARGET_COLUMN --SPEAKER_INDEX $SPEAKER_INDEX --TARGET_INDEX $TARGET_INDEX --TARGET_POSITIVE $TARGET_POSITIVE >> $LOG 2>&1
    done
done
