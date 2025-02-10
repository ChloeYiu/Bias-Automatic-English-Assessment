#!/bin/bash
#$ -S /bin/bash

# train BERT neural grader 

ALLARGS="$0 $@"
ARGS="$@"
export PATH="/scratches/dialfs/kmk/anaconda3/envs/whisper39/bin:$PATH"
condaenv=/scratches/dialfs/kmk/anaconda3/envs/whisper39


if [[ $# -lt 1 ]]; then
   echo "Usage: $0 [--biased_TSET] dataset top_outputdir profile"
   echo "  e.g: $0 --biased_TSET spanish LIESTgrp06 est spanish"
   echo ""
   exit 100
fi

# look for optional arguments
while [ $# -gt 0 ]; do
    key=$1
    case $key in
        --biased_TSET)
	    shift
        biased=$1
	    shift
        ;; 
    *)
    POSITIONAL+=("$1")
    shift
    ;;
    esac
done
set -- "${POSITIONAL[@]}"

dataset=$1
top_outdir=$2
profile=$3
if [ -n "$biased" ]; then
    SRC=$top_outdir/${dataset}_${biased}/trained_models/scores_biased.txt
    
    # Create log directory if it does not exist
    log_file=LOGs/$top_outdir/$dataset_${biased}/input_stat_$profile.LOG
    mkdir -p $(dirname "$log_file")
else
    SRC=/data/milsrg1/alta/linguaskill/relevance_v2/$dataset/$dataset.scores.txt

    # Create log directory if it does not exist
    log_file=LOGs/$top_outdir/$dataset/input_stat_$profile.LOG
    mkdir -p $(dirname "$log_file")
fi
#OUTPUT=$top_outdir/$dataset/input_stat.txt
config_file=$top_outdir/arguments.conf

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

# Activate conda environment
source activate "$condaenv"

# Load configuration
load_config "$profile" "$config_file"

cmd="python local/python/input_stat.py --DATA_FILE $SRC --TARGET_FILE $TARGET_FILE --SPEAKER_COLUMN $SPEAKER_COLUMN --TARGET_COLUMN $TARGET_COLUMN --SPEAKER_INDEX $SPEAKER_INDEX --TARGET_INDEX $TARGET_INDEX --TARGET_POSITIVE $TARGET_POSITIVE --TARGET_TO_REMOVE $TARGET_TO_REMOVE"
echo "Logging to: $log_file"
echo $cmd
$cmd >& $log_file
