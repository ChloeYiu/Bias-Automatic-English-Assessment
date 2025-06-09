#!/bin/bash
#$ -S /bin/bash

# evaluate CAV for BERT neural grader for Linguaskill, S&I style test

ALLARGS="$0 $@"
export PATH="/scratches/dialfs/kmk/anaconda3/envs/whisper39/bin:$PATH"
condaenv=/scratches/dialfs/kmk/anaconda3/envs/whisper39

# Set default values
part_range="1:1"   # Default part range
seed_range="1:5"  # Default seed range
layer_range="1:1"  # Default layer range

# look for optional arguments
while [ $# -gt 0 ]; do
    key=$1
    case $key in
        --condaenv)
	    shift
        condaenv=$1
	    shift
        ;; 
        --part_range)
        cmdopts="$cmdopts $1 $2"
        part_range=$2
        shift
        shift
        ;; 
        --seed_range)
        cmdopts="$cmdopts $1 $2"
        seed_range=$2
        shift
        shift
        ;;
        --layer_range)
        cmdopts="$cmdopts $1 $2"
        layer_range=$2
        shift
        shift
        ;;
        --class_weight)
        cmdopts="$cmdopts $1 $2"
        class_weight=$2
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

if [[ $# -ne 5 ]]; then
   echo "Usage: $0 [--condaenv path] [--part_range start:end] [--seed_range start:end] [--layer_range start:end] [--class_weight weight] testset top_outdir trainset cavset profile"
   echo "  e.g: ./local/run/run_eval_cav.sh LIESTdev02 est LIESTgrp06 LIESTgrp06 grade_C"
   echo ""
   exit 100
fi

testset=$1
top_outdir=$2
trainset=$3
cavset=$4
config_file=$top_outdir/arguments.conf
profile=$5

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

for PART in $(seq $(echo $part_range | cut -d':' -f1) $(echo $part_range | cut -d':' -f2)); do
    for SEED in $(seq $(echo $seed_range | cut -d':' -f1) $(echo $seed_range | cut -d':' -f2)); do
        for LAYER in $(seq $(echo $layer_range | cut -d':' -f1) $(echo $layer_range | cut -d':' -f2)); do
            # Extract base name of activation file without extension
            activation_base_name=$top_outdir/$trainset/activations/$testset/part${PART}/activations_bert_part${PART}_seed${SEED}_layer${LAYER}
            if [ -n "$class_weight" ]; then
                cav_base_name=$top_outdir/$trainset/cav/$cavset/part${PART}/${profile}/cav_bert_part${PART}_seed${SEED}_layer${LAYER}_$class_weight
                log_base_name=$top_outdir/$trainset/cav/$cavset/part${PART}/${profile}/${testset}/eval_cav_bert_part${PART}_seed${SEED}_layer${LAYER}_$class_weight
            else
                cav_base_name=$top_outdir/$trainset/cav/$cavset/part${PART}/${profile}/cav_bert_part${PART}_seed${SEED}_layer${LAYER}
                log_base_name=$top_outdir/$trainset/cav/$cavset/part${PART}/${profile}/${testset}/eval_cav_bert_part${PART}_seed${SEED}_layer${LAYER}
            fi
            log_file="LOGs/$log_base_name.LOG"

            # Create log directory if it does not exist
            mkdir -p $(dirname "$log_file")

            # Run the evaluation script with arguments from JSON file
            cmd="python local/python/eval_cav.py --TARGET_FILE $TARGET_FILE --ACTIVATION_FILE $activation_base_name.filtered --CAV_FILE $cav_base_name.txt --SPEAKER_COLUMN $SPEAKER_COLUMN --TARGET_COLUMN $TARGET_COLUMN --SPEAKER_INDEX $SPEAKER_INDEX --TARGET_INDEX $TARGET_INDEX --TARGET_POSITIVE $TARGET_POSITIVE --TARGET_TO_REMOVE $TARGET_TO_REMOVE --OUTPUT $log_base_name.txt"
            echo $cmd
            $cmd >& $log_file
        done
    done
done