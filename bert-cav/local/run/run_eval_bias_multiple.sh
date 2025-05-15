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
   echo "Usage: $0 [--condaenv path] [--part_range start:end] [--seed_range start:end] [--layer_range start:end] [--class_weight weight] biasset top_outdir trainset cavset profile"
   echo "  e.g: ./local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06 LIESTgrp06 thai"
   echo ""
   exit 100
fi

biasset=$1
top_outdir=$2
trainset=$3
cavset=$4
profile=$5

for PART in $(seq $(echo $part_range | cut -d':' -f1) $(echo $part_range | cut -d':' -f2)); do
    for LAYER in $(seq $(echo $layer_range | cut -d':' -f1) $(echo $layer_range | cut -d':' -f2)); do
        # Extract base name of activation file without extension
        if [ -n "$class_weight" ]; then
            log_base_name=$top_outdir/$trainset/bias/$cavset/$biasset/part${PART}/${profile}/bias_bert_part${PART}_layer${LAYER}_$class_weight
        else
            log_base_name=$top_outdir/$trainset/bias/$cavset/$biasset/part${PART}/${profile}/bias_bert_part${PART}_layer${LAYER}
            class_weight="None"
        fi
        log_file="LOGs/$log_base_name.LOG"
        echo "Logging to $log_file"

        # Create log directory if it does not exist
        mkdir -p $(dirname "$log_file")

        # Run the evaluation script with arguments from JSON file
        cmd="python local/python/eval_bias_multiple.py --TRAINSET $trainset --CAVSET $cavset --BIASSET $biasset --CLASS_WEIGHT $class_weight  --BIAS $profile --PART $PART --SEED $seed_range --LAYER $LAYER --TOP_DIR $top_outdir"
        echo $cmd
        $cmd >& $log_file
    done
done