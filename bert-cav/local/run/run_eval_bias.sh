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
   echo "Usage: $0 [--condaenv path] [--part_range start:end] [--seed_range start:end] [--layer_range start:end] [--class_weight weight] biasset top_outdir trainset cavset"
   echo "  e.g: ./local/run/run_eval_bias.sh LIESTdev02 est LIESTgrp06 LIESTgrp06 thai"
   echo ""
   exit 100
fi

biasset=$1
top_outdir=$2
trainset=$3
cavset=$4
profile=$5

for PART in $(seq $(echo $part_range | cut -d':' -f1) $(echo $part_range | cut -d':' -f2)); do
    for SEED in $(seq $(echo $seed_range | cut -d':' -f1) $(echo $seed_range | cut -d':' -f2)); do
        for LAYER in $(seq $(echo $layer_range | cut -d':' -f1) $(echo $layer_range | cut -d':' -f2)); do
            # Extract base name of activation file without extension
            if [ -n "$class_weight" ]; then
                cav_file=$top_outdir/$trainset/cav/$cavset/part${PART}/${profile}/cav_bert_part${PART}_seed${SEED}_layer${LAYER}_$class_weight.txt
                log_base_name=$top_outdir/$trainset/bias/$cavset/$biasset/part${PART}/${profile}/bias_bert_part${PART}_seed${SEED}_layer${LAYER}_$class_weight
            else
                cav_file=$top_outdir/$trainset/cav/$cavset/part${PART}/${profile}/cav_bert_part${PART}_seed${SEED}_layer${LAYER}.txt
                log_base_name=$top_outdir/$trainset/bias/$cavset/$biasset/part${PART}/${profile}/bias_bert_part${PART}_seed${SEED}_layer${LAYER}
            fi
            plot_file=$log_base_name.png
            grad_file=$top_outdir/$trainset/gradients/$biasset/part${PART}/gradients_bert_part${PART}_seed${SEED}_layer${LAYER}.filtered
            pred_file=$top_outdir/$trainset/predictions/$biasset/part${PART}/preds_bert_part${PART}_seed${SEED}.txt
            log_file="LOGs/$log_base_name.LOG"

            # Create log directory if it does not exist
            mkdir -p $(dirname "$log_file")

            # Run the evaluation script with arguments from JSON file
            cmd="python local/python/eval_bias.py --CAV_FILE $cav_file --GRADIENT_FILE $grad_file --PRED_FILE $pred_file --PLOT_FILE $plot_file --BIAS $profile"
            echo $cmd
            $cmd >& $log_file
        done
    done
done