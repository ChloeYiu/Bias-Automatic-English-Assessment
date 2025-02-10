#!/bin/bash
#$ -S /bin/bash


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

# Train set and cav set the same
if [[ $# -ne 5 ]]; then
   echo "Usage: $0 [--condaenv path] [--part_range start:end] [--seed_range start:end] [--layer_range start:end] [--class_weight weight] top_outdir trainset biasset biasmodel feature"
   echo "  e.g: ./local/run/run_compare.sh est LIESTgrp06 LIESTdev02 spanish spanish"
   echo ""
   exit 100
fi

top_outdir=$1
trainset=$2
biasset=$3
biasmodel=$4
feature=$5
config_file=$top_outdir/arguments.conf


for PART in $(seq $(echo $part_range | cut -d':' -f1) $(echo $part_range | cut -d':' -f2)); do
    for LAYER in $(seq $(echo $layer_range | cut -d':' -f1) $(echo $layer_range | cut -d':' -f2)); do
        log_base_name=$top_outdir/${trainset}_${biasmodel}/compare/${trainset}/$biasset/part${PART}/${feature}_part${PART}_layer${LAYER}

        if [ -n "$class_weight" ]; then
            output_file=${log_base_name}_$class_weight.png
            log_file="LOGs/${log_base_name}_$class_weight.LOG"
        else
            output_file=$log_base_name.png
            log_file="LOGs/$log_base_name.LOG"
            class_weight="None"
        fi

        # Create log directory if it does not exist
        mkdir -p $(dirname "$log_file")
        echo "Logging to $log_file"


        # Run the evaluation script with arguments from JSON file
        cmd="python local/python/compare.py --TRAINSET $trainset --BIASSET $biasset --CLASS_WEIGHT $class_weight  --BIASMODEL $biasmodel --FEATURE $feature --PART $PART --SEED $seed_range --LAYER $LAYER --TOP_DIR $top_outdir --OUTPUT_FILE $output_file --CONFIG_FILE $config_file"
        echo $cmd
        $cmd >& $log_file
    done
done