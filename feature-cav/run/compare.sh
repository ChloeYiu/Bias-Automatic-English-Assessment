#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

export PATH="/scratches/dialfs/alta/hkv21/mconda_setup/miniconda_py3.11/envs/pytorch2.0/bin:$PATH"
condaenv=/scratches/dialfs/alta/hkv21/mconda_setup/miniconda_py3.11/envs/pytorch2.0/
source activate "$condaenv"

which python
python --version

while [ $# -gt 0 ]; do
    key=$1
    case $key in
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

if [ $# -lt 4 ]; then
  echo "Usage: $0 <trainset> <testset> <biasmodel> <feature>"
  exit 1
fi

trainset=$1
testset=$2
biasmodel=$3
feature=$4
config_file=arguments.conf

# Check if config file exists
if [ ! -f "$config_file" ]; then
    echo "Error: Config file $config_file not found."
    exit 1
fi

top_outdir=DDN/ALTA/ASR_V2.0.0
seeds="10:90"


for part in 1; do
    log_base_name=$top_outdir/${trainset}_${biasmodel}/compare/$trainset/$testset/${feature}_part${part}_input_layer

    if [ -n "$class_weight" ]; then
        log_file="Logs/${log_base_name}_$class_weight.log"
        output_file=${log_base_name}_$class_weight.png
    else
        log_file="Logs/$log_base_name.log"
        output_file=${log_base_name}.png
        class_weight="None"
    fi

    # Create log directory if it does not exist
    mkdir -p $(dirname "$log_file")
    echo "Log file: $log_file"

    # Create log directory if it does not exist
    mkdir -p $(dirname "$log_file")
    if [ -f "$log_file" ]; then
        rm "$log_file"
    fi

    # Run the evaluation script with arguments from JSON file
    cmd="python local/training/compare.py --TRAINSET $trainset --BIASSET $testset --CLASS_WEIGHT $class_weight  --BIASMODEL $biasmodel --FEATURE $feature --PART $part --SEED $seeds --LAYER input_layer --TOP_DIR $top_outdir --OUTPUT_FILE $output_file --CONFIG_FILE $config_file"
    echo $cmd
    $cmd >> $log_file 2>&1
done