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
config_file="eval/arguments.conf"

# Check if config file exists
if [ ! -f "$config_file" ]; then
    echo "Error: Config file $config_file not found."
    exit 1
fi

seeds="2,24"

for part in 1; do
    top_outdir=eval/${trainset}_${biasmodel}/part$part
    log_base_name=$top_outdir/compare/$trainset/$testset/${feature}_part${part}_dense

    if [ -n "$class_weight" ]; then
        log_file="LOGs/${log_base_name}_$class_weight.log"
        output_file=${log_base_name}_$class_weight.png
    else
        log_file="LOGs/$log_base_name.log"
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
    cmd="python local/python/compare.py --TRAINSET $trainset --BIASSET $testset --CLASS_WEIGHT $class_weight  --BIASMODEL $biasmodel --FEATURE $feature --PART $part --SEED $seeds --LAYER dense --OUTPUT_FILE $output_file --CONFIG_FILE $config_file"
    echo $cmd
    $cmd >> $log_file 2>&1
done