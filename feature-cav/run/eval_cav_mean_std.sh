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
  echo "Usage: $0 <trainset> <cavset> <testset> <model> <profile>"
  exit 1
fi

trainset=$1
cavset=$2
testset=$3
model=$4
profile=$5
config_file="${model}/ALTA/ASR_V2.0.0/arguments.conf"

# Check if config file exists
if [ ! -f "$config_file" ]; then
    echo "Error: Config file $config_file not found."
    exit 1
fi

# Function to load configuration
load_config() {
    local profile=$1
    local config_file=$2
    eval $(awk -v profile="[$profile]" '
    $0 == profile {found=1; next}
    /^\[.*\]/ {found=0}
    found && NF {gsub(/ *= */, "=\""); print $0 "\""}
    ' $config_file)
}

# Load configuration
load_config "$profile" "$config_file"

# Check if configuration has been loaded properly
if [ -z "$TARGET_FILE" ] || [ -z "$SPEAKER_COLUMN" ] || [ -z "$TARGET_COLUMN" ]; then
    echo "Error: Configuration not loaded properly. Please check the config file."
    exit 1
fi


top_outdir=${model}/ALTA/ASR_V2.0.0/${trainset}

declare -a seeds=(10 30 50 70 90)


for part in 1; do
    file_name=""
    if [ -n "$class_weight" ]; then
        output_file=$top_outdir/cav/$cavset/$profile/$testset/eval_cav_part${part}_input_layer_$class_weight
        log_base_name=$top_outdir/cav/$cavset/$profile/$testset/eval_cav_mean_std_part${part}_input_layer_$class_weight
    else
        output_file=$top_outdir/cav/$cavset/$profile/$testset/eval_cav_part${part}_input_layer
        log_base_name=$top_outdir/cav/$cavset/$profile/$testset/eval_cav_mean_std_part${part}_input_layer
    fi


  for seed in "${seeds[@]}"; do
    if [ -n "$class_weight" ]; then
        input_base_name=$top_outdir/cav/$cavset/$profile/$testset/eval_cav_part${part}_${model}_${seed}_input_layer_$class_weight
    else
        input_base_name=$top_outdir/cav/$cavset/$profile/$testset/eval_cav_part${part}_${model}_${seed}_input_layer
    fi
    
    file_name="$file_name,$input_base_name.txt"

    log_file="Logs/$log_base_name.log"

    echo "Log file: $log_file"

    # Create log directory if it does not exist
    mkdir -p $(dirname "$log_file")
    if [ -f "$log_file" ]; then
        rm "$log_file"
    fi

    # Run the evaluation script with arguments from JSON file
    cmd="python local/training/eval_cav_mean_std.py  --INPUT_FILES $file_name --OUTPUT_FILE $output_file.txt"
    echo $cmd
    $cmd >> $log_file 2>&1
  done
done