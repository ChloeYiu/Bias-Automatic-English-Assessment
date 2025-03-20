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

if [ $# -lt 5 ]; then
  echo "Usage: $0 <trainset> <cavset> <testset> <model> <profile>"
  exit 1
fi

trainset=$1
cavset=$2
testset=$3
model=$4
profile=$5
config_file=arguments.conf

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

top_outdir=${model}/ALTA/ASR_V2.0.0/${trainset}
seeds="10:90"

for part in 1; do
    activation_base_name=$top_outdir/activations/$cavset/activations_part${part}_input_layer

    if [ -n "$class_weight" ]; then
        log_base_name=$top_outdir/bias/$cavset/$profile/$testset/bias_part${part}_input_layer_$class_weight
        log_file="Logs/$log_base_name.log"
    else
        log_base_name=$top_outdir/bias/$cavset/$profile/$testset/bias_part${part}_input_layer
        log_file="Logs/$log_base_name.log"
        class_weight="None"
    fi

    echo "Log file: $log_file"

    # Create log directory if it does not exist
    mkdir -p $(dirname "$log_file")
    if [ -f "$log_file" ]; then
        rm "$log_file"
    fi

    # Run the evaluation script with arguments from JSON file
    cmd="python local/training/eval_bias_multiple.py --TRAINSET $trainset --CAVSET $cavset --BIASSET $testset --CLASS_WEIGHT $class_weight  --BIAS $profile --PART $part --SEED $seeds --MODEL $model --LAYER input_layer --TOP_DIR $top_outdir"
    echo $cmd
    $cmd >> $log_file 2>&1
done