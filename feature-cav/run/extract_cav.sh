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

if [ $# -lt 3 ]; then
  echo "Usage: $0 <trainset> <cavset> <profile>"
  exit 1
fi

trainset=$1
cavset=$2
profile=$3
config_file="DDN/ALTA/ASR_V2.0.0/arguments.conf"

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

top_outdir=./DDN/ALTA/ASR_V2.0.0/${trainset}
declare -a seeds=(10 30 50 70 90)

for part in 1; do
  for seed in "${seeds[@]}"; do
    activation_base_name=$top_outdir/activations/$cavset/activations_part${part}_DDN_${seed}_input_layer

    if [ -n "$class_weight" ]; then
        output_base_name=$top_outdir/cav/$cavset/$profile/cav_part${part}_DDN_${seed}_input_layer_$class_weight
    else
        output_base_name=$top_outdir/cav/$cavset/$profile/cav_part${part}_DDN_${seed}_input_layer
    fi
    
    log_file="Logs/$output_base_name.log"

    # Create log directory if it does not exist
    mkdir -p $(dirname "$log_file")
    if [ -f "$log_file" ]; then
        rm "$log_file"
    fi

    # Run the evaluation script with arguments from JSON file
    cmd="python local/training/extract_cav.py --TARGET_FILE $TARGET_FILE --ACTIVATION_FILE $activation_base_name.txt --OUTPUT_FILE $output_base_name.txt --SPEAKER_COLUMN $SPEAKER_COLUMN --TARGET_COLUMN $TARGET_COLUMN --SPEAKER_INDEX $SPEAKER_INDEX --TARGET_INDEX $TARGET_INDEX --TARGET_POSITIVE $TARGET_POSITIVE --TARGET_TO_REMOVE $TARGET_TO_REMOVE"
    if [ -n "$class_weight" ]; then
        cmd="$cmd --CLASS_WEIGHT $class_weight"
    fi
    echo $cmd
    $cmd >> $log_file 2>&1

  done
done