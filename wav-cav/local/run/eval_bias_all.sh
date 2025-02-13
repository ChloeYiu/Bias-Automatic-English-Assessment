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

if [ $# -lt 3 ]; then
  echo "Usage: $0 <trainset> <cavset> <testset>"
  exit 1
fi

trainset=$1
cavset=$2
testset=$3
config_file="eval/$trainset/arguments.conf"

# Check if config file exists
if [ ! -f "$config_file" ]; then
    echo "Error: Config file $config_file not found."
    exit 1
fi

seeds="24"

for part in 1; do
    top_outdir=eval/$trainset/part$part
    if [ -n "$class_weight" ]; then
        log_base_name=$top_outdir/bias/$cavset/$testset/bias_all_part${part}_input_layer_$class_weight
        log_file="LOGs/$log_base_name.log"
    else
        log_base_name=$top_outdir/bias/$cavset/$testset/bias_all_part${part}_input_layer
        log_file="LOGs/$log_base_name.log"
        class_weight="None"
    fi

    echo "Log file: $log_file"

    # Create log directory if it does not exist
    mkdir -p $(dirname "$log_file")
    if [ -f "$log_file" ]; then
        rm "$log_file"
    fi

    # Run the evaluation script with arguments from JSON file
    cmd="python local/python/eval_bias_all.py --TRAINSET $trainset --CAVSET $cavset --BIASSET $testset --CLASS_WEIGHT $class_weight  --CONFIG_FILE $config_file --PART $part --SEED $seeds --LAYER dense --TOP_DIR $top_outdir"
    echo $cmd
    $cmd >> $log_file 2>&1
done