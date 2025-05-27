#!/bin/bash
#$ -S /bin/bash

# evaluate BERT neural grader for Linguaskill, S&I style test

ALLARGS="$0 $@"
export PATH="/scratches/dialfs/kmk/anaconda3/envs/whisper39/bin:$PATH"
condaenv=/scratches/dialfs/kmk/anaconda3/envs/whisper39

# Set default values
PART_START=1
PART_END=5

feature=false

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
        shift
        PART_START=$(echo $1 | cut -d':' -f1)
        PART_END=$(echo $1 | cut -d':' -f2)
        shift
        ;;
        --biased_TSET)
	    shift
        profile=$1
	    shift
        ;; --feature)
        cmdopts="$cmdopts $1"
        feature=true
        shift
        ;;
    *)
    POSITIONAL+=("$1")
    shift
    ;;
    esac
done
set -- "${POSITIONAL[@]}"

# Check Number of Args
if [[ $# -ne 3 ]]; then
   echo "Usage: $0 [--condaenv path] [--part_range start:end] [--feature] tset top_outdir trainset"
    echo "  e.g: ./local/run/run_post_activation.sh --part_range 1:1 --feature LIESTgrp06 est LIESTgrp06"

   echo ""
   exit 100
fi

TSET=$1
top_outdir=$2
trainset=$3

if [ "$feature" = true ]; then
    trainset=${trainset}_feature
fi

topdir=$top_outdir/${trainset}
if [ ! -d $topdir/trained_models ]; then
    echo "ERROR: no trained_models found: $top_outdir/${trainset}/trained_models"
    exit 100
fi

mkdir -p CMDs/$topdir
cmdfile=CMDs/$topdir/eval_ensemble.cmds
echo $ALLARGS >> $cmdfile
echo "------------------------------------------------------------------------" >> $cmdfile

# activate conda environment
echo "conda activate $condaenv"
source activate "$condaenv" 

export CUDA_VISIBLE_DEVICES=0,1

echo `hostname`
echo $PATH

opts=""

mkdir -p LOGs/$topdir/predictions/${TSET}
echo "LOG_DIR=LOGs/$topdir/predictions/${TSET}"

python --version

for PART in $(seq $PART_START $PART_END); do
    model_source=$topdir/trained_models/part${PART}
    if [ ! -d $model_source ]; then
        echo "ERROR: no trained_models found: $model_source"
        exit 100
    fi
    MODELS=$(ls -1 $model_source/*.th)   #provide model path and find all model 
    # MODELS=$(ls $model_source/)
    echo "PART=$PART"
    echo "MODELS=$MODELS"

    mkdir -p $topdir/predictions/$TSET/part${PART}
    ACTIVATION_DIR=$topdir/activations/$TSET/part${PART}
    GRADIENT_DIR=$topdir/gradients/$TSET/part${PART}
    echo "ACTIVATION_DIR=$ACTIVATION_DIR"
    echo "GRADIENT_DIR=$GRADIENT_DIR"
    log_file=LOGs/$topdir/predictions/$TSET/post_activation_part${PART}.LOG
    
    echo "Logging to: $log_file"
    
    python local/python/post_activation.py "$MODELS" $ACTIVATION_DIR $GRADIENT_DIR --part=$PART >& $log_file
done
