#!/bin/bash
#$ -S /bin/bash

# evaluate BERT neural grader for Linguaskill, S&I style test

ALLARGS="$0 $@"
export PATH="/scratches/dialfs/kmk/anaconda3/envs/whisper39/bin:$PATH"
condaenv=/scratches/dialfs/kmk/anaconda3/envs/whisper39

# Set default values
PART_START=1
PART_END=5

activation_fn="relu" # Default activation function
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
        --lrelu)
        cmdopts="$cmdopts $1"
        activation_fn=lrelu
        shift
        ;;
    *)
    POSITIONAL+=("$1")
    shift
    ;;
    esac
done
set -- "${POSITIONAL[@]}"

echo "Feature flag: $feature"

# Check Number of Args
if [[ $# -ne 4 ]]; then
   echo "Usage: $0 [--condaenv path] [--part_range start:end] [--biased_TSET profile] [--feature] responses_dir tset top_outdir trainset"
   echo "  e.g: ./local/run/run_eval_ensemble_with_hooks.sh --part_range 1:1 --biased_TSET grade_C /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est LIESTgrp06_grade_C"
    echo "  e.g: ./local/run/run_eval_ensemble_with_hooks.sh --part_range 1:1 --feature /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est LIESTgrp06"

   echo ""
   exit 100
fi

SRC=$1 # responses_dir
TSET=$2
top_outdir=$3
trainset=$4
if [ "$feature" = true ]; then
    trainset=${trainset}_feature
elif ["$activation_fn" = "lrelu" ]; then
    trainset=${trainset}_lrelu
fi

# check files exist
RESPONSES=$SRC/$TSET.responses.txt
if [ ! -f $RESPONSES ]; then
    echo "ERROR: responses not found: $RESPONSES"
    exit 100
fi
if [ -n "$profile" ]; then
    echo "Profile: $profile"
    TSET=${TSET}_${profile}
    SCORES=$top_outdir/${TSET}/trained_models/scores_biased.txt
else
    SCORES=$SRC/$TSET.scores.txt
fi
if [ ! -f $SCORES ]; then
    echo "ERROR: scores not found: $SCORES"
    exit 100
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

    if [ "$feature" = true ]; then
        FEATURE=/research/milsrg1/alta/linguaskill/exp-ymy23/feature-cav/data/ALTA/ASR_V2.0.0/$TSET/f4-ppl-c2-pdf/part$PART/features.txt
        if [ ! -f "$FEATURE" ]; then
            echo "ERROR: test features not found: $FEATURE"
            exit 100
        fi
    fi

    mkdir -p $topdir/predictions/$TSET/part${PART}
    OUT=$topdir/predictions/$TSET/part${PART}
    ACTIVATION_DIR=$topdir/activations/$TSET/part${PART}
    GRADIENT_DIR=$topdir/gradients/$TSET/part${PART}
    echo "OUT=$OUT"
    echo "ACTIVATION_DIR=$ACTIVATION_DIR"
    echo "GRADIENT_DIR=$GRADIENT_DIR"
    echo "SCORES=$SCORES"
    log_file=LOGs/$topdir/predictions/$TSET/preds_bert_part${PART}.LOG
    
    echo "Logging to: $log_file"
    if [ "$feature" = true ]; then
        python local/python/eval_ensemble_with_hooks.py $opts "$MODELS" $RESPONSES $SCORES $OUT $ACTIVATION_DIR $GRADIENT_DIR --part=$PART --B=8 --FEATURE=$FEATURE >& $log_file
    else
        python local/python/eval_ensemble_with_hooks.py $opts "$MODELS" $RESPONSES $SCORES $OUT $ACTIVATION_DIR $GRADIENT_DIR --activation_fn=$activation_fn --part=$PART --B=8 >& $log_file
    fi
    echo "Hook done"
done

# run eval_all.py for all parts
prediction_source=$topdir/predictions/${TSET}
PREDS=$(ls -1 $prediction_source/part?/ensemble_preds_part*.txt) 
echo "PREDS=$PREDS"
OUT=$top_outdir/${trainset}/predictions/${TSET}/ensemble_preds_${TSET}.txt
echo "OUT=$OUT"
echo python local/python/eval_all.py $opts PREDS=$PREDS OUT=$OUT
python local/python/eval_all.py $opts "$PREDS" $OUT >& LOGs/$topdir/predictions/${TSET}/preds_all.LOG
