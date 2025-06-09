#!/bin/bash
#$ -S /bin/bash

# evaluate calibrated predictions

ALLARGS="$0 $@"

export PATH="/scratches/dialfs/kmk/anaconda3/envs/whisper39/bin:$PATH"
condaenv=/scratches/dialfs/kmk/anaconda3/envs/whisper39

# Set default values
PART_START=1
PART_END=5

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
    *)
    POSITIONAL+=("$1")
    shift
    ;;
    esac
done
set -- "${POSITIONAL[@]}"

# Check Number of Args
if [[ $# -ne 4 ]]; then
   echo "Usage: $0 [--condaenv path] [--part_range start:end] top_outdir trainset calibrate_set preds_set"
   echo "  e.g: ./local/run/run_eval_calibrate.sh --part_range 2:3 Wtest LIESTtrn04 LIESTcal01 LIALTtst02"
   echo ""
   exit 100
fi

top_outdir=$1
trainset=$2
CAL_TSET=$3
PREDS_TSET=$4

# check files exist
CAL_DIR=$top_outdir/$trainset/predictions/$CAL_TSET
if [ ! -d $CAL_DIR ]; then
    echo "ERROR: cal directory not found: $CAL_DIR"
    exit 100
fi

PREDS_DIR=$top_outdir/$trainset/predictions/$PREDS_TSET
if [ ! -d $PREDS_DIR ]; then
    echo "ERROR: pred directory not found: $PREDS_DIR"
    exit 100
fi

mkdir -p CMDs/$PREDS_DIR
cmdfile=CMDs/$PREDSq_DIR/eval_calibrate.cmds
echo $ALLARGS >> $cmdfile
echo "------------------------------------------------------------------------" >> $cmdfile

# activate conda environment
echo "conda activate $condaenv"
source activate "$condaenv" 

echo `hostname`
echo $PATH

opts=""

mkdir -p LOGs/$top_outdir/${trainset}_calibrate
echo "LOG_DIR=LOGs/$top_outdir/${trainset}_calibrate"

python --version

LOG_DIR=LOGs/$PREDS_DIR
mkdir -p $LOG_DIR
echo "LOG_DIR=$LOG_DIR"

for PART in $(seq $PART_START $PART_END); do
    CAL=$CAL_DIR/part${PART}/ensemble_calcoeffs_part${PART}.txt 
    PREDS=$PREDS_DIR/part${PART}/ensemble_preds_part${PART}.txt
    if [ ! -f $CAL ]; then
        echo "ERROR: calibration file for part${PART} not found: $CAL"
        exit 100
    fi
    if [ ! -f $PREDS ]; then
        echo "ERROR: predictions file for part${PART} not found: $PREDS"
        exit 100
    fi
    echo "PART=$PART"
    echo "CAL=$CAL"
    echo "PREDS=$PREDS"

    # Get calibration coefficients
    GRADIENT=`awk '{if($1~"gradient:") print $2;}' $CAL`
    INTERCEPT=`awk '{if($1~"intercept:") print $2;}' $CAL`

    # Pass the values to the next Python script
    OUT=$PREDS_DIR/part${PART}/ensemble_cal_part${PART}.txt
    echo "OUT_PRED=$OUT"
    python ./local/python/eval_all_calibrate.py "$PREDS" $OUT --gradient=$GRADIENT --intercept=$INTERCEPT >& $LOG_DIR/eval_all_calibrate_part${PART}.LOG
done

# run calibrate.py for all parts
CAL=$CAL_DIR/ensemble_calcoeffs_all.txt
PREDS=$(ls $PREDS_DIR/part*/ensemble_preds_part*.txt)
if [ ! -f $CAL ]; then
    echo "ERROR: ensemble_cal not found: $CAL"
    exit 100
fi
if [ -z "$PREDS" ]; then
    echo "ERROR: ensemble_preds not found: $PREDS"
    exit 100
fi
echo "all parts"
echo "CAL=$CAL"
echo "PREDS=$PREDS"

# Get calibration coefficients
GRADIENT=`awk '{if($1~"gradient:") print $2;}' $CAL`
INTERCEPT=`awk '{if($1~"intercept:") print $2;}' $CAL`

echo "GRADIENT=$GRADIENT"
echo "INTERCEPT=$INTERCEPT"

# Pass the values to the next Python script
OUT2=${PREDS_DIR}/ensemble_cal_${PREDS_TSET}.txt
echo "OUT_PRED=$OUT2"

echo "Logging to $LOG_DIR/eval_all_calibrate_all.LOG"

python ./local/python/eval_all_calibrate.py "$PREDS" $OUT2 --gradient=$GRADIENT --intercept=$INTERCEPT >& $LOG_DIR/eval_all_calibrate_all.LOG


