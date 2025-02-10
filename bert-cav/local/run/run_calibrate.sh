#!/bin/bash
#$ -S /bin/bash

# calculate calibration coefficients

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
if [[ $# -ne 3 ]]; then
   echo "Usage: $0 [--condaenv path] [--part_range start:end] top_outdir trainset calibrate_set"
   echo "  e.g: ./tools/calibrate.sh --part_range 2:3 est LIESTgrp06 LIESTcal01"
   echo ""
   exit 100
fi

top_outdir=$1
trainset=$2
CAL_TSET=$3

# check files exist
CAL_DIR=$top_outdir/$trainset/predictions/$CAL_TSET
if [ ! -d $CAL_DIR ]; then
    echo "ERROR: cal directory not found: $CAL_DIR"
    exit 100
fi

mkdir -p CMDs/$CAL_DIR
cmdfile=CMDs/$CAL_DIR/calibrate.cmds
echo $ALLARGS >> $cmdfile
echo "------------------------------------------------------------------------" >> $cmdfile

# activate conda environment
echo "conda activate $condaenv"
source activate "$condaenv" 

echo `hostname`
echo $PATH

opts=""

LOG_DIR=LOGs/$CAL_DIR
mkdir -p $LOG_DIR
echo "LOG_DIR=$LOG_DIR"

python --version

for PART in $(seq $PART_START $PART_END); do
    CAL=$CAL_DIR/part${PART}/ensemble_preds_part${PART}.txt 
    if [ ! -f $CAL ]; then
        echo "ERROR: cali not found: $CAL"
        exit 100
    fi
    echo "PART=$PART"
    echo "CAL=$CAL"

    OUT=$CAL_DIR/part${PART}/ensemble_calcoeffs_part${PART}.txt
    echo "OUT_CAL=$OUT"
    python ./local/python/calibrate.py $opts $CAL $OUT >& $LOG_DIR/calibrate_part${PART}.LOG
done

# run calibrate.py for all parts
CAL=$CAL_DIR/ensemble_preds_${CAL_TSET}.txt
if [ ! -f $CAL ]; then
    echo "No overall ensemble predictions found: $CAL"
    exit 0
fi

echo "all parts"
echo "CAL=$CAL"
OUT=$CAL_DIR/ensemble_calcoeffs_all.txt
echo "OUT_CAL=$OUT"
python ./local/python/calibrate.py $opts $CAL $OUT >& $LOG_DIR/calibrate_all.LOG


