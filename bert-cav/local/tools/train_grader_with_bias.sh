#!/bin/bash
#$ -S /bin/bash

# run BERT neural grader on LIEST data

ALLARGS="$0 $@"

#export PATH="/scratches/dialfs/alta/st941/conda/anaconda3/envs/whisper39/bin:$PATH"
#condaenv=/scratches/dialfs/alta/st941/conda/anaconda3/envs/whisper39

#export PATH="/research/milsrg1/user_workspace/st941/envs/whisper39/bin:$PATH"
#condaenv=/research/milsrg1/user_workspace/st941/envs/whisper39/

export PATH="/scratches/dialfs/kmk/anaconda3/envs/whisper39/bin:$PATH"
condaenv=/scratches/dialfs/kmk/anaconda3/envs/whisper39/

# look for optional arguments
while [ $# -gt 0 ]; do
    key=$1
    case $key in
        --condaenv)
	shift
        condaenv=$1
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
if [[ $# -ne 6 ]]; then
   echo "Usage: $0 [--condaenv path] responses_dir tset top_outdir part seed profile"
   echo "  e.g: ./local/tools/train_grader.sh /scratches/dialfs/alta/linguaskill/grd-graphemic-st941/neural-text/Wspt-D3/data/LIESTtrn04 LIESTtrn04 Wspt-D3 1 1 grade_C"
   echo "  e.g. ./local/tools/train_grader.sh /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est 4 37 grade_C"
   echo ""
   exit 100
fi

PART=$4
SEED=$5
profile=$6
SRC=$1
TSET=$2
top_outdir=$3

# check files exist
RESPONSES=$SRC/$TSET.responses.txt
if [ ! -f $RESPONSES ]; then
    echo "ERROR: responses not found: $RESPONSES"
    exit 100
fi
SCORES=$top_outdir/${TSET}_${profile}/trained_models/scores_biased.txt
if [ ! -f $SCORES ]; then
    echo "ERROR: scores not found: $SCORES"
    exit 100
fi

cmddir=CMDs/$top_outdir/${TSET}_${profile}/trained_models/part${p}
mkdir -p $cmddir
cmdfile=$cmddir/train_grader.cmds
echo $ALLARGS >> $cmdfile
echo "------------------------------------------------------------------------" >> $cmdfile

# activate conda environment
echo "conda activate $condaenv"
source activate "$condaenv"

unset LD_PRELOAD # when run on the stack it has /usr/local/grid/agile/admin/cudadevice.so which will give core dumped
export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo $X_SGE_CUDA_DEVICE

echo `hostname`
echo $PATH

python --version

opts=""

# for PART in $(seq $PART_START $PART_END); do
mkdir -p $top_outdir/${TSET}_${profile}/trained_models/part${PART}
    # for SEED in $(seq $SEED_START $SEED_END); do
echo "PART=$PART, SEED=$SEED"
OUT=$top_outdir/${TSET}_${profile}/trained_models/part${PART}/bert_part${PART}_seed${SEED}.th
echo OUT=$OUT

echo python local/python/train.py $opts --OUT=$OUT --RESPONSES=$RESPONSES --GRADES=$SCORES --seed=$SEED --part=$PART
python local/python/train.py $opts  --OUT=$OUT --RESPONSES=$RESPONSES --GRADES=$SCORES --seed=$SEED --part=$PART --B=8 --epochs=2 --sch=1
