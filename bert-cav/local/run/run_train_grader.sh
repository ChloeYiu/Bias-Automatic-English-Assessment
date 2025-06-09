#!/bin/bash
#$ -S /bin/bash

# train BERT neural grader 

ALLARGS="$0 $@"
ARGS="$@"

cmdopts=""
part_range="1:1"   # Default part range
seed_range="1:10"  # Default seed range
HTE=lib/htesystem/HTE-volta.system # Default env file for sungrid queue

activation_fn="relu" # Default activation function
feature=false

# look for optional arguments
while [ $# -gt 0 ]; do
    key=$1
    case $key in
        --hte)
	shift
        HTE=$1
	shift
        ;;
        --condaenv)
        cmdopts="$cmdopts $1 $2"
	shift
	shift
        ;;
        --part_range)
        cmdopts="$cmdopts $1 $2"
        part_range=$2
    shift
    shift
        ;; 
        --seed_range)
        cmdopts="$cmdopts $1 $2"
        seed_range=$2
    shift
    shift
        ;;
        --feature)
        cmdopts="$cmdopts $1"
        feature=true
    shift
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

echo "activation function: $activation_fn"

if [[ $# -lt 3 || $# -gt 4 ]]; then
   echo "Usage: $0 [--hte] [--condaenv path] [--part_range start:end] [--seed_range start:end] responses_dir tset top_outputdir [jwait]"
   echo "  e.g: $0 --part_range 1:5 --seed_range 1:10 /scratches/dialfs/alta/linguaskill/grd-graphemic-st941/neural-text/Wspt-D3/data/LIESTtrn04 LIESTtrn04 predictions/Wspt-D3"
   echo "  e.g: $0 --part_range 1:1 --seed_range 1:5 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est --feature /research/milsrg1/alta/linguaskill/exp-ymy23/feature-cav/data/ALTA/ASR_V2.0.0/LIESTgrp06/f4-ppl-c2-pdf"
   echo "  e.g: $0 --part_range 1:1 --seed_range 1:5 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est --lrelu"
   echo ""
   exit 100
fi

SRC=$1
TSET=$2
top_outdir=$3

if [[ $# -eq 4 ]]; then
    waitid="-hold_jid $4"
else
    waitid=""
fi

# check files exist
if [ ! -f $HTE ]; then
    echo "ERROR: HTE not found: $HTE"
    exit 100
fi
RESPONSES=$SRC/$TSET.responses.txt
if [ ! -f $RESPONSES ]; then
    echo "ERROR: responses not found: $RESPONSES"
    exit 100
fi
SCORES=$SRC/$TSET.scores.txt
if [ ! -f $SCORES ]; then
    echo "ERROR: scores not found: $SCORES"
    exit 100
fi

mkdir -p CMDs/$top_outdir/${TSET}
cmdfile=CMDs/$top_outdir/${TSET}/run_train_grader.cmds
echo $ALLARGS >> $cmdfile
echo "------------------------------------------------------------------------" >> $cmdfile

# cmdopts="$cmdopts $2 $3 $4 $5 $6"
# cmdopts="$cmdopts $SRC $TSET $top_outdir"
# cmdopts="$cmdopts --part_range $part_range --seed_range $seed_range $SRC $TSET $top_outdir"

# Loop over part and seed and part range
echo "part_range: $part_range"
echo "seed_range: $seed_range"

for PART in $(seq $(echo $part_range | cut -d':' -f1) $(echo $part_range | cut -d':' -f2)); do
    mkdir -p $top_outdir/${TSET}/trained_models/part${PART}
    if [ "$feature" = true ]; then
        log_dir=LOGs/$top_outdir/${TSET}_feature/part${PART}
    elif [ $activation_fn == 'lrelu' ]; then
        log_dir=LOGs/$top_outdir/${TSET}_lrelu/part${PART}
    else
        log_dir=LOGs/$top_outdir/${TSET}/part${PART}
    fi
    mkdir -p $log_dir
    for SEED in $(seq $(echo $seed_range | cut -d':' -f1) $(echo $seed_range | cut -d':' -f2)); do

        if [ "$feature" = true ]; then
            cmdopts="$SRC $TSET $top_outdir $PART $SEED --feature_dir $feature"
        else
            cmdopts="$SRC $TSET $top_outdir $PART $SEED --activation_fn $activation_fn"
        fi
        
        OUT=$top_outdir/${TSET}/trained_models/part${PART}/bert_part${PART}_seed${SEED}.th
        echo OUT=$OUT

        source $HTE
        if [ -z $QSUBPROJECT ]; then
            QSUBPROJECT=esol
        fi

        QSUBOPTS=""
        if [ ! -z $QSUBQUEUE ]; then
            QSUBOPTS="$QSUBOPTS -l qp=$QSUBQUEUE"
        fi
        if [ ! -z $QGPUCLASS ]; then
         QSUBOPTS="$QSUBOPTS -l gpuclass=$QGPUCLASS"
        fi
        if [ ! -z $QHOSTNAME ]; then
            QSUBOPTS="$QSUBOPTS -l hostname=$QHOSTNAME"
        fi
        if [ ! -z $QCUDAMEM ]; then
            QSUBOPTS="$QSUBOPTS -l cudamem=$QCUDAMEM"
        fi
        if [ ! -z $QMAXJOBS ]; then
            QSUBOPTS="$QSUBOPTS -tc $QMAXJOBS"
        fi

        bin=./local/tools/train_grader.sh 
        LOG=$log_dir/train_grader_seed${SEED}.LOG
        if [ -f $LOG ]; then
            \rm $LOG
        fi

        echo qsub -cwd $QSUBOPTS -P $QSUBPROJECT -o $LOG -j y $waitid $bin $cmdopts --outdir $OUT
        qsub -cwd $QSUBOPTS -P $QSUBPROJECT -o $LOG -j y $waitid $bin $cmdopts
    done
done

