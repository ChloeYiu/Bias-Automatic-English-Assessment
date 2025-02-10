#!/bin/bash
#$ -S /bin/bash

# train BERT neural grader 

ALLARGS="$0 $@"
ARGS="$@"

cmdopts=""
part_range="1:1"   # Default part range
seed_range="1:10"  # Default seed range
HTE=lib/htesystem/HTE-volta.system # Default env file for sungrid queue
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

    *)
        POSITIONAL+=("$1")
        shift
        ;;
    
    esac
done
set -- "${POSITIONAL[@]}"


if [[ $# -lt 4 || $# -gt 5 ]]; then
   echo "Usage: $0 [--hte] [--condaenv path] [--part_range start:end] [--seed_range start:end] responses_dir tset top_outputdir profile [jwait]"
   echo "  e.g: $0 --part_range 1:5 --seed_range 1:10 /scratches/dialfs/alta/linguaskill/grd-graphemic-st941/neural-text/Wspt-D3/data/LIESTtrn04 LIESTtrn04 predictions/Wspt-D3 grade_C"
   echo "  e.g: $0 --part_range 1:1 --seed_range 1:5 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est grade_C"
   echo ""
   exit 100
fi

SRC=$1
TSET=$2
top_outdir=$3
profile=$4

if [[ $# -eq 5 ]]; then
    waitid="-hold_jid $5"
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
    for SEED in $(seq $(echo $seed_range | cut -d':' -f1) $(echo $seed_range | cut -d':' -f2)); do
        
        cmdopts="$SRC $TSET $top_outdir $PART $SEED $profile"
        
        OUT=$top_outdir/${TSET}_${profile}/trained_models/part${PART}/bert_part${PART}_seed${SEED}.th
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

        bin=./local/tools/train_grader_with_bias.sh 
        LOG=LOGs/$top_outdir/${TSET}_${profile}/part${PART}/train_grader_seed${SEED}.LOG
        mkdir -p $(dirname $LOG)
        if [ -f $LOG ]; then
            \rm $LOG
        fi

        echo qsub -cwd $QSUBOPTS -P $QSUBPROJECT -o $LOG -j y $waitid $bin $cmdopts --outdir $OUT
        qsub -cwd $QSUBOPTS -P $QSUBPROJECT -o $LOG -j y $waitid $bin $cmdopts
    done
done

