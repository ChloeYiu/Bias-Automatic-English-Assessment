#!/bin/bash
#$ -S /bin/bash


ALLARGS="$0 $@"
ARGS="$@"

cmdopts=""

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


if [[ $# -ne 4 ]]; then
   echo "Usage: $0 [--condaenv path] responses_dir tset top_outputdir profile"
   echo "  e.g: $0 /scratches/dialfs/alta/linguaskill/grd-graphemic-st941/neural-text/Wspt-D3/data/LIESTtrn04 LIESTtrn04 predictions/Wspt-D3 grade_C"
   echo "  e.g: $0 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est grade_C"
   echo ""
   exit 100
fi

SRC=$1
TSET=$2
top_outdir=$3
config_file=$top_outdir/arguments.conf
profile=$4

# Function to load configuration
load_config() {
    local profile=$1
    local config_file=$2
    eval $(awk -v profile="[$profile]" '
    $0 == profile {found=1; next}
    /^\[.*\]/ {found=0}
    found && NF {gsub(/ *= */, "="); print}
    ' $config_file)
}

# Activate conda environment
source activate "$condaenv"

# Load configuration
load_config "$profile" "$config_file"

SCORES=$SRC/$TSET.scores.txt
if [ ! -f $SCORES ]; then
    echo "ERROR: scores not found: $SCORES"
    exit 100
fi

# cmdopts="$cmdopts $2 $3 $4 $5 $6"
# cmdopts="$cmdopts $SRC $TSET $top_outdir"
# cmdopts="$cmdopts --part_range $part_range --seed_range $seed_range $SRC $TSET $top_outdir"

OUT_FILE=$top_outdir/${TSET}_${profile}/trained_models/scores_biased.txt
log_file=LOGs/$top_outdir/${TSET}_${profile}/create_biased_score.LOG
mkdir -p "$(dirname "$log_file")"

cmd="python local/python/create_biased_score.py --TARGET_FILE $TARGET_FILE --GRADES_FILE $SCORES --OUT_FILE $OUT_FILE --SPEAKER_COLUMN $SPEAKER_COLUMN --TARGET_COLUMN $TARGET_COLUMN --SPEAKER_INDEX $SPEAKER_INDEX --TARGET_INDEX $TARGET_INDEX --TARGET_POSITIVE $TARGET_POSITIVE"

echo $cmd
$cmd >& $log_file
