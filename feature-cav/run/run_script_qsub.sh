#!/bin/bash
#$ -S /bin/bash
bin=$1
shift
cmdopts="$@"
HTE=lib/htesystem/HTE-volta.system # Default env file for sungrid queue

# look for optional arguments
while [ $# -gt 0 ]; do
    key=$1
    case $key in
        -hold_jid)
        shift
        hold_jid=$1
        shift
    ;;
    *)
    POSITIONAL+=("$1")
    shift
    ;;
    esac
done
set -- "${POSITIONAL[@]}"

LOGDIR=Logs/run_script_qsub
LOG=$LOGDIR/$(basename ${bin%.*}).log
if [ ! -d $LOGDIR ]; then
    mkdir -p $LOGDIR
fi
if [ -f $LOG ]; then
    \rm $LOG
fi

# check files exist
if [ ! -f $HTE ]; then
    echo "ERROR: HTE not found: $HTE"
    exit 100
fi

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

if [ ! -z $hold_jid ]; then
    echo qsub -hold_jid $hold_jid -cwd $QSUBOPTS -P $QSUBPROJECT -o $LOG -j y $waitid $bin $cmdopts --outdir $OUT 
    qsub -hold_jid $hold_jid -cwd $QSUBOPTS -P $QSUBPROJECT -o $LOG -j y $waitid $bin $cmdopts 
else
    echo qsub -cwd $QSUBOPTS -P $QSUBPROJECT -o $LOG -j y $waitid $bin $cmdopts --outdir $OUT
    qsub -cwd $QSUBOPTS -P $QSUBPROJECT -o $LOG -j y $waitid $bin $cmdopts  
fi