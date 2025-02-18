
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

if [ $# -lt 2 ]; then
  echo "Usage: $0 <trainset> <calset>"
  exit 1
fi

trainset=$1
calset=$2

LOG=LOGs/run/ensemble.log
    mkdir -p $(dirname $LOG)
    if [ -f $LOG ]; then
        \rm $LOG
    fi

echo "Logging to: $LOG"

for part in 1; do
    top_outdir=eval/$trainset/part$part
    calibration_pred_file=$top_outdir/predictions/$calset/preds_wav2vec_part${part}_ensemble.txt
    calibration_res_file=$top_outdir/predictions/$calset/calib_wav2vec_part${part}_ensemble.txt
    cmd="python local/python/calibrate.py --CALIBRATION_PRED_FILE $calibration_pred_file --CALIBRATION_RES_FILE $calibration_res_file"

    echo $cmd
    $cmd >> $LOG 2>&1
done