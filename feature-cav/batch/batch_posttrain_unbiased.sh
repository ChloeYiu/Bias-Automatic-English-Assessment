#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

model=$1

# run/evaluate.sh LIESTgrp06 LIESTgrp06 $model
# run/evaluate.sh LIESTgrp06 LIESTdev02 $model
# run/evaluate.sh LIESTgrp06 LIESTcal01 $model
run/post_activation.sh LIESTgrp06 LIESTgrp06 $model
run/post_activation.sh LIESTgrp06 LIESTdev02 $model
# run/ensemble_score.sh LIESTgrp06 LIESTdev02 $model
# run/calibrate.sh LIESTgrp06 LIESTdev02 LIESTcal01 $model
# run/score.sh LIESTgrp06 LIESTdev02 LIESTcal01 $model
batch/batch_extract_cav.sh $model
batch/batch_eval_cav.sh $model
batch/batch_eval_cav_mean_std.sh $model
batch/batch_eval_bias_multiple.sh $model
run/eval_bias_all.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model
run/eval_bias_all.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model --class_weight balanced