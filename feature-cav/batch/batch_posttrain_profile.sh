#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

profile=$1
model=$2

run/evaluate.sh LIESTgrp06_$profile LIESTgrp06_$profile $model
run/evaluate.sh LIESTgrp06_$profile LIESTdev02 $model
run/post_activation.sh LIESTgrp06_$profile LIESTgrp06_$profile $model
run/post_activation.sh LIESTgrp06_$profile LIESTdev02 $model
run/extract_cav.sh LIESTgrp06_$profile LIESTgrp06_$profile $model $profile
run/extract_cav.sh LIESTgrp06_$profile LIESTgrp06_$profile $model $profile --class_weight balanced
run/eval_bias_multiple.sh LIESTgrp06_$profile LIESTgrp06_$profile LIESTdev02 $model $profile 
run/eval_bias_multiple.sh LIESTgrp06_$profile LIESTgrp06_$profile LIESTdev02 $model $profile --class_weight balanced
run/compare.sh LIESTgrp06 LIESTdev02 $profile $profile $model
run/compare.sh LIESTgrp06 LIESTdev02 $profile $profile $model --class_weight balanced
