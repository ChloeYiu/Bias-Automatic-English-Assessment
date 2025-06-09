#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

profile=$1

# local/run/predict_with_hook.sh LIESTgrp06 LIESTgrp06 --biased_train $profile --biased_test $profile
# local/run/predict_with_hook.sh LIESTgrp06 LIESTdev02 --biased_train $profile
local/run/post_activation.sh LIESTgrp06_$profile LIESTgrp06_$profile 
local/run/post_activation.sh LIESTgrp06_$profile LIESTdev02 $profile
local/run/extract_cav.sh LIESTgrp06_$profile LIESTgrp06_$profile $profile
local/run/extract_cav.sh LIESTgrp06_$profile LIESTgrp06_$profile $profile --class_weight balanced
local/run/eval_bias_multiple.sh LIESTgrp06_$profile LIESTgrp06_$profile LIESTdev02 $profile
local/run/eval_bias_multiple.sh LIESTgrp06_$profile LIESTgrp06_$profile LIESTdev02 $profile --class_weight balanced
local/run/compare.sh LIESTgrp06 LIESTdev02 $profile $profile
local/run/compare.sh LIESTgrp06 LIESTdev02 $profile $profile --class_weight balanced
