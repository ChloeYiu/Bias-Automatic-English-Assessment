#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

# local/run/predict_with_hook.sh LIESTgrp06 LIESTgrp06 
# local/run/predict_with_hook.sh LIESTgrp06 LIESTdev02
local/run/post_activation.sh LIESTgrp06 LIESTgrp06
local/run/post_activation.sh LIESTgrp06 LIESTdev02
batch/batch_extract_cav.sh 
batch/batch_eval_cav.sh 
batch/batch_eval_cav_mean_std.sh 
batch/batch_eval_bias_multiple.sh 
local/run/eval_bias_all.sh LIESTgrp06 LIESTgrp06 LIESTdev02 
local/run/eval_bias_all.sh LIESTgrp06 LIESTgrp06 LIESTdev02 --class_weight balanced