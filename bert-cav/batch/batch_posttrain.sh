#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

profile=$1

#local/run/run_eval_ensemble_with_hooks.sh --biased_TSET $profile --part_range 1:1 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est LIESTgrp06_$profile
local/run/run_eval_ensemble_with_hooks.sh --part_range 1:1 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTdev02 LIESTdev02 est LIESTgrp06_$profile  
#local/run/run_extract_cav.sh LIESTgrp06_$profile est LIESTgrp06_$profile $profile
#local/run/run_extract_cav.sh LIESTgrp06_$profile est LIESTgrp06_$profile $profile --class_weight balanced
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06_$profile LIESTgrp06_$profile $profile
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06_$profile LIESTgrp06_$profile $profile --class_weight balanced
local/run/run_compare.sh est LIESTgrp06 LIESTdev02 $profile $profile
local/run/run_compare.sh est LIESTgrp06 LIESTdev02 $profile $profile --class_weight balanced