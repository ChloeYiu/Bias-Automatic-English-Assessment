#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

profile=$1

run/evaluate.sh LIESTgrp06_$profile LIESTgrp06_$profile
run/evaluate.sh LIESTgrp06_$profile LIESTdev02
run/extract_cav.sh LIESTgrp06_$profile LIESTgrp06_$profile $profile
run/extract_cav.sh LIESTgrp06_$profile LIESTgrp06_$profile $profile --class_weight balanced
run/eval_bias_multiple.sh LIESTgrp06_$profile LIESTgrp06_$profile LIESTdev02 $profile
run/eval_bias_multiple.sh LIESTgrp06_$profile LIESTgrp06_$profile LIESTdev02 $profile --class_weight balanced
run/compare.sh LIESTgrp06 LIESTdev02 $profile $profile
run/compare.sh LIESTgrp06 LIESTdev02 $profile $profile --class_weight balanced
