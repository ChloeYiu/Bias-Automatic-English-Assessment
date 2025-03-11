#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

model=$1

run/eval_bias_multiple.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model grade_A
run/eval_bias_multiple.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model grade_B2
run/eval_bias_multiple.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model grade_C
run/eval_bias_multiple.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model thai
run/eval_bias_multiple.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model spanish
run/eval_bias_multiple.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model young
run/eval_bias_multiple.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model male
# run/eval_bias_multiple.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model grade_A --class_weight balanced
# run/eval_bias_multiple.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model grade_B2 --class_weight balanced
# run/eval_bias_multiple.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model grade_C --class_weight balanced
# run/eval_bias_multiple.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model thai --class_weight balanced
# run/eval_bias_multiple.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model spanish --class_weight balanced
# run/eval_bias_multiple.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model young --class_weight balanced
# run/eval_bias_multiple.sh LIESTgrp06 LIESTgrp06 LIESTdev02 $model male --class_weight balanced