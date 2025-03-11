#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

model=$1

run/eval_cav.sh  LIESTgrp06 LIESTgrp06 LIESTgrp06 $model grade_A 
run/eval_cav.sh  LIESTgrp06 LIESTgrp06 LIESTgrp06 $model grade_B2
run/eval_cav.sh  LIESTgrp06 LIESTgrp06 LIESTgrp06 $model grade_C
run/eval_cav.sh  LIESTgrp06 LIESTgrp06 LIESTgrp06 $model thai
run/eval_cav.sh  LIESTgrp06 LIESTgrp06 LIESTgrp06 $model spanish
run/eval_cav.sh  LIESTgrp06 LIESTgrp06 LIESTgrp06 $model young
run/eval_cav.sh  LIESTgrp06 LIESTgrp06 LIESTgrp06 $model male
run/eval_cav.sh  LIESTgrp06 LIESTgrp06 LIESTgrp06 $model grade_A --class_weight balanced
run/eval_cav.sh  LIESTgrp06 LIESTgrp06 LIESTgrp06 $model grade_B2 --class_weight balanced
run/eval_cav.sh  LIESTgrp06 LIESTgrp06 LIESTgrp06 $model grade_C --class_weight balanced
run/eval_cav.sh  LIESTgrp06 LIESTgrp06 LIESTgrp06 $model thai --class_weight balanced
run/eval_cav.sh  LIESTgrp06 LIESTgrp06 LIESTgrp06 $model spanish --class_weight balanced
run/eval_cav.sh  LIESTgrp06 LIESTgrp06 LIESTgrp06 $model young --class_weight balanced
run/eval_cav.sh  LIESTgrp06 LIESTgrp06 LIESTgrp06 $model male --class_weight balanced