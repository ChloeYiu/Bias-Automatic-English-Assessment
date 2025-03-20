#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

model=$1

run/extract_cav.sh LIESTgrp06 LIESTgrp06 $model grade_A 
run/extract_cav.sh LIESTgrp06 LIESTgrp06 $model grade_B2 
run/extract_cav.sh LIESTgrp06 LIESTgrp06 $model grade_C 
run/extract_cav.sh LIESTgrp06 LIESTgrp06 $model thai 
run/extract_cav.sh LIESTgrp06 LIESTgrp06 $model spanish 
run/extract_cav.sh LIESTgrp06 LIESTgrp06 $model young 
run/extract_cav.sh LIESTgrp06 LIESTgrp06 $model male 
run/extract_cav.sh LIESTgrp06 LIESTgrp06 $model grade_A  --class_weight balanced
run/extract_cav.sh LIESTgrp06 LIESTgrp06 $model grade_B2  --class_weight balanced
run/extract_cav.sh LIESTgrp06 LIESTgrp06 $model grade_C  --class_weight balanced
run/extract_cav.sh LIESTgrp06 LIESTgrp06 $model thai  --class_weight balanced
run/extract_cav.sh LIESTgrp06 LIESTgrp06 $model spanish  --class_weight balanced
run/extract_cav.sh LIESTgrp06 LIESTgrp06 $model young  --class_weight balanced
run/extract_cav.sh LIESTgrp06 LIESTgrp06 $model male  --class_weight balanced