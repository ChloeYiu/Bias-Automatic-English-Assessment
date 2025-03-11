#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

model=$1

run/extract_cav.sh LIESTgrp06 LIESTgrp06 grade_A $model
run/extract_cav.sh LIESTgrp06 LIESTgrp06 grade_B2 $model
run/extract_cav.sh LIESTgrp06 LIESTgrp06 grade_C $model
run/extract_cav.sh LIESTgrp06 LIESTgrp06 thai $model
run/extract_cav.sh LIESTgrp06 LIESTgrp06 spanish $model
run/extract_cav.sh LIESTgrp06 LIESTgrp06 young $model
run/extract_cav.sh LIESTgrp06 LIESTgrp06 male $model
run/extract_cav.sh LIESTgrp06 LIESTgrp06 grade_A $model --class_weight balanced
run/extract_cav.sh LIESTgrp06 LIESTgrp06 grade_B2 $model --class_weight balanced
run/extract_cav.sh LIESTgrp06 LIESTgrp06 grade_C $model --class_weight balanced
run/extract_cav.sh LIESTgrp06 LIESTgrp06 thai $model --class_weight balanced
run/extract_cav.sh LIESTgrp06 LIESTgrp06 spanish $model --class_weight balanced
run/extract_cav.sh LIESTgrp06 LIESTgrp06 young $model --class_weight balanced
run/extract_cav.sh LIESTgrp06 LIESTgrp06 male $model --class_weight balanced