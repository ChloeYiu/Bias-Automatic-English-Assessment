#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

local/run/run_eval_ensemble_with_hooks.sh --feature --part_range 1:1 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est LIESTgrp06
local/run/run_eval_ensemble_with_hooks.sh  --feature --part_range 1:1 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTdev02 LIESTdev02 est LIESTgrp06
local/run/run_eval_ensemble_with_hooks.sh  --feature --part_range 1:1 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTcal01 LIESTcal01 est LIESTgrp06
local/run/run_calibrate.sh --part_range 1:1 est LIESTgrp06_feature LIESTcal01
local/run/run_eval_calibrate.sh --part_range 1:1 est LIESTgrp06_feature LIESTcal01 LIESTdev02
local/run/run_post_activation.sh --part_range 1:1 --feature LIESTgrp06 est LIESTgrp06
local/run/run_post_activation.sh --part_range 1:1 --feature LIESTdev02 est LIESTgrp06
local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature grade_A
local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature grade_B2
local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature grade_C
local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature thai
local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature spanish
local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature young
local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature male
local/run/run_extract_cav.sh LIESTgrp06 est LIESTgrp06_feature grade_A
local/run/run_extract_cav.sh LIESTgrp06 est LIESTgrp06_feature grade_B2
local/run/run_extract_cav.sh LIESTgrp06 est LIESTgrp06_feature grade_C
local/run/run_extract_cav.sh LIESTgrp06 est LIESTgrp06_feature thai
local/run/run_extract_cav.sh LIESTgrp06 est LIESTgrp06_feature spanish
local/run/run_extract_cav.sh LIESTgrp06 est LIESTgrp06_feature young
local/run/run_extract_cav.sh LIESTgrp06 est LIESTgrp06_feature male
local/run/run_eval_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 grade_A
local/run/run_eval_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 grade_B2
local/run/run_eval_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 grade_C
local/run/run_eval_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 thai
local/run/run_eval_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 spanish
local/run/run_eval_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 young
local/run/run_eval_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 male
local/run/run_eval_cav.sh LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 grade_A
local/run/run_eval_cav.sh LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 grade_B2
local/run/run_eval_cav.sh LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 grade_C
local/run/run_eval_cav.sh LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 thai
local/run/run_eval_cav.sh LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 spanish
local/run/run_eval_cav.sh LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 young
local/run/run_eval_cav.sh LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 male
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 grade_A
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 grade_A
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 grade_B2
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 grade_C
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 thai
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 spanish
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 young
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 male
local/run/run_eval_cav_mean_std.sh LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 grade_A
local/run/run_eval_cav_mean_std.sh LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 grade_B2
local/run/run_eval_cav_mean_std.sh LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 grade_C
local/run/run_eval_cav_mean_std.sh LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 thai
local/run/run_eval_cav_mean_std.sh LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 spanish
local/run/run_eval_cav_mean_std.sh LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 young
local/run/run_eval_cav_mean_std.sh LIESTgrp06 est LIESTgrp06_feature LIESTgrp06 male
local/run/run_eval_bias_multiple.sh --class_weight balanced LIESTdev02 est LIESTgrp06_feature LIESTgrp06 thai
local/run/run_eval_bias_multiple.sh --class_weight balanced LIESTdev02 est LIESTgrp06_feature LIESTgrp06 spanish
local/run/run_eval_bias_multiple.sh --class_weight balanced LIESTdev02 est LIESTgrp06_feature LIESTgrp06 young
local/run/run_eval_bias_multiple.sh --class_weight balanced LIESTdev02 est LIESTgrp06_feature LIESTgrp06 male
local/run/run_eval_bias_multiple.sh --class_weight balanced LIESTdev02 est LIESTgrp06_feature LIESTgrp06 grade_A
local/run/run_eval_bias_multiple.sh --class_weight balanced LIESTdev02 est LIESTgrp06_feature LIESTgrp06 grade_B2
local/run/run_eval_bias_multiple.sh --class_weight balanced LIESTdev02 est LIESTgrp06_feature LIESTgrp06 grade_C
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06_feature LIESTgrp06 thai
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06_feature LIESTgrp06 spanish
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06_feature LIESTgrp06 young
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06_feature LIESTgrp06 male
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06_feature LIESTgrp06 grade_A
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06_feature LIESTgrp06 grade_B2
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06_feature LIESTgrp06 grade_C
local/run/run_eval_bias_all.sh LIESTdev02 est LIESTgrp06_feature LIESTgrp06 --title "Text with Feature"
local/run/run_eval_bias_all.sh LIESTdev02 est LIESTgrp06_feature LIESTgrp06 --class_weight balanced --title "Text with Feature"