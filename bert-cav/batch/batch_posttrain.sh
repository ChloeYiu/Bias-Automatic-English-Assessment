#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

local/run/run_eval_ensemble_with_hooks.sh --part_range 1:1 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est LIESTgrp06
local/run/run_eval_ensemble_with_hooks.sh --part_range 1:1 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTdev02 LIESTdev02 est LIESTgrp06
local/run/run_post_activation.sh --part_range 1:1 LIESTgrp06 est LIESTgrp06
local/run/run_post_activation.sh --part_range 1:1 LIESTdev02 est LIESTgrp06
local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 grade_A
local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 grade_B2
local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 grade_C
local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 thai
local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 spanish
local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 young
local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 male
local/run/run_extract_cav.sh LIESTgrp06 est LIESTgrp06 grade_A
local/run/run_extract_cav.sh LIESTgrp06 est LIESTgrp06 grade_B2
local/run/run_extract_cav.sh LIESTgrp06 est LIESTgrp06 grade_C
local/run/run_extract_cav.sh LIESTgrp06 est LIESTgrp06 thai
local/run/run_extract_cav.sh LIESTgrp06 est LIESTgrp06 spanish
local/run/run_extract_cav.sh LIESTgrp06 est LIESTgrp06 young
local/run/run_extract_cav.sh LIESTgrp06 est LIESTgrp06 male
local/run/run_eval_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 grade_A
local/run/run_eval_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 grade_B2
local/run/run_eval_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 grade_C
local/run/run_eval_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 thai
local/run/run_eval_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 spanish
local/run/run_eval_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 young
local/run/run_eval_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 male
local/run/run_eval_cav.sh LIESTgrp06 est LIESTgrp06 LIESTgrp06 grade_A
local/run/run_eval_cav.sh LIESTgrp06 est LIESTgrp06 LIESTgrp06 grade_B2
local/run/run_eval_cav.sh LIESTgrp06 est LIESTgrp06 LIESTgrp06 grade_C
local/run/run_eval_cav.sh LIESTgrp06 est LIESTgrp06 LIESTgrp06 thai
local/run/run_eval_cav.sh LIESTgrp06 est LIESTgrp06 LIESTgrp06 spanish
local/run/run_eval_cav.sh LIESTgrp06 est LIESTgrp06 LIESTgrp06 young
local/run/run_eval_cav.sh LIESTgrp06 est LIESTgrp06 LIESTgrp06 male
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 grade_A
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 grade_A
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 grade_B2
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 grade_C
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 thai
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 spanish
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 young
local/run/run_eval_cav_mean_std.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 LIESTgrp06 male
local/run/run_eval_cav_mean_std.sh LIESTgrp06 est LIESTgrp06 LIESTgrp06 grade_A
local/run/run_eval_cav_mean_std.sh LIESTgrp06 est LIESTgrp06 LIESTgrp06 grade_B2
local/run/run_eval_cav_mean_std.sh LIESTgrp06 est LIESTgrp06 LIESTgrp06 grade_C
local/run/run_eval_cav_mean_std.sh LIESTgrp06 est LIESTgrp06 LIESTgrp06 thai
local/run/run_eval_cav_mean_std.sh LIESTgrp06 est LIESTgrp06 LIESTgrp06 spanish
local/run/run_eval_cav_mean_std.sh LIESTgrp06 est LIESTgrp06 LIESTgrp06 young
local/run/run_eval_cav_mean_std.sh LIESTgrp06 est LIESTgrp06 LIESTgrp06 male
local/run/run_eval_bias_multiple.sh --class_weight balanced LIESTdev02 est LIESTgrp06 LIESTgrp06 thai
local/run/run_eval_bias_multiple.sh --class_weight balanced LIESTdev02 est LIESTgrp06 LIESTgrp06 spanish
local/run/run_eval_bias_multiple.sh --class_weight balanced LIESTdev02 est LIESTgrp06 LIESTgrp06 young
local/run/run_eval_bias_multiple.sh --class_weight balanced LIESTdev02 est LIESTgrp06 LIESTgrp06 male
local/run/run_eval_bias_multiple.sh --class_weight balanced LIESTdev02 est LIESTgrp06 LIESTgrp06 grade_A
local/run/run_eval_bias_multiple.sh --class_weight balanced LIESTdev02 est LIESTgrp06 LIESTgrp06 grade_B2
local/run/run_eval_bias_multiple.sh --class_weight balanced LIESTdev02 est LIESTgrp06 LIESTgrp06 grade_C
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06 LIESTgrp06 thai
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06 LIESTgrp06 spanish
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06 LIESTgrp06 young
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06 LIESTgrp06 male
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06 LIESTgrp06 grade_A
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06 LIESTgrp06 grade_B2
local/run/run_eval_bias_multiple.sh LIESTdev02 est LIESTgrp06 LIESTgrp06 grade_C
local/run/run_eval_bias_all.sh LIESTdev02 est LIESTgrp06 LIESTgrp06 --title "Text"
local/run/run_eval_bias_all.sh LIESTdev02 est LIESTgrp06 LIESTgrp06 --class_weight balanced --title "Text"