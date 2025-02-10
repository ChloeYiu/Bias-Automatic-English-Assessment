#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

profile=$1

local/run/run_create_biased_score.sh /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est $profile
local/run/run_train_grader_with_bias.sh --part_range 1:1 --seed_range 1:5 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est $profile
