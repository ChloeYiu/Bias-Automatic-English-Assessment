#!/bin/bash
#$ -S /bin/bash

ALLARGS="$0 $@"
ARGS="$@"

profile=$1

run/create_biased_score.sh $profile
run/process_biased_data.sh $profile
run/train.sh LIESTgrp06_$profile LIESTdev02

