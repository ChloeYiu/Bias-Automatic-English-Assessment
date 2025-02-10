./local/run/run_eval_ensemble_with_hooks.sh --part_range 1:1 --biased_TSET grade_A /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est LIESTgrp06_grade_A && \
./local/run/run_eval_ensemble_with_hooks.sh --part_range 1:1 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTdev02 LIESTdev02 est LIESTgrp06_grade_A && \ 
./local/run/run_extract_cav.sh LIESTgrp06_grade_A est LIESTgrp06_grade_A grade_A && \
./local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06_grade_A est LIESTgrp06_grade_A grade_A && \
./local/run/run_eval_cav.sh LIESTdev02 est LIESTgrp06_grade_A LIESTgrp06_grade_A grade_A && \
./local/run/run_eval_cav.sh --class_weight balanced LIESTdev02 est LIESTgrp06_grade_A LIESTgrp06_grade_A grade_A && \
./local/run/run_eval_bias.sh LIESTdev02 est LIESTgrp06_grade_A LIESTgrp06_grade_A grade_A && \
./local/run/run_eval_bias.sh --class_weight balanced LIESTdev02 est LIESTgrp06_grade_A LIESTgrp06_grade_A grade_A 



