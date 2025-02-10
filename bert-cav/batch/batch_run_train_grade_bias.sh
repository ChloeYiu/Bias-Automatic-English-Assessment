./local/run/run_create_biased_score.sh /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est grade_B2 && \
echo "Biased score created" && \
./local/run/run_train_grader_with_bias.sh --part_range 1:1 --seed_range 1:5 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est grade_B2 && \
./local/run/run_create_biased_score.sh /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est grade_A && \
./local/run/run_train_grader_with_bias.sh --part_range 1:1 --seed_range 1:5 /data/milsrg1/alta/linguaskill/relevance_v2/LIESTgrp06 LIESTgrp06 est grade_A && \