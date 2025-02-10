./local/run/run_extract_cav.sh --class_weight balanced LIESTgrp06 est LIESTgrp06 young && \
./local/run/run_extract_cav.sh LIESTgrp06 est LIESTgrp06 young && \
./local/run/run_eval_cav.sh --class_weight balanced LIESTdev02 est LIESTgrp06 LIESTgrp06 young && \
./local/run/run_eval_cav.sh LIESTdev02 est LIESTgrp06 LIESTgrp06 young