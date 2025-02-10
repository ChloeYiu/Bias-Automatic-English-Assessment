# get input size
awk '{print NF; exit}' data/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.SA/f4-text+ivconf+lnnormgdlm/data/features.txt

# data process
for part in SA SB SC SD SE; do
    python local/feature_extraction/process_data.py --data_dir ./data/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data
done

# train
nohup bash -c "
for part in SA SB SC SD SE; do
  for seed in 0 10 90; do
    python local/training/DDN_Trainers.py \
      --train_data ./data/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data \
      --dev_data ./data/GKTS4-D3-kmk-v2/rnnlm/LIESTcal01/grader.${part}/f4-text+ivconf+lnnormgdlm/data \
      --grader_seed ${seed} \
      --input_size 31
  done
done
" > output.log 2>&1 &

nohup bash -c "
for part in SC; do
  for seed in 0 10 90; do
    python local/training/DDN_Trainers.py \
      --train_data ./data/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data \
      --dev_data ./data/GKTS4-D3-kmk-v2/rnnlm/LIESTcal01/grader.${part}/f4-text+ivconf+lnnormgdlm/data \
      --grader_seed ${seed} \
      --input_size 31
  done
done
" > output.SC.log 2>&1 &

# evaluate - cal & dev
for part in SA SB SC SD SE; do
  for seed in 0 10 90; do
    python local/training/DDN_evaluate.py \
      --data_dir ./data/GKTS4-D3-kmk-v2/rnnlm/LIESTcal01/grader.${part}/f4-text+ivconf+lnnormgdlm/data \
      --model_dir ./DDN/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/DDN_${seed}
  done
done

# calibrate - cal
for part in SA SB SC SD SE; do
  for seed in 0 10 90; do
    python local/training/calibrate.py \
      --pred_file ./DDN/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/DDN_${seed}/LIESTcal01/LIESTcal01_pred_ref.txt
  done
done

# score - cal & dev*
for part in SA SB SC SD SE; do 
  for seed in 0 10 90; do 
    python local/training/score.py \
      --pred_file ./DDN/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/DDN_${seed}/LIESTdev02/LIESTdev02_pred_ref.txt \
      --calib_model ./DDN/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/DDN_${seed}/LIESTcal01/calib_model.pkl;
  done; 
done

# ensemble - cal & dev
for part in SA SB SC SD SE; do
  python local/training/Ensemble_scores.py \
    --ensemble_dir ./DDN/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data \
    --dataname LIESTcal01
done

# ensemble - calib & score
for part in SA SB SC SD SE; do
  python local/training/calibrate.py \
    --pred_file ./DDN/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/ens_LIESTcal01/LIESTcal01_pred_ref.txt
done

for part in SA SB SC SD SE; do
  python local/training/score.py \
    --pred_file ./DDN/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/ens_LIESTcal01/LIESTcal01_pred_ref.txt \
    --calib_model ./DDN/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/ens_LIESTcal01/calib_model.pkl
done

# ensemble - Compute stats for individual models - cal & dev
for part in SA SB SC SD SE; do
  python local/training/Get_chkpoints_stats.py  \
    --model_dir ./DDN/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data \
    --dataname LIESTcal01
done

# OVERALL
# average - cal & dev
for part in SA SB SC SD SE; do
  # Define the array of files for each dataset and seed
  files=($(ls ./DDN/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/ens_LIESTdedv03/LIESTdev03_pred_ref.txt))
  # Run the Python script with the ensemble files
  python local/training/Avg_ens_scores.py \
    --ensemble_files "${files[@]}" \
    --dataname LIESTdev03
done

files=($(ls ./DDN/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.{SA,SB,SC,SD,SE}/f4-text+ivconf+lnnormgdlm/data/ens_LIESTcal01/LIESTcal01_pred_ref.txt))
python local/training/Avg_ens_scores.py --ensemble_files ${files[@]} --dataname LIESTcal01

# average - calib & score
python local/training/calibrate.py --pred_file ./DDN/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/Avg_parts_LIESTcal01/LIESTcal01_pred_ref.txt

python local/training/score.py --pred_file ./DDN/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/Avg_parts_LIESTcal01/LIESTcal01_pred_ref.txt --calib_model ./DDN/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/Avg_parts_LIESTcal01/calib_model.pkl


DDN-MT:

# data process
for part in SA SB SC SD SE; do
  python local/training/compute_FA_transform.py --data_dir ./data/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.$part/f4-text+ivconf+lnnormgdlm/data --n_components 10
done

# train
nohup bash -c "
for part in SA SB SC SD SE; do
  for seed in 0 10 90; do
    python local/training/DDN-MT_Trainers.py \
      --train_data ./data/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data \
      --dev_data ./data/GKTS4-D3-kmk-v2/rnnlm/LIESTcal01/grader.${part}/f4-text+ivconf+lnnormgdlm/data \
      --grader_seed \$seed \
      --input_size 31 \
      --fa_model_path ./data/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/FA_transform.pkl
  done
done
" > output-MT.log 2>&1 &

# evaluate - cal & dev
for part in SA SB SC SD SE; do
  for seed in 0 10 90; do
    python local/training/DDN_evaluate.py \
      --data_dir ./data/GKTS4-D3-kmk-v2/rnnlm/LIESTdev03/grader.${part}/f4-text+ivconf+lnnormgdlm/data \
      --model_dir ./DDN-MT/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/DDN-MT_${seed}
  done
done

# calibrate - cal
for part in SA SB SC SD SE; do
  for seed in 0 10 90; do
    python local/training/calibrate.py \
      --pred_file ./DDN-MT/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/DDN-MT_${seed}/LIESTcal01/LIESTcal01_pred_ref.txt
  done
done

# score - cal & dev*
for part in SA SB SC SD SE; do 
  for seed in 0 10 90; do 
    python local/training/score.py \
      --pred_file ./DDN-MT/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/DDN-MT_${seed}/LIESTdev03/LIESTdev03_pred_ref.txt \
      --calib_model ./DDN-MT/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/DDN-MT_${seed}/LIESTcal01/calib_model.pkl;
  done; 
done

# ensemble - cal & dev
for part in SA SB SC SD SE; do
  python local/training/Ensemble_scores.py \
    --ensemble_dir ./DDN-MT/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data \
    --dataname LIESTcal01
done

# ensemble - calib & score
for part in SA SB SC SD SE; do
  python local/training/calibrate.py \
    --pred_file ./DDN-MT/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/ens_LIESTcal01/LIESTcal01_pred_ref.txt
done

for part in SA SB SC SD SE; do
  python local/training/score.py \
    --pred_file ./DDN-MT/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/ens_LIESTdev03/LIESTdev03_pred_ref.txt \
    --calib_model ./DDN-MT/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/ens_LIESTcal01/calib_model.pkl
done

# ensemble - Compute stats for individual models - cal & dev
for part in SA SB SC SD SE; do
  python local/training/Get_chkpoints_stats.py  \
    --model_dir ./DDN-MT/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data \
    --dataname LIESTcal01
done

# OVERALL
# average - cal & dev
for part in SA SB SC SD SE; do
  # Define the array of files for each dataset and seed
  files=($(ls ./DDN-MT/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.${part}/f4-text+ivconf+lnnormgdlm/data/ens_LIESTdev03/LIESTdev03_pred_ref.txt))
  # Run the Python script with the ensemble files
  python local/training/Avg_ens_scores.py \
    --ensemble_files "${files[@]}" \
    --dataname LIESTdev03
done

files=($(ls ./DDN-MT/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/grader.{SA,SB,SC,SD,SE}/f4-text+ivconf+lnnormgdlm/data/ens_LIESTcal01/LIESTcal01_pred_ref.txt))
python local/training/Avg_ens_scores.py --ensemble_files ${files[@]} --dataname LIESTcal01

# average - calib & score
python local/training/calibrate.py --pred_file ./DDN-MT/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/Avg_parts_LIESTcal01/LIESTcal01_pred_ref.txt

python local/training/score.py --pred_file ./DDN-MT/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/Avg_parts_LIESTcal01/LIESTcal01_pred_ref.txt --calib_model ./DDN/GKTS4-D3-kmk-v2/rnnlm/LIESTgrp06/Avg_parts_LIESTcal01/calib_model.pkl
