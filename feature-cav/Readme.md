
# Graders
This project focuses on developing graders.

## Project Directory Structure
- **CMDs**: This directory is used for logging the commands that are executed.
- **data**: Contains input data; the structure within this folder mirrors the organization of the models.
- **DDN**: This folder is created by scripts and stores the trained DDN models.
- **local**: Contains scripts utilized by the model for training graders.
- **Logs**: Contains log files generated during the execution of scripts, tracking the progress of the current run.
- **processed_data**: Stores the processed data.
- **README.md**
- **run**

## Usage
## Easy usage instructions:

1. Clone the GitHub repository:
    - git clone https://github.com/harivydana/Graders.git

2. Setting up environments:
    - ```cd Graders/ ```
    - Install the suitable miniconda environment here using the following commands:
        - Download [miniconda](https://repo.anaconda.com/miniconda/Miniconda3-py311_24.5.0-0-Linux-x86_64.sh)
        - Install the miniconda version using this command ``bash Miniconda3-py311_24.5.0-0-Linux-x86_64.sh`` This install ``python 3.11`` and the required packages.
        - Under miniconda and install conda environment using the below commands and it will install torch and all the other tools to run whisper models.
        ```bash
            source ${miniconda_path}/etc/profile.d/conda.sh
            # conda_env_path is a dir to install conda environment,
            # The path defines the name of the environment
            conda env create -f whisper_environment.yaml --prefix ${conda_env_path}/envs/altagraders
        ```
    - For ALTA users you can directly access the environemnt from the path.
                    conda activate /scratches/dialfs/alta/hkv21/mconda_setup/miniconda_py3.11/envs/pytorch2.0




## Data
    The data used for training a grader consists of features and grades. The features are stored in a text file where the first column contains recording IDs, followed by multiple feature columns. Similarly, the grades file contains two columns: the first column includes recording IDs, and the second column contains the corresponding grades. Below are examples of the data structure.

### Example Directory Structure
The data directory is organized as follows

**Data Folder Structure:**

                    ```
                    data/GKTS4-D3/rnnlm/LIESTtrn04/
                    ├── grader.SA
                    │   └── f4-text
                    │       └── data
                    │           ├── features.txt
                    │           └── grades.txt
                    ├── grader.SB
                    ├── grader.SC
                    ├── grader.SD
                    └── grader.SE
                    ```


We recommend organizing the data with the following structure:

- `data/GKTS4-D3/rnnlm/${train_set}/grader.${part}/f4-text/data/features.txt`
- `data/GKTS4-D3/rnnlm/${calib_set}/grader.${part}/f4-text/data/features.txt`
- `data/GKTS4-D3/rnnlm/${test_set}/grader.${part}/f4-text/data/features.txt`

This structure should be maintained consistently for the training, calibration, and test sets.

```plaintext
...
CEL11U-00016 2.042203 2.133505 1.019918 .... .... .... .... 1.664565 -0.000004
CEL11U-00018 0.530644 0.248533 0.439956 .... .... .... .... 0.219640 0.000010
CEL11U-00024 0.555984 0.260841 0.520007 .... .... .... .... 0.202155 -0.000069
```

data/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data/grades.txt
```plaintext
...
CEL11U-00016    3
CEL11U-00018    4
CEL11U-00024    4
...
```


## Steps Train & Evaluate Graders

## Data Processing Instructions
Set your directory paths as follows:

```bash
train_dir="./data/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data"
calib_dir="./data/GKTS4-D3/rnnlm/LIESTcal01/grader.SA/f4-text/data"
test_dir="./data/GKTS4-D3/rnnlm/LIALTtst02/grader.SA/f4-text/data"

train="LIESTtrn04"
calib_dir="LIESTcal01"
test_dir="LIALTtst02"
```


1. Process the Data
    Process individual directories:
    ```bash
        python local/feature_extraction/process_data.py --data_dir ${train_dir}
        python local/feature_extraction/compute_whitening_transform.py --data_dir ${train_dir}
        python local/feature_extraction/process_data.py --data_dir ${calib_dir}
        python local/feature_extraction/process_data.py --data_dir ${test_dir}
    ```
    After running the above comands, you will create a preporcessed data dir, and the directory structure will be as shown below: processed_data/

```
├── GKTS4-D3/
│   └── rnnlm/
│       ├── LIESTtrn04/  # Training data set
│       │   ├── grader.SA/
│       │   │   └── f4-text/
│       │   │       └── data/
│       │   │           ├── data.npy  # Processed features
│       │   │           └── scaler.pkl  # Scaler information (optional)
│       │   ├── ...          # Similar structure for grader.SB, grader.SC, etc.
│       ├── LIESTcal01/  # Calibration data set (similar structure)
│       └── LIALTtst02/  # Test data set (similar structure)
```

In the above described data structure the following files will be created
        - *`processed_data/`*: Root directory containing processed data.
        - *`data.npy`*: File containing the processed features in NumPy format.
        - *`scaler.pkl`* (optional): File has the mean and variance estimated for normalizing the data.



2. Training the DDN models:
    - The preprocessed data is used for training the DDN models, as demonstrated below. The following example shows how to initiate model training:

        ```bash
        python local/training/DDN_Trainers.py --train_data ${train_dir} --dev_data ${calib_dir} --grader_seed 0
        ```

    - You can access all the available options for training the DDN networks by running the following command

        ```bash
            python local/training/DDN_Trainers.py  --help
        ```



Upon completing the training, the model will be saved in the directory `./DDN/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data/DDN_0`. Within this folder, the following files will be created:

- **log.txt**: Contains the log generated during the training process.
- **model.ckpt**: This file stores the model checkpoint.
- **lightning_logs**: This directory includes the file `version_0/hparams.yaml`, which records the hyperparameters used during training. This information can be utilized for future model evaluation.

- **A snippet for training multiple seeds and parts can be found in the Tutorial 2 section below.**



3. Assess the Models

    - To evaluate the model, use the script below:

        ```bash
        model_dir="./DDN/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data/DDN_0"

        python local/training/DDN_evaluate.py --data_dir ${calib_dir} --model_dir ${model_dir}
        python local/training/DDN_evaluate.py --data_dir ${test_dir} --model_dir ${model_dir}
        ```

    After evaluating the model, two files will be created:
    - `./DDN/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data/DDN_0/LIESTcal01/LIESTcal01_pred_ref.txt`

    - `./DDN/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data/DDN_0/LIESTcal01/LIESTcal01_pred_std.txt`

    **LIESTcal01_pred_ref.txt**: This file will contain three columns: `recording-id`, `predicted-grade`, `reference-grade`

    **LIESTcal01_pred_std.txt**: This file will have three columns: `recording-id`, `predicted-grade-mean`, `predicted-grade-std`



4. Calibration Model
    A calibration model is a linear regression model trained to calibrate the predicted scores.

    -   To achieve this, the calibration file takes `./DDN/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data/DDN_0/LIESTcal01/LIESTcal01_pred_ref.txt` as input to estimate the parameters for the calibration model. The trained model will be saved in `./DDN/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data/DDN_0/LIESTcal01/calib_model.pkl`, and the same model parameters will also be saved in text format at `./DDN/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data/DDN_0/LIESTcal01/calib_model.txt`.

    ```bash
    calib_set_name=$(basename $(dirname $(dirname $(dirname "$calib_dir"))))
    python local/training/calibrate.py --pred_file ${model_dir}/${calib_set_name}/${calib_set_name}_pred_ref.txt
    ```






5. Score the model:
    This scrit takes ```pred_file``` and ```calibration model ``` and computes the metrics shown below, The metrics are computed for both uncalibrated and calibrated scores.

    ```bash
    test_set_name=$(basename $(dirname $(dirname $(dirname "$test_dir"))))
    python local/training/score.py --pred_file ${model_dir}/${test_set_name}/${test_set_name}_pred_ref.txt --calib_model ${model_dir}/${calib_set_name}/calib_model.pkl
    python local/training/score.py --pred_file ${model_dir}/${calib_set_name}/${calib_set_name}_pred_ref.txt --calib_model ${model_dir}/${calib_set_name}/calib_model.pkl
    ```

    This script will generate two file ``` calib_results.json ``` and ```uncalib_results.json ``` in the below paths
    - ./DDN/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data/DDN_0/LIESTcal01/calib_results.json
    - ./DDN/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data/DDN_0/LIESTcal01/uncalib_results.json

    Each of the above file will have the format as shown below:
    - calib_results.json:


                {
                "MSE": 0.452,
                "PCC": 74.493,
                "MAE": 0.537,
                "lt_0.5": 53.004,
                "lt_1": 85.853,
                "RMSE": 0.672
                }

6. Creating and ensemble and scoring

    - The following sequence of steps will generate ensemble predictions, calibrate them, and compute scores. This script calculates ensemble scores by averaging the predictions from all seeds, namely   `DDN_0`, `DDN_1`, ..., `DDN_10`. The average score is then recorded in the output.


    ```bash
    model_dir="./DDN/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data"
    calib_set_name=$(basename $(dirname $(dirname $(dirname "$calib_dir"))))

    python local/training/Ensemble_scores.py --ensemble_dir ${model_dir} --dataname ${calib_set_name}
    python local/training/calibrate.py --pred_file ${model_dir}/ens_${calib_set_name}/${calib_set_name}_pred_ref.txt
    python local/training/score.py --pred_file ${model_dir}/ens_${calib_set_name}/${calib_set_name}_pred_ref.txt --calib_model ${model_dir}/ens_${calib_set_name}/calib_model.pkl
    ```

    - After executing these commands, you will create the file `./DDN/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data/ens_LIESTcal01/LIESTcal01_pred_ref.txt`, which contains the prediction scores averaged across all seeds. You can follow similar steps for the test set. Following these steps, you will generate the files `calib_results.json` and `uncalib_results.json`, which will contain the results as shown above.



8. Compute stats for individual models:
    - **```run this only after scoring all the seeds```**

    - This will give the statiscits of the individual seeds to compare against the ensemble:

    ``` bash
    model_dir="./DDN/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data"
    calib_set_name=$(basename $(dirname $(dirname $(dirname "$calib_dir"))))
    python local/training/Get_chkpoints_stats.py --model_dir ${model_dir} --dataname ${calib_set_name}
    ```

    After running the above you will generate the table as shown below:

        ``` plaintext
            +--------+--------+-------+
            |        |   Mean |   Std |
            +========+========+=======+
            | MAE    |  0.539 | 0.005 |
            +--------+--------+-------+
            | MSE    |  0.46  | 0.019 |
            +--------+--------+-------+
            | PCC    | 75.809 | 1.987 |
            +--------+--------+-------+
            | RMSE   |  0.678 | 0.013 |
            +--------+--------+-------+
            | lt_0.5 | 53.501 | 0.497 |
            +--------+--------+-------+
            | lt_1   | 86.112 | 0.316 |
            +--------+--------+-------+
        ```


7. Compute overall all prediction and Score.
    The overall prediction is calculated as the average of all parts, derived from averaging all the uncalibrated ensembled scores. This is achieved using the following script:
    They are computed by the below script:
    ``` bash
        files=($(ls ./DDN/GKTS4-D3/rnnlm/LIESTtrn04/grader.{SA,SB,SC,SD,SE}/f4-text/data/ens_${calib_set_name}/${calib_set_name}_pred_ref.txt))
        calib_set_name=$(basename $(dirname $(dirname $(dirname "$calib_dir"))))
        python /scratches/dialfs/alta/hkv21/Graders_github/Graders/local/training/Avg_ens_scores.py --ensemble_files ${files[@]} --dataname ${calib_set_name}
    ```





## Tutorial-2

In this section, we will execute the same sequence of steps as above, but we will run them concurrently for multiple parts. The following snippets are useful for executing this:

1. **Feature Processing:** The commands below can be used to extract features for all parts.
- Process data all parts (SA, SB, SC, SD, SE):
- This script is also a recipie

- Inside the script set the following variables

``` bash
# path_template: This variable defines a template for the file path, with placeholders for the set name (setname) and part (part). The placeholders will be dynamically replaced as we iterate through different sets and parts.

# parts: This array contains different parts (SA, SB, SC, SD, SE), which represent different data subsets. The script will iterate over these parts to generate the correct paths for each combination of set name and part.

path_template="./data/GKTS4-D3/rnnlm/\${setname}/grader.\${part}/f4-text/data"
parts=(SA SB SC SD SE)
train_set="LIESTtrn04"
calib_set="LIESTcal01"
test_set="LIALTtst02"
```



```bash
    bash run/process_data.sh Logs/porcesss_data.log 2>&1
```



2. Training and evaluate, ensmble, DDN models for all parts

- This script also computes overall score for the DDN models:
- This script is also a recipie
- You have to set the same varibles as shown in ```Feature Processing```.

```bash
    bash run/process_data.sh Logs/porcesss_data.log 2>&1
```



1. Results:
    - all the results are calibrated, all are computed on ```LIALTtst02```
    - single models :-

            +----+----------------+---------------+----------------+----------------+
            |    | PCC            | RMSE          | lt_0.5         | lt_1           |
            +====+===============+===============+================+=================+
            | SA | 79.041 ± 0.206 | 0.673 ± 0.003 | 55.983 ± 0.736 | 87.423 ± 0.184 |
            | SB | 67.035 ± 0.293 | 0.822 ± 0.010 | 45.939 ± 0.818 | 79.476 ± 0.797 |
            | SC | 82.153 ± 0.083 | 0.673 ± 0.001 | 54.716 ± 0.684 | 86.856 ± 0.382 |
            | SD | 75.810 ± 0.091 | 0.724 ± 0.002 | 58.821 ± 0.547 | 84.235 ± 0.138 |
            | SE | 85.837 ± 0.054 | 0.661 ± 0.001 | 55.240 ± 0.691 | 87.642 ± 0.360 |
            +----+----------------+---------------+----------------+----------------+

    - Ensemble per part:-

            +-----+--------+--------+----------+--------+
            |     |    PCC |   RMSE |   lt_0.5 |   lt_1 |
            +=====+========+========+==========+========+
            |  SA | 79.060 |  0.673 |   56.332 | 87.336 |
            |  SB | 67.085 |  0.822 |   44.978 | 79.476 |
            |  SC | 82.168 |  0.673 |   55.022 | 86.900 |
            |  SD | 75.826 |  0.724 |   59.389 | 84.279 |
            |  SE | 85.844 |  0.661 |   55.022 | 87.773 |
            +-----+--------+--------+----------+--------+

    - overall score:

            +----------+--------+--------+----------+--------+
            |          |    PCC |   RMSE |   lt_0.5 |   lt_1 |
            +==========+========+========+==========+========+
            |  overall | 89.802 |  0.466 |   74.672 | 96.507 |
            +----------+--------+--------+----------+--------+





### Training DDN-MT Graders:
1. Feature Processing:
    - The feature procerssing is same as the DDN grader, except for one additional ```component model```.
    - you can use the processsed_data folder for training DDN-MT Grader.

```bash
        python local/training/compute_FA_transform.py --data_dir ${data_dir} --n_components 10
```

- This will create ```processed_data/GKTS4-D3/rnnlm/LIESTtrn04/grader.SA/f4-text/data/FA_transform.pkl``` file, which is used while training the model.


2. Training the DDN-MT models:
    - The preprocessed data is used for training the DDN-MT models, as demonstrated below. The following example shows how to initiate model training:

```bash
python local/training/DDN-MT_Trainers.py --train_data ${train_dir} --dev_data ${calib_dir} --grader_seed ${grader_seed} --fa_model_path ${train_dir}/FA_transform.pkl
```

- You can access all the available options for training the DDN-MT networks by running the following command

```bash
    python local/training/DDN-MT_Trainers.py  --help
```


3. Results:

    - DDN vs. DDN-MT (overall score):

            +----------+--------+--------+----------+--------+
            |          |    PCC |   RMSE |   lt_0.5 |   lt_1 |
            +==========+========+========+==========+========+
            |   DDN    | 89.802 |  0.466 |   74.672 | 96.507 |
            |  DDN-MT  | 89.721 |  0.464 |   74.672 | 96.943 |
            +----------+--------+--------+----------+--------+