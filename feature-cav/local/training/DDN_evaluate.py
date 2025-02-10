#! /usr/bin/python

import warnings
warnings.filterwarnings("ignore")

import os
import sys

from pathlib import Path
import torch
import json

import yaml
from argparse import Namespace
from ddn_model import LitModel
from utils_Fns import process_data_file
from torch.utils.data import TensorDataset

import joblib

from cav import ActivationGetter

#--------------------------------------------------------
def Load_models_from_chkpt(wt_file):
        hparam_root = "/".join(str(wt_file).split('/')[:-1])
        hparam_file = os.path.join(hparam_root, 'lightning_logs/version_0/hparams.yaml')
        base_cfg = yaml.load(open(hparam_file), Loader=yaml.FullLoader)
        model = LitModel(base_cfg).load_from_checkpoint(wt_file)
        return model

#--------------------------------------------------------
def main(cfg):

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open('CMDs/DDN_evaluate.cmds', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')


    data_dir = Path(cfg.process_data_path, *Path(cfg.data_dir).parts[1:])
    #train_data = Path(cfg.process_data_path, *train_data_path.parts[1:])
    dataname = Path(data_dir).parent.parent.name # originally three parents, because we no longer use f4-text
    print('Data Name:', dataname)
    #

    working_dir=f"{cfg.model_dir}/{dataname}"
    Path(working_dir).mkdir(parents=True, exist_ok=True)
    logpath = Path(f'Logs/{working_dir}')

    logpath.mkdir(exist_ok=True, parents=True)
    file = Path(Path(sys.argv[0]).name).stem
    logging = open(os.path.join(logpath, file+'.log'),'w')

    ## Writes the argumnets used to create this file
    with open(os.path.join(working_dir, f"{file}_hparams.json"), 'w') as f:
        hparams = vars(cfg)
        json.dump(hparams, f, indent=4)
    # ---------------------------------------------------------
    ## below will be logic of the code



    ## Load the model from params file
    #---------------------------------------------
    hparam_root=Path(f"{cfg.model_dir}/lightning_logs")
    version_dirs = [d for d in hparam_root.iterdir() if d.is_dir() and d.name.startswith('version_')]
    latest_version_dir = sorted(version_dirs, key=lambda x: int(x.name.split('_')[-1]))[-1]
    hparam_file = os.path.join(str(latest_version_dir), 'hparams.yaml')

    base_cfg = yaml.load(open(hparam_file), Loader=yaml.FullLoader)
    loaded_cfg = Namespace(**base_cfg)

    wt_file=os.path.join(f"{cfg.model_dir}",'model.ckpt')

    model = LitModel(loaded_cfg)
    model_name = "_".join(Path(cfg.model_dir).parts[-2:])
    activation_dir = cfg.ACTIVATION_DIR
    gradient_dir = cfg.GRADIENT_DIR
    print(f"Model name: {model_name}")
    activation_getter = ActivationGetter(model, model_name, activation_dir, gradient_dir, ['input_layer'], 1)
    #state_dict = torch.load(wt_file)['state_dict']

    state_dict = torch.load(wt_file, map_location=torch.device('cpu'))
    state_dict=state_dict['state_dict']

    model.load_state_dict(state_dict)
    model.eval()

    #---------------------------------------------


    test_data=Path(data_dir, 'data.npy')
    dev_feat, dev_labels, dev_uttids = process_data_file(test_data)

    if loaded_cfg.whiten_features:
        scalar_path=Path(cfg.process_data_path, *Path(loaded_cfg.train_data).parts[1:])
        scalar_model=joblib.load(os.path.join(scalar_path, 'scaler.pkl'))
        dev_feat = scalar_model.transform(dev_feat)
    #--------------------------------------------------

    uttid_to_idx = {uttid: idx for idx, uttid in enumerate(dev_uttids)}
    idx_to_uttid = {idx: uttid for idx, uttid in enumerate(dev_uttids)}

    encoded_uttids = [uttid_to_idx[uttid] for uttid in dev_uttids]
    encoded_uttids = torch.tensor(encoded_uttids, dtype=torch.long)
    dev_dataset = TensorDataset(torch.tensor(dev_feat).float(), torch.tensor(dev_labels).float(), encoded_uttids)

    pred_mu, pred_log_std, tgt_mu, eval_uttlist = model.evaluate(dev_dataset, activation_getter)


    with open(os.path.join(working_dir, f'{dataname}_pred_ref.txt'), 'w') as pred_ref, \
            open(os.path.join(working_dir, f'{dataname}_pred_std.txt'), 'w') as pred_std :

        pred_ref.write("uttid pred_mu tgt_mu\n")
        pred_std.write("uttid pred_mu log_std\n")

        for pmu, pstd, utt_idx, tmu in zip(pred_mu, pred_log_std, eval_uttlist, tgt_mu):

            pmu, pstd, tmu = pmu.item(), pstd.item(), tmu.item()
            utt=idx_to_uttid[utt_idx.item()]

            line = f"{utt} {pmu} {tmu}\n"
            pred_ref.write(line)

            std_line = f"{utt} {pmu} {pstd}\n"
            pred_std.write(std_line)

    
    # Save the activations for each layer
    activation_getter.store_activations(dev_uttids)
    activation_getter.store_gradients(dev_uttids)


if __name__ == '__main__':
    import sys
    import os
    import yaml
    import argparse

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Configuration for DDN training.')
    parser.add_argument('--process_data_path', type=str, default='processed_data',help=' ')
    parser.add_argument('--data_dir', type=str, default='', help='Path to development data')
    parser.add_argument('--model_dir', type=str, help='Working root directory')
    parser.add_argument('--ACTIVATION_DIR', type=str, help='directory to store activations')
    parser.add_argument('--GRADIENT_DIR', type=str, help='directory to store gradients')
    #parser.add_argument('--dataname', type=str, required=True, help='dataname')
    cfg = parser.parse_args()
    main(cfg)