#! /usr/bin/python

import warnings
warnings.filterwarnings("ignore")
import os
from pathlib import Path
import sys
import torch
import numpy as np
import joblib
import json

from torch.utils.data import TensorDataset, DataLoader
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch import loggers as pl_loggers
from model import LitModel, LitBertModel, LitDNNModel, LogToFileCallback
from utils_Fns import process_data_file


def Trainer(cfg):
    #set seeds for torch
    ## if you dont give grader seed value seperately then use name as the seed
    #grader_seed = cfg.grader_seed if cfg.grader_seed else int(cfg.grader_seed_name)

    model_type = cfg.model_type

    torch.manual_seed(cfg.grader_seed)
    pl.pytorch.seed_everything(cfg.grader_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.grader_seed)
        torch.cuda.manual_seed_all(cfg.grader_seed)


    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')

    with open(f'CMDs/{model_type}_Trainers.cmds', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')


    train_data_path = Path(cfg.train_data)
    dev_data_path = Path(cfg.dev_data)

    train_data = Path(cfg.process_data_path, *train_data_path.parts[1:])
    dev_data = Path(cfg.process_data_path, *dev_data_path.parts[1:])

    working_root_dir = Path(model_type, *train_data.parts[1:])
    working_dir = os.path.join(working_root_dir,f'{model_type}_'+ str(cfg.grader_seed))

    #working_dir = cfg.working_root_dir
    output_path = Path(working_dir)
    logpath = Path("Logs/{}".format(working_dir))

    output_path.mkdir(exist_ok=True, parents=True)
    logpath.mkdir(exist_ok=True, parents=True)

    file = Path(Path(sys.argv[0]).name).stem
    logging = open(os.path.join(logpath, file+'.log'),'w')

    ## Writes the argumnets used to create this file
    with open(os.path.join(output_path, "{}_hparams.json".format(file)), 'w') as f:
        hparams = vars(cfg)
        json.dump(hparams, f, indent=4)
    # ---------------------------------------------------------
    ## below will be logic of the code


    # Load the data
    train_feat, train_labels,_ = process_data_file(os.path.join(train_data,'data.npy'))
    dev_feat, dev_labels,_ = process_data_file(os.path.join(dev_data,'data.npy'))

    if cfg.whiten_features:
        scalar_model=joblib.load(os.path.join(train_data, 'scaler.pkl'))

        train_feat = scalar_model.transform(train_feat)
        dev_feat = scalar_model.transform(dev_feat)


    #create train and dev data loaders
    train_dataset = TensorDataset(torch.Tensor(train_feat).float(), torch.Tensor(train_labels).float())
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    dev_dataset = TensorDataset(torch.Tensor(dev_feat).float(), torch.Tensor(dev_labels).float())
    dev_loader = DataLoader(dev_dataset, batch_size=cfg.batch_size, shuffle=False)

    if model_type=="DDN_BERT":
        print("Using DDN BERT model")
        model = LitBertModel(cfg)
    elif model_type=="DNN":
        print("Using DNN model")
        model = LitDNNModel(cfg)
    else:
        print("Using normal model")
        model = LitModel(cfg)    
    ###############################################################
    checkpoint_callback = ModelCheckpoint(monitor = cfg.checkpoint_monitor,
                                            dirpath = working_dir,
                                            save_top_k = cfg.save_top_k,
                                            filename = cfg.checkpoint_filename)
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR='_'
    checkpoint_callback.CHECKPOINT_JOIN_CHAR="_"

    ###loggers
    tb_logger = pl_loggers.TensorBoardLogger(working_dir)
    checkpoint_earlystoping = EarlyStopping(monitor = cfg.monitor,
                                            min_delta = cfg.min_delta,
                                            patience = cfg.patience,
                                            verbose = False,
                                            mode = cfg.mode,
                                            strict = True,
                                            check_finite = True,
                                            stopping_threshold = None,
                                            divergence_threshold = None,
                                            check_on_train_epoch_end = None)

    callbacks=[]
    callbacks.append(checkpoint_callback)
    callbacks.append(LogToFileCallback(os.path.join(working_dir,'log.txt')))
    if cfg.earlystopping_flag:
        callbacks.append(checkpoint_earlystoping)

    trainer =  pl.Trainer(default_root_dir = working_dir,
                            max_epochs = cfg.max_epochs,
                            accelerator = cfg.accelerator,
                            strategy = cfg.strategy,
                            num_nodes = 1,
                            logger = [tb_logger],
                            log_every_n_steps = 1,
                            val_check_interval = cfg.validate_interval,
                            reload_dataloaders_every_n_epochs = 5,
                            gradient_clip_val = cfg.gradient_clip_val,
                            precision = cfg.precision,
                            callbacks = callbacks,
                            accumulate_grad_batches = cfg.accumulate_grad_batches, enable_progress_bar=False)
    trainer.fit(model, train_loader, dev_loader)



if __name__ == '__main__':

    import sys
    import os

    import argparse

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Configuration for DDN training.')

    # Experiment configuration
    parser.add_argument('--model_type', type=str, required=True, help='Type of DDN using either "DDN" or "DNN_BERT"')

    # Data and experiment directories
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--dev_data', type=str, required=True, help='Path to development data')
    parser.add_argument('--process_data_path', type=str, default='processed_data',help=' ')

    parser.add_argument('--whiten_features', type=str, default=True, help='Path to training data')

    parser.add_argument('--working_root_dir', type=str, help='Working root directory')


    # Grader manual seeds
    parser.add_argument('--grader_seed', type=int, default=None, help='List of manual seeds as integers')
    #parser.add_argument('--grader_seed_name', type=int, default=750, help='List of manual seeds as integers')


    # Data loader configuration
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for data loading')

    # Early stopping configuration
    parser.add_argument('--earlystopping_flag', type=bool, default=False, help='Flag for early stopping')
    parser.add_argument('--monitor', type=str, default="epoch_loss_mse", help='Metric to monitor for early stopping')
    parser.add_argument('--min_delta', type=float, default=0, help='Minimum change in monitored quantity to qualify as an improvement')
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--mode', type=str, default="min", help='Mode for monitoring (min or max)')

    # Model checkpoint configuration
    parser.add_argument('--checkpoint_monitor', type=str, default="epoch_loss_mse", help='Metric to monitor for checkpoints')
    parser.add_argument('--save_top_k', type=int, default=1, help='Save the top K models')
    parser.add_argument('--checkpoint_mode', type=str, default="min", help='Mode for monitoring checkpoint (min or max)')
    parser.add_argument('--checkpoint_filename', type=str, default="model", help='Filename template for the model checkpoint')

    # Checkpoint path
    parser.add_argument('--ckpt_path', type=str, default="None", help='Path to the checkpoint')

    # Model configuration
    parser.add_argument('--input_size', type=int, default=24, help='Input size of the model')
    parser.add_argument('--n_hidden', type=int, default=185, help='Number of hidden units')
    parser.add_argument('--hidden_1', type=int, default=600, help='Number of hidden units')
    parser.add_argument('--hidden_2', type=int, default=20, help='Number of hidden units')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--n_hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--loss_fn_type', type=str, default="NLL_MVN", help='Loss function type')
    parser.add_argument('--torch_manual_seed', type=int, default=10, help='Manual seed for PyTorch')
    parser.add_argument('--elementwise_affine', type=bool, default=True, help='Elementwise affine in batch normalization')

    # Trainer configuration
    parser.add_argument('--max_epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--num_gpus', type=int, default=0, help='Number of GPUs to use')
    parser.add_argument('--accelerator', type=str, default="cpu", help='Accelerator type')
    parser.add_argument('--strategy', type=str, default="ddp", help='Training strategy')
    parser.add_argument('--validate_interval', type=float, default=1.0, help='Validation interval')
    parser.add_argument('--gradient_clip_val', type=float, default=10, help='Value for gradient clipping')
    parser.add_argument('--precision', type=int, default=32, help='Precision for training')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Number of batches to accumulate gradients over')

    # Optimizer configuration
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0000, help='Weight decay for the optimizer')
    parser.add_argument('--lr_decay', type=float, default=1, help='Learning rate decay factor')

    parser.add_argument('--var_lim', type=float, default=10, help='Value to clamp the variance')
    parser.add_argument('--min_grade', type=float, default=0, help='Value to clamp the min grade')
    parser.add_argument('--max_grade', type=float, default=6, help='Value to clamp the max grade')

    # Parse the arguments
    cfg = parser.parse_args()

    # Print the parsed arguments
    print(cfg)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    cfg.accelerator='gpu' if cuda_visible_devices else 'cpu'

    Trainer(cfg)
