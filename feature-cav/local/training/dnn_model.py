#! /usr/bin/python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl

import lightning as pl
from lightning.pytorch.callbacks import Callback

import torchmetrics
from torchmetrics.functional.regression import pearson_corrcoef
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.init as init

class LitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.input_size = cfg.input_size
        self.n_hidden = cfg.n_hidden
        self.dropout = cfg.dropout
        self.n_hidden_layers = cfg.n_hidden_layers


        self.input_layer = nn.Linear(self.input_size, self.n_hidden)
        self.Dropout = nn.Dropout(p=self.dropout)

        #torch.nn.GELU
        self.hidden_layers = nn.ModuleList([nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden),
                nn.LeakyReLU(),
                nn.Dropout(p=self.dropout)) for _ in range(self.n_hidden_layers)])

        # Output layer
        self.output_layer = nn.Linear(self.n_hidden, 1)


        self._define_cost()

        self.L1_Loss = nn.L1Loss(reduction='mean')
        self.L2_Loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction="mean")
        self.save_hyperparameters(cfg)
        self.val_step_logs=[]
        self.init_weights()
        self.eps = 1e-8


        # self.var_lim=torch.tensor(cfg.var_lim)
        # self.log_var_lim=torch.log(self.var_lim).cuda() if os.environ.get('CUDA_VISIBLE_DEVICES') else torch.log(self.var_lim)
        self.max_grade=cfg.max_grade
        self.min_grade=cfg.min_grade


    def forward(self, x):
        x = F.leaky_relu(self.input_layer(x))
        x = self.Dropout(x)

        # Forward pass through hidden layers with dropout
        for layer in self.hidden_layers:
            x = layer(x)
        # Forward pass through output layer, just linear with 1d output for regression
        x = self.output_layer(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=self.hparams.lr_decay)

        #return optimizer
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}

    def training_step(self, batch, batch_idx):

        y_hat, loss = self.compute_forward_and_loss(batch, batch_idx)
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():

            x, y = batch
            y_hat, loss = self.compute_forward_and_loss(batch, batch_idx)
            self.log('val_loss_perstep', loss.item())
            self.val_step_logs.append({'val_loss_perstep' : loss.item(), 'y_pred' : y_hat, 'y_tgt' : y})
        return loss

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.validation_step(batch, batch_idx)
        return loss

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()

    def evaluation_step(self, batch, batch_idx):
        y_hat, loss = self.compute_forward_and_loss(batch, batch_idx)
        return y_hat, loss

    def compute_forward_and_loss(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self._compute_cost(y_hat, y)
        return y_hat, loss


    def evaluate(self, dataloader, activation_getter):
        # This method is used for evaluation
        self.eval()  # Set the model to evaluation mode
        activation_getter.add_hooks()

        predictions_list = []  # List to accumulate predictions
        y_tgt_list = []
        eval_utt_list = []
        #Iterate through the data loader to make predictions

        for batch in dataloader:
            x, y_tgt, utt_ids = batch
            x, y_tgt = x.unsqueeze(0), y_tgt.unsqueeze(0)

            predictions, loss = self.evaluation_step((x, y_tgt), 0)  # 0 is a placeholder for batch_idx
            predictions_list.append(predictions)
            y_tgt_list.append(y_tgt)
            eval_utt_list+=[utt_ids]

            for layer_name in activation_getter.layer_names:
                activations = activation_getter.activation_cache[layer_name]
                # Compute gradients for this sample
                grad = torch.autograd.grad(
                    outputs=predictions,
                    inputs=activations,
                    grad_outputs=torch.ones_like(predictions),
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True
                )
                activation_getter.store_tmp_gradient(grad)

        # Concatenate the predictions to create a single tensor
        predictions_tensor = torch.cat(predictions_list, dim=0).unsqueeze(1)
        tgt = torch.cat(y_tgt_list, dim=0)

        activation_getter.remove_hooks()

        return predictions_tensor, tgt, eval_utt_list

    def predict_step(self, x):
        with torch.no_grad():
            y_hat = self.forward(x)
            return y_hat

    def on_validation_epoch_end(self):

        val_lst = [ele['val_loss_perstep'] for ele in self.val_step_logs]
        val_loss = torch.mean(torch.tensor(val_lst))

        y_pred = torch.cat([ele['y_pred'] for ele in self.val_step_logs],dim=0)
        y_tgt = torch.cat([ele['y_tgt'] for ele in self.val_step_logs],dim=0)

        y_pred = y_pred[:, 0]
        y_pred = y_pred.unsqueeze(1)

        Estimated_Metrics = self.Compute_metrics(y_pred, y_tgt)

        self.log("epoch_loss_mse", Estimated_Metrics["epoch_loss_mse"])
        self.log("epoch_loss_PCC", Estimated_Metrics["epoch_loss_PCC"])
        self.log("epoch_loss_mae", Estimated_Metrics["epoch_loss_mae"])
        self.log("epoch_lt_0.5", Estimated_Metrics["epoch_lt_05"])
        self.log("epoch_lt_1", Estimated_Metrics["epoch_lt_1"])
        self.log("val_loss", val_loss)

        self.val_step_logs=[]


    def _define_cost(self):
        # define cost with MSE
        self.Loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction="mean")

    def _compute_cost(self, logits, targets):
        # compute cost with MSE
        loss=self.Loss_fn(logits, targets)
        return loss

    def compute_PCC(self, preds, targets):
        return pearson_corrcoef(preds, targets)


    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.normal_(module.weight, mean=0, std=0.001)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def Compute_metrics(self, y_pred, y_tgt):

        epoch_loss_mse = self.L2_Loss(y_pred, y_tgt)
        epoch_loss_PCC = pearson_corrcoef(y_pred, y_tgt)
        epoch_loss_mae = self.L1_Loss(y_pred, y_tgt)

        epoch_lt_05 = ((abs(y_tgt-y_pred) < 0.5)*1).sum()/y_tgt.shape[0]
        epoch_lt_1 = ((abs(y_tgt-y_pred) < 1)*1).sum()/y_tgt.shape[0]

        return {"epoch_loss_mse":epoch_loss_mse,
                "epoch_loss_PCC":epoch_loss_PCC,
                "epoch_loss_mae":epoch_loss_mae,
                "epoch_lt_05":epoch_lt_05,
                "epoch_lt_1":epoch_lt_1}


class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")

        #if pl_module.hparams.pruning:
        #    layer_type = (torch.nn.Linear, torch.nn.Conv1d)
        #    prune_model_global_unstructured(pl_module, trainer, layer_type, save_copy=True)
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


    def on_validation_epoch_end(self, pl_trainer, pl_module):
        print("Validation is done.")

        val_lst = [ele['val_loss_perstep'] for ele in outputs]
        self.log("val_loss",torch.mean(val_lst))

class LogToFileCallback(pl.Callback):
    def __init__(self, log_file_path):
        super().__init__()
        self.log_file_path = log_file_path


    def on_train_epoch_end(self, trainer, pl_module):
        print(f"epochno :{trainer.current_epoch}")
        log_metrics = trainer.callback_metrics
        log_line = [f"Epoch: {trainer.current_epoch}"] + [f"{key}: {value:.4f}" for key, value in log_metrics.items()]
        log_line = ' === '.join(log_line)



        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_line + '\n')