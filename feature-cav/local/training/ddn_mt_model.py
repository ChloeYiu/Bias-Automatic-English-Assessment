#! /usr/bin/python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl

import lightning as pl
import joblib

from torch.distributions import Normal, LogNormal, kl_divergence
torch.distributions.kl_divergence
from ddn_model import LitModel as LitModel_base
from pathlib import Path
import torch.nn.init as init
from torch.optim.lr_scheduler import ExponentialLR


class LitModel(LitModel_base):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.input_size = cfg.input_size
        self.n_hidden = cfg.n_hidden
        self.dropout = cfg.dropout
        self.n_hidden_layers = cfg.n_hidden_layers

        self.input_layer = nn.Linear(self.input_size, self.n_hidden)
        self.Dropout = nn.Dropout(p=self.dropout)

        self.hidden_layers = nn.ModuleList([nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden),
                nn.LeakyReLU(),
                nn.Dropout(p=self.dropout)) for _ in range(self.n_hidden_layers)])

        # Output layer
        self.output_layer = nn.Linear(self.n_hidden, 2)
        self.alpha = cfg.fa_alpha

        fa_model_path = Path(cfg.process_data_path, *Path(cfg.train_data).parts[1:])
        Fa_model = os.path.join(fa_model_path, 'FA_transform.pkl')

        if os.path.isfile(Fa_model):
            self.Fa_loading_matrix, self.Fa_mean, self.Fa_variance = self.Load_Fa_Components(Fa_model)
            self.sigma = cfg.fa_sigma
            self.dsigma = cfg.fa_dsigma
            self.dbias = cfg.fa_dbias
            self.epsilon = cfg.epsilon
            self.beta = cfg.fa_beta
            self.reverse_kl = cfg.reverse_kl

        self.loss_fn_type=cfg.loss_fn_type
        self._define_cost()
        self.L1_Loss = nn.L1Loss(reduction='mean')
        self.L2_Loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction="mean")
        self.save_hyperparameters(cfg)
        self.val_step_logs=[]
        self.init_weights()
        self.eps=1e-8

        self.var_lim=torch.tensor(cfg.var_lim)
        self.log_var_lim=torch.log(self.var_lim).cuda() if os.environ.get('CUDA_VISIBLE_DEVICES') else torch.log(self.var_lim)
        self.max_grade=cfg.max_grade
        self.min_grade=cfg.min_grade

    def nn_forward(self, x):

        x = F.leaky_relu(self.input_layer(x))
        x = F.dropout(x, self.dropout)

        # Forward pass through hidden layers with dropout
        for layer in self.hidden_layers:
            x = layer(x)

        # Forward pass through output layer, just linear with 1d output for regression
        x = self.output_layer(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=self.hparams.lr_decay)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}


    def forward(self, batch):
        x, y = batch

        # Clean data ....
        y_hat = self.nn_forward(x)

        mu_x, logvar_x = y_hat[:,0], y_hat[:,1]


        print(f"Before Clamp, logvar_x: {logvar_x.mean()}, logvar_x: {logvar_x}")

        mu_x = torch.clamp(mu_x, min=self.min_grade, max=self.max_grade)
        logvar_x=torch.clamp(logvar_x, min=-self.log_var_lim, max=self.log_var_lim)


        print(f"After Clamp, logvar_x: {logvar_x.mean()}, logvar_x: {logvar_x} ")
        #Pg=self.Get_Noraml_dist(mu_x, logvar_x)

        NLL_cost = self._compute_cost_NLL_MVN(mu_x, logvar_x, y)
        print(f"NLL_cost:{NLL_cost}, mu_x:{mu_x}, logvar_x: {logvar_x.mean()}, targets: {y.squeeze()}")


        # OOD Data ....
        data, fa_distance = self.Sample_OOD_Data(x, sigma=self.sigma, dsigma=self.dsigma, dbias=self.dbias, epsilon=self.epsilon)
        if self.hparams.accelerator=='gpu':
            data = data.cuda()
            fa_distance = fa_distance.cuda()

        # fa_distance is variance
        yhat_ood = self.nn_forward(data)
        mu_ood, logvar_ood = yhat_ood[:, 0], yhat_ood[:, 1]

        mu_ood = torch.clamp(mu_ood, min=self.min_grade, max=self.max_grade)
        logvar_ood=torch.clamp(logvar_ood, min=-self.log_var_lim, max=self.log_var_lim)

        noise_cost = self._construct_KL_cost(mu_ood, fa_distance, mu_ood, logvar_ood)
        cost = self.hparams.fa_beta * NLL_cost + self.alpha * noise_cost


        self.log("NLL_cost", NLL_cost)
        self.log("Noise_cost", noise_cost)
        return y_hat, cost


    def Sample_OOD_Data(self, input, sigma=1.3, dsigma=6.0, dbias=20.0, epsilon=1e-8):

        # Generate random samples
        batch_size = input.shape[0]

        # latent size
        n_z = self.Fa_loading_matrix.shape[0]
        feat_size = self.Fa_loading_matrix.shape[1]

        z = torch.randn(batch_size, n_z)
        eta = torch.randn(batch_size, feat_size)

        #print(f"fa_var: {self.Fa_variance}")

        # Calculate data
        z = torch.sqrt(torch.tensor(epsilon + sigma)) * z
        data = torch.mm(z, self.Fa_loading_matrix) + self.Fa_mean + torch.sqrt(epsilon + self.Fa_variance * sigma) * eta

        # Calculate distances
        distances = dsigma * z.norm(dim=1) + dbias
        return data, distances


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
        with torch.no_grad():
            y_hat, loss = self.compute_forward_and_loss(batch, batch_idx)
        return y_hat, loss

    def compute_forward_and_loss(self, batch, batch_idx):
        x, y = batch
        y_hat, cost = self.forward(batch)
        return y_hat, cost

    def _define_cost(self):
        if self.loss_fn_type=="MSE":
            self.Loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction="mean")

        elif self.loss_fn_type=="NLL_MVN":
            pass;


    def _compute_cost_NLL_MVN(self, mu, log_vars, targets):
            mu_tgt = targets.squeeze()
            loss = 0.5 * torch.mean(((mu - mu_tgt))**2/(torch.exp(log_vars) + self.eps + log_vars))
            return loss


    #def _construct_KL_cost(self, targets, target_var, means, log_var):
    #    cost = -0.5 * torch.mean(((means - targets)**2 + target_var) / (torch.exp(log_var) + self.eps + log_var - torch.log(target_var)))
    #    #cost = 0.5 * tf.reduce_mean((tf.nn.l2_loss(means - targets) + target_var) / tf.exp(tf.maximum(log_var, epsilon)) + log_var - tf.log(target_var))
    #    return cost

    def _construct_KL_cost(self, targets, target_var, means, log_var):
        mu_q=targets
        log_var_q = torch.log(target_var)
        mu_p = means
        log_var_p = log_var

        var_p = torch.exp(log_var_p)
        var_q = torch.exp(log_var_q)

        # Compute KL divergence
        # cost = #0.5 * torch.mean(log_var_q - log_var_p + ((var_p + (mu_p - mu_q)**2)) / var_q - 1)
        cost = 0.5 * torch.mean((((mu_p - mu_q)**2 + var_p)/var_q) + log_var_q - log_var_p - 1)

        return cost

    def evaluate(self, dataloader):
        # This method is used for evaluation
        self.eval()  # Set the model to evaluation mode
        predictions_list = []  # List to accumulate predictions
        y_tgt_list = []
        eval_utt_list = []
        #Iterate through the data loader to make predictions
        with torch.no_grad():  # Disable gradient computation for inference
            for batch in dataloader:
                x, y_tgt, utt_ids = batch
                x, y_tgt = x.unsqueeze(0), y_tgt.unsqueeze(0)

                predictions, loss = self.evaluation_step((x, y_tgt), 0)  # 0 is a placeholder for batch_idx
                predictions_list.append(predictions)
                y_tgt_list.append(y_tgt)
                eval_utt_list+=[utt_ids]

        # Concatenate the predictions to create a single tensor
        predictions_tensor = torch.cat(predictions_list, dim=0)
        tgt_mu = torch.cat(y_tgt_list, dim=0)


        pred_mu = predictions_tensor[:, 0].unsqueeze(1)
        pred_log_std = predictions_tensor[:, 1].unsqueeze(1)

        return pred_mu, pred_log_std, tgt_mu, eval_utt_list

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

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.normal_(module.weight, mean=0, std=0.001)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def Load_Fa_Components(self, Fa_model):

        loaded_data = joblib.load(Fa_model)
        #fa_loading_matrix = loaded_data.components_
        #fa_mean = torch.tensor(loaded_data.mean_)
        #fa_variance = torch.tensor(loaded_data.noise_variance_)
        #fa_loading_matrix = torch.tensor(fa_loading_matrix).float()

        Fa_loading_matrix = torch.tensor(loaded_data.components_).float()
        Fa_mean = torch.tensor(loaded_data.mean_).float()
        Fa_variance = torch.tensor(loaded_data.noise_variance_).float()
        return Fa_loading_matrix, Fa_mean, Fa_variance

    def Get_Noraml_dist(self, mu, logvar):
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        std = torch.sqrt(torch.exp(logvar))
        Pd = Normal(mu, std)
        return Pd