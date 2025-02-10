#! /usr/bin/python

import torch
from torchmetrics.functional.regression import pearson_corrcoef
import pandas as pd
import numpy as np
import os
import sys
import yaml


#-------------------------------------------------------
def process_data_file(data):
    #breakpoint()
    np_data = np.load(data)
    feats, labels = torch.from_numpy(np_data[:, 1:-1].astype(np.float32)), torch.from_numpy(np_data[:, -1].astype(np.float32))
    uttid_lists = np_data[:, 0].tolist()

    labels = labels.unsqueeze(1)
    return feats, labels, uttid_lists

#--------------------------------------------------------------
def mean_std_rounding(inp_stats_list):
    inp_stats_list = torch.tensor(inp_stats_list)
    mu = (torch.mean(inp_stats_list).item())
    mu = round(mu, 3)

    sigma2 = (torch.std(inp_stats_list).item())
    sigma2 = round(sigma2, 3)
    return mu, sigma2
#-----------------------------------------------------

def rename_keys(input_dict, key_mapping):
    # Create a new dictionary to store the renamed keys
    renamed_dict = {}
    for key, value in input_dict.items():
        # Check if the key needs to be renamed, and use the new key if specified in the key_mapping
        new_key = key_mapping.get(key, key)
        renamed_dict[new_key] = value
    return renamed_dict

#-----------------------------------------------------
def Compute_metrics(y_pred, y_tgt):
    epoch_loss_mse = torch.nn.functional.mse_loss(y_pred, y_tgt, reduction='mean')
    epoch_loss_PCC = pearson_corrcoef(y_pred, y_tgt)
    epoch_loss_mae = torch.nn.functional.l1_loss(y_pred, y_tgt, reduction='mean')

    epoch_lt_05 = ((abs(y_tgt-y_pred) < 0.5)*1).sum()/y_tgt.shape[0]
    epoch_lt_1 = ((abs(y_tgt-y_pred) < 1)*1).sum()/y_tgt.shape[0]

    epoch_loss_rmse = torch.sqrt(epoch_loss_mse)

    output_dict={
            "MSE":epoch_loss_mse,
            "PCC":epoch_loss_PCC*100,
            "MAE":epoch_loss_mae,
            "lt_0.5":epoch_lt_05*100,
            "lt_1":epoch_lt_1*100,
            "RMSE":epoch_loss_rmse}

    output_dict = {k:v.item() for k, v in output_dict.items()}
    return output_dict

#----------- ------------------------- ----------------------
def compute_metrics_from_df(df):
    print(f"Dataframe has columns: {df.columns}")
    print(f"Dataframe was tested on datasets: {df.dataset.unique()}")
    print(f"Dataframe was tested on models: {df.model_path.unique()}")


    seed_model_stats=[]
    (f"Metrics :   & pcc & mse & mae & lt50 & lt100")
    #--- -- --- --- ---- ---- ----
    for ele in df.model_path.unique():
        d1 = df[df.model_path==ele]


        per_model_stats=[]
        for chkpt_file in d1.chkpt_file.unique():
            d2 = d1[d1.chkpt_file==chkpt_file]

            pmu = torch.stack(d2.pred_mu.to_list(),dim=0)
            tmu = torch.stack(d2.tgt_mu.to_list(),dim=0)
            computed_metrics = Compute_metrics(pmu, tmu)

            computed_metrics.update({"model_path": ele})

            seed_model_stats.append(computed_metrics)
            per_model_stats.append(computed_metrics)


        print(f"Printing stats for model file {ele}")
        seed_model = pd.DataFrame(per_model_stats)
        mse_list = torch.stack(seed_model.epoch_loss_mse.to_list(),dim=0)
        mae_list = torch.stack(seed_model.epoch_loss_mae.to_list(),dim=0)
        pcc_list = torch.stack(seed_model.epoch_loss_PCC.to_list(),dim=0)
        lt50_list = torch.stack(seed_model.epoch_lt_05.to_list(),dim=0)
        lt1_list = torch.stack(seed_model.epoch_lt_1.to_list(),dim=0)


        pcc, pcc_2sigma = mean_std_rounding(pcc_list*100)
        mse, mse_2sigma = mean_std_rounding(mse_list)
        mae, mae_2sigma = mean_std_rounding(mae_list)
        lt50, lt50_2sigma = mean_std_rounding(lt50_list*100)
        lt100, lt100_2sigma = mean_std_rounding(lt1_list*100)

        print(f"Seed Model:   & {pcc} $\pm$ {pcc_2sigma} & {mse} $\pm$ {mse_2sigma} & {mae} $\pm$ {mae_2sigma} & {lt50} $\pm$ {lt50_2sigma} & {lt100} $\pm$ {lt100_2sigma} ")


        #-------------------------------------------------------------------------

        d4 = d1.groupby('eval_uttlist').agg({'pred_mu': list, 'tgt_mu': list}).reset_index()

        d4["aug_pred_mu"] = d4["pred_mu"].apply(lambda x: torch.mean(torch.stack(x,dim=0)))
        d4["aug_tgt_mu"]  = d4["tgt_mu"].apply(lambda x: torch.mean(torch.stack(x,dim=0)))

        pmu = torch.stack(d4.aug_pred_mu.to_list(),dim=0)
        tmu = torch.stack(d4.aug_tgt_mu.to_list(),dim=0)

        computed_metrics = Compute_metrics(pmu, tmu)
        computed_metrics = {i:val.item() for i,val in computed_metrics.items()}


        pcc_ens = round(computed_metrics['epoch_loss_PCC']*100,2)
        mse_ens = round(computed_metrics['epoch_loss_mse'], 2)
        mae_ens = round(computed_metrics['epoch_loss_mae'], 2)
        lt50_ens = round(computed_metrics['epoch_lt_05']*100, 2)
        lt100_ens = round(computed_metrics['epoch_lt_1']*100, 2)


        print(f"Ens model:   & {pcc_ens} & {mse_ens}  & {mae_ens}  & {lt50_ens}  & {lt100_ens}")

        print(f"               ---------------                        ")


#-------------------------------------------------------------------------
def consolidate_datafrmaes_perspk(df):
    df2_output_list=[]
    for ele in df.model_path.unique():
        d1 = df[df.model_path==ele]

        for spk in df.eval_uttlist.unique():
            d2=d1[d1.eval_uttlist==spk]

            pred_mu = d2.pred_mu.mean()
            tgt_mu = d2.tgt_mu.mean()

            df2_output_list.append({"spk":spk, "model_path":ele, "pred_mu":pred_mu, "tgt_mu":tgt_mu})
    pd_calib = pd.DataFrame(df2_output_list)
    return pd_calib