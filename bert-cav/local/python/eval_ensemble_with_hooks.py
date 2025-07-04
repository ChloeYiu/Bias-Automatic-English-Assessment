'''
Evaluate set of BERT neural graders, save the activation and compute ensemble scores
'''

import os
import torch
torch.cuda.empty_cache()

import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from data_prep import get_data, get_mask_with_feature
import argparse
from tools import AverageMeter, get_default_device, calculate_rmse, calculate_pcc, calculate_less1, calculate_less05, calculate_avg, calculate_src, calculate_krc
from models import BERTGrader, BERTFeatureGrader, BERTLReLUGrader
import statistics

from path import makeDir, checkDirExists, checkFileExists
from cmdlog import makeCmdPath

from cav import ActivationGetter
import sys

def eval(val_loader, activation_getter, device):    
    targets = []
    preds = []
    model = activation_getter.model

    # switch to eval mode
    model.eval()

    activation_getter.add_hooks()
    # with torch.nograd():
    for i, (id, mask, target) in enumerate(val_loader):

        id = id.to(device)
        mask = mask.to(device)
        target = target.to(device)

        # Forward pass
        pred = model(id, mask)
        
        for layer_num in activation_getter.layer_nums:
            activation = activation_getter.activation_cache[layer_num]
            # Compute gradients for this sample
            grad = torch.autograd.grad(
                outputs=pred,
                inputs=activation,
                grad_outputs=torch.ones_like(pred),
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )
            activation_getter.store_tmp_gradient(grad)

        # Store
        preds += pred.tolist()
        targets += target.tolist()
    activation_getter.remove_hooks()

    return preds, targets

def get_single_stats(all_preds, targets):
    rmses = []
    pccs = []
    avgs = []
    less05s = []
    less1s = []
    srcs = []
    krcs = []

    for preds in all_preds:
        rmses.append(calculate_rmse(torch.FloatTensor(preds), torch.FloatTensor(targets)).item())
        pccs.append(calculate_pcc(torch.FloatTensor(preds), torch.FloatTensor(targets)).item())
        avgs.append(calculate_avg(torch.FloatTensor(preds)).item())
        less05s.append(calculate_less05(torch.FloatTensor(preds), torch.FloatTensor(targets)))
        less1s.append(calculate_less1(torch.FloatTensor(preds), torch.FloatTensor(targets)))
        srcs.append(calculate_src(torch.FloatTensor(preds), torch.FloatTensor(targets)).item())
        krcs.append(calculate_krc(torch.FloatTensor(preds), torch.FloatTensor(targets)).item())

    rmse_mean = statistics.mean(rmses)
    rmse_std = statistics.pstdev(rmses)
    pcc_mean = statistics.mean(pccs)
    pcc_std = statistics.pstdev(pccs)
    avg_mean = statistics.mean(avgs)
    avg_std = statistics.pstdev(avgs)
    less05_mean = statistics.mean(less05s)
    less05_std = statistics.pstdev(less05s)
    less1_mean = statistics.mean(less1s)
    less1_std = statistics.pstdev(less1s)
    src_mean = statistics.mean(srcs)
    src_std = statistics.pstdev(srcs)
    krc_mean = statistics.mean(krcs)
    krc_std = statistics.pstdev(krcs)

    return rmse_mean, rmse_std, pcc_mean, pcc_std, avg_mean, avg_std, less05_mean, less05_std, less1_mean, less1_std, src_mean, src_std, krc_mean, krc_std

def get_ensemble_stats(all_preds, targets):
    y_sum = torch.zeros(len(all_preds[0]))
    for preds in all_preds:
        y_sum += torch.FloatTensor(preds)
    ensemble_preds = y_sum/len(all_preds)

    rmse = calculate_rmse(ensemble_preds, torch.FloatTensor(targets))
    pcc = calculate_pcc(ensemble_preds, torch.FloatTensor(targets))
    avg = calculate_avg(ensemble_preds)
    less05 = calculate_less05(ensemble_preds, torch.FloatTensor(targets))
    less1 = calculate_less1(ensemble_preds, torch.FloatTensor(targets))
    src = calculate_src(ensemble_preds, torch.FloatTensor(targets))
    krc = calculate_krc(ensemble_preds, torch.FloatTensor(targets))

    return ensemble_preds.tolist(), rmse.item(), pcc.item(), avg.item(), less05, less1, src.item(), krc.item()

def main(args):
    model_paths = args.MODELS
    model_paths = model_paths.split()
    responses_file = args.RESPONSES
    out_dir = args.OUT
    activation_dir = args.ACTIVATION_DIR
    gradient_dir = args.GRADIENT_DIR
    grades_file = args.GRADES
    feature_file = args.FEATURE
    batch_size = args.B
    part=args.part
    feature_size = args.feature_size
    activation_fn = args.activation_fn

    for model_path in model_paths:
        print('model_path ' + model_path)
        checkFileExists(model_path)
    checkFileExists(responses_file)
    checkFileExists(grades_file)
    makeDir(out_dir, False)
    
    print("model: "+str(model_paths))
    
    # Save the command run
    makeCmdPath(out_dir, 'Evaluate ensemble')
    
    # Get the device
    device = get_default_device()

    # Load the data
    input_ids, mask, labels, speakerids = get_data(responses_file, grades_file, part=part)
    print(len(labels))

    if feature_file:
        speakerids, mask = get_mask_with_feature(feature_file, speakerids, mask, feature_size)
        
    test_ds = TensorDataset(input_ids, mask, labels)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    # Load the models
    models = []
    for model_path in model_paths:
        if feature_file:
            print("Loading model with features")
            model = BERTFeatureGrader(feature_size=feature_size)
        elif activation_fn == 'lrelu':
            print("Loading model with LReLU activation")
            model = BERTLReLUGrader()
        else:
            print("Loading model without features")
            model = BERTGrader()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.to(device)
        models.append(model)

    targets = None
    all_preds = []

    for i, model in enumerate(models):
        model_name = os.path.splitext(os.path.basename(model_paths[i]))[0]
        activation_getter = ActivationGetter(model, model_name, activation_dir, gradient_dir, [1], batch_size)

        preds, targets = eval(test_dl, activation_getter, device)
        all_preds.append(preds)

        # Save the predicted scores for each model
        model_out_file = os.path.join(out_dir, f'preds_{model_name}.txt')

        with open(model_out_file, 'w') as f:
            text = 'SPEAKERID REF PRED'
            f.write(text)
        for spk, ref, pred in zip(speakerids, targets, preds):
            with open(model_out_file, 'a') as f:
                text = '\n'+spk + ' ' + str(ref) + ' ' + str(pred)
                f.write(text)

        # Save the activations for each layer
        activation_getter.store_activations(speakerids)
        activation_getter.store_gradients(speakerids)

        # Get single stats for each model
        rmse_mean, rmse_std, pcc_mean, pcc_std, avg_mean, avg_std, less05_mean, less05_std, less1_mean, less1_std, src_mean, src_std, krc_mean, krc_std = get_single_stats([preds], targets)
        print(f"\nSTATS FOR {model_name}\n")
        print("SINGLE STATS\n")
        print("RMSE: "+str(rmse_mean))
        print("PCC: "+str(pcc_mean))
        print("AVG: "+str(avg_mean))
        print("LESS05: "+str(less05_mean))
        print("LESS1: "+str(less1_mean))
        print("SRC: "+str(src_mean))
        print("KRC: "+str(krc_mean))

    # Get single stats (over all models)
    rmse_mean, rmse_std, pcc_mean, pcc_std, avg_mean, avg_std, less05_mean, less05_std, less1_mean, less1_std, src_mean, src_std, krc_mean, krc_std = get_single_stats(all_preds, targets)
    print("STATS FOR ", model_paths)
    print()
    print("\nOVERALL SINGLE STATS\n")
    print("RMSE: "+str(rmse_mean)+" +- "+str(rmse_std))
    print("PCC: "+str(pcc_mean)+" +- "+str(pcc_std))
    print("AVG: "+str(avg_mean)+" +- "+str(avg_std))
    print("LESS05: "+str(less05_mean)+" +- "+str(less05_std))
    print("LESS1: "+str(less1_mean)+" +- "+str(less1_std))
    print("SRC: "+str(src_mean)+" +- "+str(src_std))
    print("KRC: "+str(krc_mean)+" +- "+str(krc_std))

    # Get ensemble stats
    ensemble_preds, rmse, pcc, avg, less05, less1, src, krc = get_ensemble_stats(all_preds, targets)
    print()
    print("ENSEMBLE STATS\n")
    print("RMSE: ", rmse)
    print("PCC: ", pcc)
    print("AVG: ", avg)
    print("LESS05: ", less05)
    print("LESS1: ", less1)
    print("SRC: ", src)
    print("KRC: ", krc)

    # Save the ensemble predicted scores
    ensemble_out_file = os.path.join(out_dir, f'ensemble_preds_part{part}.txt')
    with open(ensemble_out_file, 'w') as f:
        text = 'SPEAKERID REF PRED'
        f.write(text)
    for spk, ref, pred in zip(speakerids, targets, ensemble_preds):
        with open(ensemble_out_file, 'a') as f:
            text = '\n'+spk + ' ' + str(ref) + ' ' + str(pred)
            f.write(text)
    

if __name__ == "__main__":
    # print("Command-line arguments:", sys.argv)
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODELS', type=str, help='trained .th models separated by space')
    commandLineParser.add_argument('RESPONSES', type=str, help='responses text file')
    commandLineParser.add_argument('GRADES', type=str, help='scores text file')
    commandLineParser.add_argument('OUT', type=str, help='output directory for predictions')
    commandLineParser.add_argument('ACTIVATION_DIR', type=str, help='directory to store activations')
    commandLineParser.add_argument('GRADIENT_DIR', type=str, help='directory to store gradients')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")
    commandLineParser.add_argument('--FEATURE', type=str, default='', help="Specify test feature file")
    commandLineParser.add_argument('--activation_fn', type=str, default='relu', help="Specify test feature file")
    commandLineParser.add_argument('--part', type=int, default=3, help="Specify part of exam")
    commandLineParser.add_argument('--feature_size', type=int, default=356, help="Specify feature size")

    args = commandLineParser.parse_args()
    main (args)
    
