'''
Same as eval_all.py but also linear calibrates using passed coefficients
'''

import torch
import sys
import os
import argparse
from tools import calculate_rmse, calculate_pcc, calculate_less1, calculate_less05, calculate_avg, calculate_src, calculate_krc

from path import makeDir, checkDirExists, checkFileExists
from cmdlog import makeCmdPath
import statistics

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

def main (args):
    pred_files = args.PREDS
    pred_files = pred_files.split()
    out_file = args.OUT
    gradient = args.gradient
    intercept = args.intercept

    for pred_file in pred_files:
        checkFileExists(pred_file)
        
    out_dir = os.path.dirname(out_file)
    makeDir (out_dir, False)
    
    # Save the command run
    makeCmdPath(out_dir, 'Apply calibration to predictions and evaluate')

    # Create spk_to_pred and spk_to_ref dicts
    pred_dicts = []
    ref_dicts = []
    for pred_file in pred_files:
        pred_dict = {}
        ref_dict = {}
        with open(pred_file) as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = lines[1:] # exclude header

        for line in lines:
            items = line.split()
            speakerid = items[0]
            ref = float(items[1])
            pred = float(items[2])
            # calibrate prediction
            pred = (pred*gradient)+intercept
            pred_dict[speakerid] = pred
            ref_dict[speakerid] = ref

        pred_dicts.append(pred_dict)
        ref_dicts.append(ref_dict)

    # Form id, ref, pred lists
    speakerids = []
    refs = []
    preds = []

    for id in pred_dicts[0]:
        speakerids.append(id)
        ref_sum = 0
        ref_counter = 0

        for ref_dict in ref_dicts:
            try:
                ref = ref_dict[id]
                ref_sum += ref
                ref_counter += 1
            except:
                continue
        if ref_counter > 0:
            ref_overall = ref_sum/ref_counter
        else:
            ref_overall = 0.0
        refs.append(ref_overall)

        pred_sum = 0
        pred_counter = 0

        for pred_dict in pred_dicts:
            try:
                pred = pred_dict[id]
                pred_sum += pred
                pred_counter += 1
            except:
                continue
        if pred_counter > 0:
            pred_overall = pred_sum/pred_counter
        else:
            pred_overall = 0.0
        preds.append(pred_overall)

        # Get single stats (over all models)
    rmse_mean, rmse_std, pcc_mean, pcc_std, avg_mean, avg_std, less05_mean, less05_std, less1_mean, less1_std, src_mean, src_std, krc_mean, krc_std = get_single_stats(preds, refs)
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

    # Get all the stats
    rmse = calculate_rmse(torch.FloatTensor(preds), torch.FloatTensor(refs)).item()
    pcc = calculate_pcc(torch.FloatTensor(preds), torch.FloatTensor(refs)).item()
    avg = calculate_avg(torch.FloatTensor(preds)).item()
    less05 = calculate_less05(torch.FloatTensor(preds), torch.FloatTensor(refs))
    less1 = calculate_less1(torch.FloatTensor(preds), torch.FloatTensor(refs))
    src = calculate_src(torch.FloatTensor(preds), torch.FloatTensor(refs)).item()
    krc = calculate_krc(torch.FloatTensor(preds), torch.FloatTensor(refs)).item()

    print("ALL PARTS STATS after calibration\n")
    print("RMSE: ", rmse)
    print("PCC: ", pcc)
    print("AVG: ", avg)
    print("LESS05: ", less05)
    print("LESS1: ", less1)
    print("SRC: ", src)
    print("KRC: ", krc)

    # Save the predicted scores
    with open(out_file, 'w') as f:
        text = 'SPEAKERID REF PRED'
        f.write(text)
    for spk, ref, pred in zip(speakerids, refs, preds):
        with open(out_file, 'a') as f:
            text = '\n'+spk + ' ' + str(ref) + ' ' + str(pred)
            f.write(text)

if __name__ == "__main__":
    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('PREDS', type=str, help='pred.txt files separated by space')
    commandLineParser.add_argument('OUT', type=str, help='predicted scores file')
    commandLineParser.add_argument('--gradient', type=float, default=0.0, help='calibration gradient')
    commandLineParser.add_argument('--intercept', type=float, default=0.0, help='calibration y-intercept')

    args = commandLineParser.parse_args()
    main(args)
