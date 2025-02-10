#! /usr/bin/python

import numpy as np
import os
import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.INFO)



#####################################################################
def process_data_np_matrix_v2(feats, grades, logging):
    # Read and process feature and grade lines
    with open(feats, 'r') as f:
        feat_lines = [line.strip().split() for line in f.readlines()[1:]]
    with open(grades, 'r') as f:
        grade_lines = [line.strip().split() for line in f.readlines()[1:]]

    feat_dict = {f[0]: np.array(f[1:], dtype=float) for f in feat_lines}
    grade_dict = {g[0]: float(g[1]) for g in grade_lines}

    # Check if features and grades match
    if len(feat_dict) != len(grade_dict):
        print(f"Feature and grade counts do not match: {len(feat_dict)} vs {len(grade_dict)}")

    # Prepare output data
    data = [np.concatenate([[k], feat_dict[k], [grade_dict[k]]]) for k in feat_dict if k in grade_dict]

    np_data = np.array(data)
    return np_data[:, 1:-1], np_data[:, -1:], np_data[:, :1]













#####################################################################
def process_data_np_matrix(feats, grades,logging):

    feat_lines = open(feats,"r").readlines()
    feat_lines = [line.strip().replace("\t", " ") for line in feat_lines[1:]]

    curr_line = feat_lines[0]
    logging.info(f"current data is formatted in the given format")

    uid = curr_line.split(" ")[0]
    ufeat = curr_line.split(" ")[1:]
    logging.info(f"uttid: {uid},  feature: {ufeat}")
    feat_dict = {line.split(" ")[0]: line.split(" ")[1:] for line in feat_lines}

    grades_lines = open(grades,"r").readlines()
    grades_lines = [line.strip().replace("\t", " ") for line in grades_lines[1:]]
    grades_dict = {line.split(" ")[0]: line.split(" ")[1] for line in grades_lines}

    ##   checks:
    check_grades_file = [len(line.split(" ")) for line in grades_lines]
    if not all(x == 2 for x in check_grades_file):
        logging.info(f"The grades files needs to have only data with two columns, looks like a columns has extra lines, make it strictly 2")
        logging.info(f"Data format in grades file  c1cluxxxx   3.5 ")
        logging.info(f"if you want to process data with no grades then atlease create dummy grade 1 or 2 for all data")


    if (len(feat_lines)!=len(grades_lines)):
        logging.info(f"grades and feats has different no of lines feats len is {len(feat_lines)} , where as grades len is{len(grades_lines)}")
    else:
        logging.info(f"feat_len: {len(feat_lines)}, grades len : {len(grades_lines)}")


    #---- --- --- --- -----
    feat_mat = feat_dict
    target_mat = grades_dict

    output_list = []

    for key, val in feat_mat.items():
        if target_mat.get(key, None):
            val = np.array(val, dtype=float)
            tgt = np.array([target_mat[key]], dtype=float)
            K = np.array([key])

            data_tgt_append = np.concatenate((K, val, tgt),axis=0)
            output_list.append(data_tgt_append)

        else:
            print(f"{key} is not present in the grades list")

    ###   ##

    np_data = np.array(output_list)
    utt_list = np_data[:,0]
    train_data_feat = np_data[:, 1:-1]
    train_data_tgts = np_data[:, -1]

    train_data_tgts = np.expand_dims(train_data_tgts, axis=1)
    train_data_uttids = np.expand_dims(utt_list, axis=1)
    return train_data_feat, train_data_tgts, train_data_uttids





########################################################################
def read_text_target_np_matrix(feat, target, part):

    feat_mat = [line.strip().replace("\t"," ") for line in open(feat,"r").readlines()]
    target_mat = [line.strip().replace("\t"," ") for line in open(target,"r").readlines()]

    feat_mat = {line.split(' ')[0]:line.split(' ')[1:] for line in feat_mat[1:]}
    target_mat = {line.split(' ')[0]:line.split(' ')[1:] for line in target_mat[1:]}

    output_list = []
    for key, val in feat_mat.items():

        if target_mat.get(key, None):
            val = np.array(val, dtype=float)
            tgt = np.array([target_mat[key][part-1]], dtype=float)

            data_tgt_append = np.concatenate((val,tgt),axis=0)
            output_list.append(data_tgt_append)

        else:
            print(f"{key} is not present in the grades list")

    ###   ##

    np_data = np.array(output_list)
    train_data_feat = np_data[:, :-1]
    train_data_tgts = np_data[:, -1]
    train_data_tgts = np.expand_dims(train_data_tgts, axis=1)
    return train_data_feat, train_data_tgts
########################################################################



"""
def read_text_target_np_matrix_uttid(feat, target, part):
    breakpoint()
    feat_mat = [line.strip().replace("\t"," ") for line in open(feat,"r").readlines()]
    feat_mat[0] = 'Speaker ' + feat_mat[0]
    target_mat = [line.strip().replace("\t"," ") for line in open(target,"r").readlines()]

    #feat_mat = {line.split(' ')[0]:line.split(' ')[1:] for line in feat_mat[1:]}
    #target_mat = {line.split(' ')[0]:line.split(' ')[1:] for line in target_mat[1:]}

    #feat_mat = [pd.DataFrame({line.split(' ')[0]:line.split(' ')[1:]}) for line in feat_mat]
    #target_mat = [pd.DataFrame({line.split(' ')[0]:line.split(' ')[1:]}) for line in target_mat]


    feat_mat_dict = [line.split(' ') for line in feat_mat]
    columns = feat_mat_dict[0]
    feat_mat_dict = feat_mat_dict[1:]
    df1 = pd.DataFrame(feat_mat_dict, columns=columns)

    target_mat = [line.split(' ') for line in target_mat]
    columns = target_mat[0]
    target_mat = target_mat[1:]
    df2 = pd.DataFrame(target_mat, columns=columns)

    breakpoint()
    df.columns = pd.MultiIndex.from_tuples(
    [(col[0], col[1], 'extra_name') for col in df.columns],
    names=['level_0', 'level_1', 'extra_level'])

    pd_merged = pd.merge(df1, df2,on="Speaker",how="outer")

    breakpoint()






    output_list = []

    for key, val in feat_mat.items():

        if target_mat.get(key, None):
            val = np.array(val, dtype=float)
            tgt = np.array([target_mat[key][part-1]], dtype=float)

            data_tgt_append = np.concatenate((val,tgt),axis=0)
            output_list.append(data_tgt_append)

        else:
            print(f"{key} is not present in the grades list")

    ###   ##

    np_data = np.array(output_list)
    train_data_feat = np_data[:, :-1]
    train_data_tgts = np_data[:, -1]
    train_data_tgts = np.expand_dims(train_data_tgts, axis=1)
    return train_data_feat, train_data_tgts
"""

def read_text_target_np_matrix_uttid(feat, target, part):

    feat_mat = [line.strip().replace("\t"," ") for line in open(feat,"r").readlines()]
    target_mat = [line.strip().replace("\t"," ") for line in open(target,"r").readlines()]

    feat_mat = {line.split(' ')[0]:line.split(' ')[1:] for line in feat_mat[1:]}
    target_mat = {line.split(' ')[0]:line.split(' ')[1:] for line in target_mat[1:]}

    output_list = []
    for key, val in feat_mat.items():

        if target_mat.get(key, None):
            val = np.array(val, dtype=float)
            tgt = np.array([target_mat[key][part-1]], dtype=float)
            K = np.array([key])
            data_tgt_append = np.concatenate((K, val, tgt),axis=0)
            output_list.append(data_tgt_append)

        else:
            print(f"{key} is not present in the grades list")

    ###   ##

    np_data = np.array(output_list)
    utt_list = np_data[:,0]
    train_data_feat = np_data[:, 1:-1]
    train_data_tgts = np_data[:, -1]

    train_data_tgts = np.expand_dims(train_data_tgts, axis=1)
    train_data_uttids = np.expand_dims(utt_list, axis=1)
    return train_data_feat, train_data_tgts, train_data_uttids



class SplitOnTabsAndNewlinesAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Split the input string on tabs and newlines
        setattr(namespace, self.dest, values.split('\t') + values.split('\n'))