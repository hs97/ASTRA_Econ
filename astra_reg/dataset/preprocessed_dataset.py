"""
Code for self-training with weak supervision.
Author: Giannis Karamanolakis (gkaraman@cs.columbia.edu)
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import glob
import joblib
from copy import deepcopy
import shutil

# PreprocessedDataset is used for loading exactly the same dataset splits and features as in our experiments

class PreprocessedDataset:
    # Load pre-processed dataset as used in https://github.com/awasthiabhijeet/Learning-From-Rules
    def __init__(self, datapath="../../data", orig_train=True, dataset='trec', seed=42):
        self.dataset = dataset
        self.seed = seed
        self.basedatafolder = os.path.join(datapath, self.dataset.upper())
        self.datafolder = os.path.join(self.basedatafolder, 'seed{}/'.format(seed))
        self.language = 'english'
        self.orig_train = orig_train

    def load_data(self, method):
        if method == 'train' and not self.orig_train:
            method = 'unlabeled'
        texts = joblib.load(os.path.join(self.datafolder, "{}_x.pkl".format(method)))
        texts = texts.tolist()

        labels = joblib.load(os.path.join(self.datafolder, "{}_labels.pkl".format(method)))
        labels = labels.squeeze().tolist()
        rule_preds = joblib.load(os.path.join(self.datafolder, "{}_rule_preds.pkl".format(method)))
        rule_preds = rule_preds.tolist()

        if method =='train':
            exemplars = joblib.load(os.path.join(self.datafolder, 'train_exemplars.pkl'))
            return {'texts': texts, 'labels': labels, 'weak_labels': rule_preds, 'exemplar_labels': exemplars}
        else:
            return {'texts': texts, 'labels': labels, 'weak_labels': rule_preds}
