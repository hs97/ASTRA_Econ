"""
Code for self-training with weak supervision for regression analysis.
Original Author: Giannis Karamanolakis (gkaraman@cs.columbia.edu)
Modified by: Haoyu Sheng (haoyu_sheng@brown.edu)
"""

import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from collections import defaultdict
implemented_metrics = ['mse', 'explained_var', 'r2']

class Evaluator:
    # A class that implements all evaluation metrics and prints relevant statistics
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.metric = args.metric
        assert self.metric in implemented_metrics, "Evaluation metric not implemented: {}".format(self.metric)

    def evaluate(self, preds, labels, proba=None, comment="", verbose=True):
        assert len(preds) == len(labels), "pred should have same length as true: pred={} gt={}".format(
                len(preds),
                len(labels)
        )

        preds = np.array(preds)
        labels = np.array(labels)

        total_num = len(preds)
        self.logger.info("Evaluating {} on {} examples".format(comment, total_num))

        # Ignore pred == -999 but also penalize by considering all of them as wrong predictions...
        ignore_ind = preds == -999
        keep_ind = preds != -999
        ignore_num = np.sum(ignore_ind)
        ignore_perc = ignore_num / float(total_num)
        if ignore_num > 0:
            self.logger.info("Ignoring {:.4f}% ({}/{}) predictions".format(100*ignore_perc, ignore_num, total_num))

        preds = preds[keep_ind]
        labels = labels[keep_ind]
        if proba is not None:
            proba = proba[keep_ind]
        if len(preds) == 0:
            self.logger.info("Passed empty {} list to Evaluator. Skipping evaluation".format(comment))
            return defaultdict(int)

        pred = list(preds)
        true = list(labels)
        mse = mean_squared_error(true, pred)
        r2 = r2_score(true, pred)
        var_explained = explained_variance_score(true, pred)
        adjust_coef = (total_num - ignore_num) / float(total_num)

        res = {
            'mse': mse * adjust_coef,
            'r2': 100 * r2 * adjust_coef,
            'var_explained': 100 * var_explained * adjust_coef,
            'ignored': ignore_num,
            'total': total_num
        }
        res["perf"] = res[self.metric]

        self.logger.info("{} performance: {:.2f}".format(comment, res["perf"]))

        return res
