"""
Code for self-training with weak supervision for regression based on Karamanolakis.
Author: Haoyu Sheng
"""

import numpy as np

# Weak Source Classes
# Here is the place to define heuristic rules (labeling functions)
# Note: most rules are already provided in benchmarks as pre-processed files (for efficiency).

class ECONRules:
    # Weak Source Class
    # has to implement apply function that applied to a dataset
    # predict() function that applies to a single text.
    def __init__(self, datapath="../data"):
        self.num_rules = 7
        self.preprocess = None

    def apply(self, dataset):
        preds = dataset.data['weak_labels']
        return preds
