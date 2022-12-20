"""
Code for self-training with weak supervision for regression analysis.
Original Author: Giannis Karamanolakis (gkaraman@cs.columbia.edu)
Modified by: Haoyu Sheng (haoyu_sheng@brown.edu)
"""

import os
from torch.utils.data import Dataset
from dataset import PreprocessedDataset
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.utils import shuffle

preprocessed_dataset_list = ['trec', 'youtube', 'sms', 'census', 'mitr', 'spouse', 'econ', 'econ_reg', 'econ_reg_ffill', 'econ_reg_EU']

def get_dataset_obj(dataset):
    if dataset in preprocessed_dataset_list:
        return PreprocessedDataset
    else:
        raise(BaseException('dataset not supported: {}'.format(dataset)))


class DataHandler:
    # This module is responsible for feeding the data to teacher/student
    # If teacher is applied, then student gets the teacher-labeled data instead of ground-truth labels
    def __init__(self, args, logger=None, student_preprocess=None, teacher_preprocess=None):
        self.args = args
        self.dataset = args.dataset
        self.logger = logger
        self.student_preprocess = student_preprocess
        self.teacher_preprocess = teacher_preprocess
        self.datasets = {}
        self.seed = args.seed
        np.random.seed(self.seed)

    def load_dataset(self, method='train'):
        dataset = WSDataset(self.args, method=method,
                            student_preprocess=self.student_preprocess,
                            teacher_preprocess=self.teacher_preprocess,
                            logger=self.logger)
        self.datasets[method] = dataset
        return dataset

    def create_pseudodataset(self, wsdataset):
        dataset = PseudoDataset(self.args, wsdataset, self.logger)
        return dataset


class WSDataset(Dataset):
    # WSDataset: Dataset for Weak Supervision.
    def __init__(self, args, method, logger=None, student_preprocess=None, teacher_preprocess=None):
        super(WSDataset, self).__init__()
        self.args = args
        self.seed = args.seed
        self.dataset = args.dataset
        self.datapath = args.datapath
        self.method = method
        self.downsample = args.downsample
        self.savefolder = os.path.join(args.experiment_folder, "preprocessed/")
        os.makedirs(self.savefolder, exist_ok=True)
        self.student_preprocess = student_preprocess
        self.teacher_preprocess = teacher_preprocess
        self.logger = logger
        if self.dataset in preprocessed_dataset_list:
            self.dataset_obj = get_dataset_obj(args.dataset)(datapath=self.datapath, dataset=self.dataset, seed=self.seed)
        else:
            self.dataset_obj = get_dataset_obj(args.dataset)(datapath=self.datapath)
        self.data = {}
        self.load_dataset()
        self.report_stats()

    def report_stats(self):
        if 'labels' in self.data:
            self.logger.info("{} DATASET: {} examples".format(self.method, len(self.data['labels'])))
            self.logger.info("{} LABELS:\n{}".format(self.method, sum(self.data['labels']) / len(self.data['labels'])))
        return

    def load_dataset(self):
        if not os.path.exists(self.dataset_obj.datafolder):
            self.dataset_obj.preprocess()
        data = self.dataset_obj.load_data(self.method)
        for key, value in data.items():
            self.data[key] = value
        if self.student_preprocess is not None:
            self.logger.info("Pre-processing {} data for student...".format(self.method))
            self.data['preprocessed_texts'] = self.student_preprocess(self.data['texts'])
        if self.teacher_preprocess is not None:
            self.logger.info("Pre-processing {} data for teacher...".format(self.method))
            self.data['preprocessed_lfs'] = self.teacher_preprocess(self)

    def oversample(self, oversample_num):
        # Over-sample labeled data
        # Used for fair comparison to the "ImplyLoss" paper that is doing the same pre-processing step.
        self.logger.info("Oversampling {} data {} times".format(self.method, oversample_num))
        for key, value in self.data.items():
            oversampled_values = []
            for i in range(oversample_num):
                oversampled_values.extend(value)
            self.data[key] = oversampled_values
        self.report_stats()
        return

    def __len__(self):
        return len(self.data['texts'])

    def __getitem__(self, item):
        ret = {
            'text': self.data['texts'][item],
            'input_ids': torch.tensor(self.data['input_ids'][item]),
            'attention_mask': torch.tensor(self.data['attention_mask'][item]),
            'label': torch.tensor(self.data['labels'][item]) if 'labels' in self.data else None
        }
        return ret


class PseudoDataset(Dataset):
    # PseudoDataset: a Dataset class that provides extra functionalities for teacher-student training.
    def __init__(self, args, wsdataset, logger=None):
        super(PseudoDataset, self).__init__()
        self.args = args
        self.seed = args.seed
        self.dataset = args.dataset
        self.method = wsdataset.method
        self.logger = logger
        self.dataset_obj = wsdataset.dataset_obj
        self.logger.info("copying data from {} dataset".format(wsdataset.method))
        if args.dataset == 'mitr':
            self.original_data = wsdataset.data
            self.data = self.original_data
        else:
            self.original_data = deepcopy(wsdataset.data)
            self.data = deepcopy(self.original_data)
        self.logger.info("done")

    def keep(self, keep_indices, update=None):
        self.logger.info("Creating Pseudo Dataset with {} items...".format(len(keep_indices)))
        for key, values in self.data.items():
            self.data[key] = [values[i] for i in keep_indices]
        self.data['original_indices'] = keep_indices
        if update is not None:
            for key, values in update.items():
                self.logger.info("Updating {}".format(key))
                assert len(values) == len(keep_indices), "update values need to have same dimension as indices {} vs {}".format(len(values), len(keep_indices))
                self.data[key] = list(values)

    def report_stats(self, column_name='labels'):
        if 'labels' in self.data:
            df = pd.DataFrame()
            df['ind'] = np.arange(len(self.data[column_name]))
            df['label'] = self.data[column_name]
            self.logger.info("PSEUDO-DATASET:\n{} examples\n with mean PSEUDO-LABELS:\n{}".format(df.shape[0], df['label'].mean()))
        return

    def drop(self, col='teacher_labels', value=-1):
        indices = [i for i, l in enumerate(self.data[col]) if l != value]
        self.keep(indices)

    def append(self, dataset, merge_cols=None):
        """
        :param dataset: the dataset to append to self
        :param merge_cols: a dictionary showing which columns to merge...
        :return: self + dataset concatenated
        """
        self.logger.info("Merging datasets {}, {}".format(self.method, dataset.method))
        self.logger.info("Size before merging: {}".format(len(self)))
        def extend_values(col1, col2):
            if isinstance(col1, list) and isinstance(col2, list):
                col1.extend(col2)
            elif isinstance(col1, list):
                if col2.ndim == 2:
                    col1 = np.repeat(np.array(col1)[..., np.newaxis], col2.shape[1], axis=1)
                else:
                    col2 = col2.tolist()
                col1 = np.concatenate([col1, col2])
            elif isinstance(col2, list):
                if col1.ndim == 2:
                    col2 = np.repeat(np.array(col2)[..., np.newaxis], col1.shape[1], axis=1)
                else:
                    col1 = col1.tolist()
                col1 = np.concatenate([col1, col2])
            else:
                col1 = np.concatenate([col1, col2])
            return col1

        if merge_cols is None:
            merge_cols = {}

        N = len(self.original_data['texts'])
        M = len(dataset)
        common = set(self.data) & set(dataset.data)
        self_only = set(self.data) - set(dataset.data)
        other_only = set(dataset.data) - set(self.data)

        for key in common:
            #self.data[key].extend(dataset.data[key])
            self.data[key] = extend_values(self.data[key], dataset.data[key])

        for key in self_only:
            #self.data[key].extend([-1] * M)
            self.data[key] = extend_values(self.data[key], [-1] * M)

        for key in other_only:
            self.data[key] = [-1] * N
            # self.data[key].extend(dataset.data[key])
            self.data[key] = extend_values(self.data[key], dataset.data[key])

        for col1, col2 in merge_cols.items():
            self.logger.info("Merging {} to {}".format(col1, col2))
            self.data[col1][-M:] = dataset.data[col2]
        self.logger.info("Size after merging: {}".format(len(self)))
        return

    def __len__(self):
        return len(self.data['texts'])

    def __getitem__(self, item):
        ret = {
            'text': self.data['texts'][item],
            'input_ids': torch.tensor(self.data['input_ids'][item]),
            'attention_mask': torch.tensor(self.data['attention_mask'][item]),
            'label': torch.tensor(self.data['labels'][item]) if 'labels' in self.data else None
        }
        return ret