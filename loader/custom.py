#!/usr/bin/env python
import sys
import os
import os.path
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import logging
import pandas as pd
from loader.base import BaseLoader
from spec.task import DontPatronizeMe
from datetime import datetime


class CustomLoader(BaseLoader):
    name = "Fold"

    def __init__(self, token=""):
        super().__init__(token)

    def fold(self, k):
        """
        Generate k-fold cross validation dataset
        :param k: number of folds
        :return: train_data, test_data
        """
        fold_size = int(len(self.all_data) / k)
        self.all_data = np.random.permutation(self.all_data)
        self.test_data, self.train_data = np.split(self.all_data.copy(), [fold_size], axis=0)
        for i in range(k):
            yield self.train_data, self.test_data
            if i + 1 != k:
                self.train_data[i * fold_size:(i + 1) * fold_size], self.test_data = \
                    self.test_data, self.train_data[i * fold_size:(i + 1) * fold_size].copy()

    def nested_fold(self, k):
        """
        Generate nested k-fold cross validation
        :param k: number of folds
        :return: train_data, validation_data, test_data
        """
        fold_size = int(len(self.all_data) / k)
        self.all_data = np.random.permutation(self.all_data)

        # do a copy and split before the inner loop begins
        self.test_data, self.train_data = np.split(self.all_data.copy(), [fold_size], axis=0)

        # outer loop
        for j in range(k):
            inner_fold_size = int(len(self.train_data) / k)
            self.valid_data, self.inner_train_data = np.split(self.train_data.copy(), [inner_fold_size], axis=0)

            # inner loop
            for i in range(k - 1):
                yield self.inner_train_data, self.valid_data, self.test_data, j, i
                if i + 1 != k - 1:
                    self.inner_train_data[i * inner_fold_size:(i + 1) * inner_fold_size], self.valid_data = \
                        self.valid_data, self.inner_train_data[i * inner_fold_size:(i + 1) * inner_fold_size].copy()
            if j + 1 != k:
                self.train_data[j * fold_size:(j + 1) * fold_size], self.test_data = \
                    self.test_data, self.train_data[j * fold_size:(j + 1) * fold_size].copy()
