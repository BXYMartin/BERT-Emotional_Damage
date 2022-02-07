#!/usr/bin/env python
import sys
import os
import os.path
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import logging
from loader.task import DontPatronizeMe
from datetime import datetime


class Official:
    base_dir = "runtime"
    data_dir = "data"
    score_filename = "score.txt"
    final_filename = "task1.txt"
    all_data_filename = "dontpatronizeme_pcl.tsv"
    test_data_filename = "dontpatronizeme_test.tsv"

    inner_train_data = []
    train_data = []
    test_data = []
    all_data = []

    def __init__(self, test=False):
        # Initialize directory
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        date_token = "Dev " if test else "Test "
        date_token += datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.storage_folder = os.path.join(self.base_dir, date_token)
        logging.info(f"Initialized score directory in {self.storage_folder}")
        self.ref_dir = os.path.join(self.storage_folder, "ref")
        self.res_dir = os.path.join(self.storage_folder, "res")
        if not os.path.exists(self.storage_folder):
            os.mkdir(self.storage_folder)
        else:
            raise RuntimeError(f"Directory {self.storage_folder} already exists!")
        if not os.path.exists(self.ref_dir):
            os.mkdir(self.ref_dir)
        if not os.path.exists(self.res_dir):
            os.mkdir(self.res_dir)

        # Initialize task dataset
        task = DontPatronizeMe(os.path.join(self.data_dir), os.path.join(self.data_dir, self.test_data_filename))
        task.load_task1()
        task.load_test()
        self.all_data = task.train_task1_df
        self.final_data = task.test_set_df

        logging.debug(f"All data header:")
        logging.debug(self.all_data.head())
        logging.debug(f"Final evaluation data header:")
        logging.debug(self.final_data.head())

    def final(self, labels):
        file_path = os.path.join(self.res_dir, self.final_filename)
        with open(file_path, "w") as final_file:
            for label in labels:
                final_file.write(str(int(label)) + "\n")
        logging.info(f"Final result written to {file_path}")

    def eval(self, labels, predictions):
        task_confusion_matrix = confusion_matrix(labels, predictions)
        task_precision = precision_score(labels, predictions)
        task_recall = recall_score(labels, predictions)
        task_f1 = f1_score(labels, predictions)

        file_path = os.path.join(self.storage_folder, self.score_filename)
        with open(file_path, "w") as score_file:
            score_file.write('task1_precision:' + str(task_precision) + '\n')
            score_file.write('task1_recall:' + str(task_recall) + '\n')
            score_file.write('task1_f1:' + str(task_f1) + '\n')

        logging.info(f"Confusion matrix")
        logging.info(task_confusion_matrix)
        logging.info(f"Precision score")
        logging.info(task_precision)
        logging.info(f"Recall score")
        logging.info(task_recall)
        logging.info(f"F1 score")
        logging.info(task_f1)
        logging.info(f"Score file written to {file_path}")

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
