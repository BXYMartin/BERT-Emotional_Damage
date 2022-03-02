#!/usr/bin/env python
import sys
import os
import os.path
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import logging
import pandas as pd
from spec.task import DontPatronizeMe
from datetime import datetime

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


class BaseLoader:
    base_dir = "runtime"
    data_dir = "data"
    prob_dir = "resource/data"
    final_prob_dir = "final_prob"
    name = "Base"
    score_filename = "score.txt"
    final_filename = "task1.txt"
    prob_filename = "probs.txt"
    label_filename = "labels.txt"
    train_split_filename = "train_semeval_parids-labels.csv"
    dev_split_filename = "dev_semeval_parids-labels.csv"
    all_data_filename = "dontpatronizeme_pcl.tsv"
    test_data_filename = "dontpatronizeme_test.tsv"

    inner_train_data = []
    train_data = []
    test_data = []
    all_data = []

    def __init__(self, token=""):
        # Initialize directory
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)
        date_token = token
        if len(token) == 0:
            date_token = self.name + " "
            date_token += datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.storage_folder = os.path.join(self.base_dir, date_token)
        logging.info(f"Initialized score directory in {self.storage_folder}")
        self.ref_dir = os.path.join(self.storage_folder, "ref")
        self.res_dir = os.path.join(self.storage_folder, "res")
        if not os.path.exists(self.storage_folder):
            os.mkdir(self.storage_folder)
        else:
            logging.warning(f"Directory {self.storage_folder} already exists!")
        if not os.path.exists(self.ref_dir):
            os.mkdir(self.ref_dir)
        if not os.path.exists(self.res_dir):
            os.mkdir(self.res_dir)

        # Initialize task dataset
        task = DontPatronizeMe(os.path.join(self.data_dir), os.path.join(self.data_dir, self.test_data_filename))
        task.load_task1()
        task.load_test()
        task.load_task2(return_one_hot=False)
        self.all_data = task.train_task1_df
        self.final_data = task.test_set_df
        self.all_categorical_data = task.train_task2_df

        logging.debug(f"All data header:")
        logging.debug(self.all_data.head())
        logging.debug(f"Final evaluation data header:")
        logging.debug(self.final_data.head())

    def final_prob(self, probs):
        prob_file_dir = os.path.join(self.storage_folder, self.final_prob_dir)
        if not os.path.exists(prob_file_dir):
            os.mkdir(prob_file_dir)
        else:
            logging.warning(f"Directory {prob_file_dir} already exists!")
        prob_file_path = os.path.join(self.storage_folder, self.final_prob_dir, self.prob_filename)
        np.savetxt(prob_file_path, probs)
        logging.info(f"Final probabilities files written to {prob_file_dir}.")

    def final_convert(self, threshold=0.9):
        final_prob_file_dir = os.path.join(self.storage_folder, self.final_prob_dir, self.prob_filename)
        probs = np.loadtxt(final_prob_file_dir)
        prediction = np.zeros((probs.shape[0],))
        for index in range(probs.shape[0]):
            if probs[index, 1] > threshold:
                prediction[index] = 1
        self.final(prediction, epoch_num='{}_{}'.format(999, threshold))

    def final(self, labels, epoch_num=''):
        file_path = os.path.join(self.res_dir, self.final_filename)
        file_path += f'_{epoch_num}'
        with open(file_path, "w") as final_file:
            for label in labels:
                final_file.write(str(int(label)) + "\n")
        logging.critical(f"Final result written to {file_path}")

    def prob(self, labels, probs):
        prob_file_dir = os.path.join(self.storage_folder, self.prob_dir)
        if not os.path.exists(prob_file_dir):
            os.mkdir(prob_file_dir)
        else:
            logging.warning(f"Directory {prob_file_dir} already exists!")
        prob_file_path = os.path.join(self.storage_folder, self.prob_dir, self.prob_filename)
        label_file_path = os.path.join(self.storage_folder, self.prob_dir, self.label_filename)
        np.savetxt(label_file_path, labels)
        np.savetxt(prob_file_path, probs)
        logging.info(f"Labels and probabilities files written to {prob_file_dir}.")

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

    def eval_per(self, labels, predictions, class_name, class_value):
        task_confusion_matrix = confusion_matrix(labels, predictions)
        task_precision = precision_score(labels, predictions)
        task_recall = recall_score(labels, predictions)
        task_f1 = f1_score(labels, predictions)
        filename = "score" + f'{class_name}_{class_value}.txt'
        file_path = os.path.join(self.storage_folder, filename)
        with open(file_path, "w") as score_file:
            score_file.write('task1_precision:' + str(task_precision) + '\n')
            score_file.write('task1_recall:' + str(task_recall) + '\n')
            score_file.write('task1_f1:' + str(task_f1) + '\n')
            score_file.write(f'{class_name}_{class_value} total_samples:' + str(len(labels)) + '\n')

        logging.info(f'{class_name}_{class_value} result:')
        logging.info(f"Confusion matrix")
        logging.info(task_confusion_matrix)
        logging.info(f"Precision score")
        logging.info(task_precision)
        logging.info(f"Recall score")
        logging.info(task_recall)
        logging.info(f"F1 score")
        logging.info(task_f1)
        logging.info(f"Score file written to {file_path}")
