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


class OfficialLoader(BaseLoader):
    name = "Split"

    def __init__(self, token=""):
        super().__init__(token)

    def split(self):
        logging.info(f"Performing split using official split files.")
        train_ids = pd.read_csv(os.path.join(self.data_dir, self.train_split_filename))
        dev_ids = pd.read_csv(os.path.join(self.data_dir, self.dev_split_filename))
        train_ids.par_id = train_ids.par_id.astype(str)
        dev_ids.par_id = dev_ids.par_id.astype(str)
        self.train_data = []  # par_id, label and text
        for idx in range(len(train_ids)):
            par_id = train_ids.par_id[idx]
            text = self.all_data.loc[self.all_data.par_id == par_id].text.values[0]
            label = self.all_data.loc[self.all_data.par_id == par_id].label.values[0]
            self.train_data.append({
                'par_id': par_id,
                'text': text,
                'label': label
            })
        self.train_data = pd.DataFrame(self.train_data)
        self.test_data = []  # par_id, label and text
        for idx in range(len(dev_ids)):
            par_id = dev_ids.par_id[idx]
            text = self.all_data.loc[self.all_data.par_id == par_id].text.values[0]
            label = self.all_data.loc[self.all_data.par_id == par_id].label.values[0]
            self.test_data.append({
                'par_id': par_id,
                'text': text,
                'label': label
            })
        self.test_data = pd.DataFrame(self.test_data)
        logging.info(f"Successfully split TEST({len(self.train_data)})/DEV({len(self.test_data)}).")

    def balance(self):
        positive_class = self.train_data[self.train_data.label == 1]
        negative_class = self.train_data[self.train_data.label == 0]
        positive_label = len(positive_class)
        negative_label = len(negative_class)
        minimum_label = min(positive_label, negative_label)
        positive_class = positive_class[:minimum_label]
        negative_class = negative_class[:minimum_label]
        self.train_data = pd.concat([positive_class, negative_class])
