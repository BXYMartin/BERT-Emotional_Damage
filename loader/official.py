#!/usr/bin/env python
import sys
import os
import os.path
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import logging
import pandas as pd
from model.BackTranslate import BackTranslate
from loader.base import BaseLoader
from spec.task import DontPatronizeMe
from datetime import datetime


class OfficialLoader(BaseLoader):
    name = "Split"
    augmentation_data_filename = "back_translation_balanced_dataset.csv"

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
        self.train_data = self.train_data.sample(frac=1, axis=1).reset_index(drop=True)

    def augmentation(self):
        if os.path.isfile(os.path.join(self.data_dir, self.augmentation_data_filename)):
            logging.info(f"Using cached balanced dataset from {os.path.join(self.data_dir, self.augmentation_data_filename)}")
            self.train_data = pd.read_csv(os.path.join(self.data_dir, self.augmentation_data_filename))
            logging.info(f"Cached dataset: Positive/Negative {len(self.train_data[self.train_data.label == 1])}/{len(self.train_data[self.train_data.label == 0])}")
            #self.train_data = self.train_data.drop_duplicates()
            #logging.info(
            #    f"No duplicated dataset: Positive/Negative {len(self.train_data[self.train_data.label == 1])}/{len(self.train_data[self.train_data.label == 0])}")
            # self.train_data.to_csv(os.path.join(self.data_dir, self.augmentation_data_filename))
            return
        positive_class = self.train_data[self.train_data.label == 1]
        negative_class = self.train_data[self.train_data.label == 0]
        positive_label = len(positive_class)
        negative_label = len(negative_class)
        minimum_label = min(positive_label, negative_label)
        maximum_label = max(positive_label, negative_label)
        target_class = positive_class
        other_class = negative_class
        target_label = 1
        if minimum_label == negative_label:
            target_class = negative_class
            other_class = positive_class
            target_label = 0
        sources = target_class["text"].to_list()
        print(sources[:5])
        target_langs = ["fr", "es", "it", "pt", "ro", "ca", "gl", "la"]
        model = BackTranslate()
        for round in range(int(maximum_label / minimum_label) + 1):
            target_lang = target_langs[round % len(target_langs)]
            logging.info(f"Data augumentation round {round}, intermediate language {target_lang}")
            translated = model.back_translate(sources, target_lang=target_lang)
            insertion = []
            for text in translated:
                insertion.append({
                    'par_id': 0,
                    'text': text,
                    'label': target_label
                })
            logging.info(f"Generated {len(insertion)} ({target_label}) samples")
            target_class = target_class.append(insertion)
        logging.info(f"After augumentation: {len(target_class)}/{len(other_class)} samples")
        self.train_data = pd.concat([target_class, other_class])
        self.train_data.to_csv(os.path.join(self.data_dir, self.augmentation_data_filename))
        self.train_data = self.train_data.sample(frac=1, axis=1).reset_index(drop=True)

