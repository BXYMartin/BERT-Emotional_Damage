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
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from loader.dont_patronize_me import DontPatronizeMe

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


class OfficialLoader(BaseLoader):
    name = "Split"
    augmentation_data_filename = "back_translation_balanced_dataset.csv"
    official_train_data_filename = "official_split_train_dataset_AAA.csv"
    official_train_data_cleaned_filename = "official_split_train_dataset_AAA_cleaned.csv"
    official_train_data_all_filename = "official_split_train_dataset_AAA_all.csv"
    official_train_data_all_cleaned_filename = "official_split_train_dataset_AAA_all_cleaned.csv"
    official_train_data_truncation_filename = "official_split_train_dataset_AAA_truncation.csv"
    official_test_data_filename = "official_split_test_dataset.csv"
    official_test_data_cleaned_filename = "official_split_test_dataset_cleaned.csv"
    official_test_data_truncation_filename = "official_split_test_dataset_truncation.csv"
    augmentation_data_all_filename = "back_translation_balanced_dataset_all.csv"
    official_final_data_truncation_filename = "official_final_dataset_truncation.csv"

    def __init__(self, token=""):
        super().__init__(token)

    def split(self):
        if os.path.isfile(os.path.join(self.data_dir, self.official_train_data_filename)) and os.path.isfile(
                os.path.join(self.data_dir, self.official_test_data_filename)):
            logging.info(f"Using cached official split files.")
            self.train_data = pd.read_csv(os.path.join(self.data_dir, self.official_train_data_filename))
            self.test_data = pd.read_csv(os.path.join(self.data_dir, self.official_test_data_filename))
            logging.info(f"Loaded cached files TEST({len(self.train_data)})/DEV({len(self.test_data)}).")
            print(f"[split] Loaded cached files TEST({len(self.train_data)})/DEV({len(self.test_data)}).")
            return
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
        self.train_data.to_csv(os.path.join(self.data_dir, self.official_train_data_filename))
        self.test_data.to_csv(os.path.join(self.data_dir, self.official_test_data_filename))
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
            logging.info(
                f"Using cached balanced dataset from {os.path.join(self.data_dir, self.augmentation_data_filename)}")
            self.train_data = pd.read_csv(os.path.join(self.data_dir, self.augmentation_data_filename))
            logging.info(
                f"Cached dataset: Positive/Negative {len(self.train_data[self.train_data.label == 1])}/{len(self.train_data[self.train_data.label == 0])}")
            # self.train_data = self.train_data.drop_duplicates()
            # logging.info(
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

    def split_upsample_truncation(self):
        if os.path.isfile(os.path.join(self.data_dir, self.official_train_data_truncation_filename)) and os.path.isfile(
                os.path.join(self.data_dir, self.official_test_data_filename)):
            self.train_data = pd.read_csv(os.path.join(self.data_dir, self.official_train_data_truncation_filename))
            logging.info(f"[split_upsample_truncation] Using cached train_data: {self.official_train_data_truncation_filename}")
            self.test_data = pd.read_csv(os.path.join(self.data_dir, self.official_test_data_truncation_filename))
            logging.info(f"[split_upsample_truncation] Using cached test_data: {self.official_test_data_truncation_filename}")
            self.final_data = pd.DataFrame(
                pd.read_csv(os.path.join(self.data_dir, self.official_final_data_truncation_filename)))
            logging.info(
                f"[split_upsample_truncation] Using cached final_data: {self.official_final_data_truncation_filename}")
            logging.info(f"Loaded cached files TEST({len(self.train_data)})/DEV({len(self.test_data)}).")
            print(
                f"[split_upsample_truncation] Loaded cached files TEST({len(self.train_data)})/DEV({len(self.test_data)}).")
            return

    def split_upsample_cleaned(self):
        if os.path.isfile(os.path.join(self.data_dir, self.official_train_data_cleaned_filename)) and os.path.isfile(
                os.path.join(self.data_dir, self.official_test_data_cleaned_filename)):
            logging.info(f"Using cached official split files.")
            self.train_data = pd.read_csv(os.path.join(self.data_dir, self.official_train_data_cleaned_filename))
            self.test_data = pd.read_csv(os.path.join(self.data_dir, self.official_test_data_cleaned_filename))
            logging.info(f"Loaded cached files TEST({len(self.train_data)})/DEV({len(self.test_data)}).")
            print(f"[split_AAA] Loaded cached files TEST({len(self.train_data)})/DEV({len(self.test_data)}).")
            return
        raise NotImplementedError("Upsample Train Dataset not found.")

    def split_upsample(self):
        if os.path.isfile(os.path.join(self.data_dir, self.all_data_filename)) and os.path.isfile(
                os.path.join(self.data_dir, self.official_test_data_filename)):
            logging.info(f"Using cached official split files.")
            self.train_data = pd.read_csv(os.path.join(self.data_dir, self.official_train_data_filename))
            self.test_data = pd.read_csv(os.path.join(self.data_dir, self.official_test_data_filename))
            logging.info(f"Loaded cached files TEST({len(self.train_data)})/DEV({len(self.test_data)}).")
            print(f"[split_AAA] Loaded cached files TEST({len(self.train_data)})/DEV({len(self.test_data)}).")
            return
        logging.info(f"Performing split using official split files.")
        train_ids = pd.read_csv(os.path.join(self.data_dir, self.all_data_filename))
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
        oversampler = RandomOverSampler(sampling_strategy='minority')
        print(f"x_train.shape = {self.train_data.shape}")
        x = np.array(self.train_data[['par_id', 'text']])
        y = np.array(self.train_data['label'])
        print(f"y_numpy = \n{y}")
        oversample_x, oversample_y = oversampler.fit_resample(x,
                                                              y)
        # oversample_x = oversample_x.ravel()
        print(f"oversample_x.shape = {oversample_x.shape}\toversample_y.shape = {oversample_y.shape}")
        print(Counter(oversample_y))
        self.train_data = []
        for i in range(len(oversample_x)):
            # df.at[4, 'B']
            par_id = oversample_x[i][0]
            text = oversample_x[i][1]
            # print(f'par_id={par_id}\ttext={text}')
            label = oversample_y[i]
            # print(f'label = {label}')
            self.train_data.append({
                'par_id': par_id,
                'text': text,
                'label': label
            })
        self.train_data = pd.DataFrame(self.train_data)
        print(self.train_data['label'].value_counts())
        print(f'final self.train_data.head =\n{self.train_data.head}')
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
        self.train_data.to_csv(os.path.join(self.data_dir, self.official_train_data_filename))
        self.test_data.to_csv(os.path.join(self.data_dir, self.official_test_data_filename))
        logging.info(f"Successfully split TEST({len(self.train_data)})/DEV({len(self.test_data)}).")

        
    def split_upsample_all_cleaned(self):
        if os.path.isfile(os.path.join(self.data_dir, self.official_train_data_all_cleaned_filename)):
            logging.info(f"Using cached official split files: {self.official_train_data_all_filename}")
            self.train_data = pd.read_csv(os.path.join(self.data_dir, self.official_train_data_all_cleaned_filename))
            return
        raise NotImplementedError("Upsample Cleaned Dataset not found.")


    def split_upsample_all(self):
        if os.path.isfile(os.path.join(self.data_dir, self.official_train_data_all_filename)):
            logging.info(f"Using cached official split files: {self.official_train_data_all_filename}")
            self.train_data = pd.read_csv(os.path.join(self.data_dir, self.official_train_data_all_filename))
            return
        dpm = DontPatronizeMe(self.data_dir, 'dontpatronizeme_pcl.tsv')
        dpm.load_task1()
        train_df = dpm.train_task1_df
        print(train_df.head())
        logging.info(f"Performing split using official split files.")
        self.train_data = train_df
        oversampler = RandomOverSampler(sampling_strategy='minority')
        print(f"x_train.shape = {self.train_data.shape}")
        x = np.array(self.train_data[['par_id', 'text']])
        y = np.array(self.train_data['label'])
        oversample_x, oversample_y = oversampler.fit_resample(x,
                                                              y)
        # oversample_x = oversample_x.ravel()
        print(f"oversample_x.shape = {oversample_x.shape}\toversample_y.shape = {oversample_y.shape}")
        print(Counter(oversample_y))
        self.train_data = []
        for i in range(len(oversample_x)):
            # df.at[4, 'B']
            par_id = oversample_x[i][0]
            text = oversample_x[i][1]
            # print(f'par_id={par_id}\ttext={text}')
            label = oversample_y[i]
            # print(f'label = {label}')
            self.train_data.append({
                'par_id': par_id,
                'text': text,
                'label': label
            })
        self.train_data = pd.DataFrame(self.train_data)
        print(self.train_data['label'].value_counts())
        print(f'final self.train_data.head =\n{self.train_data.head}')
        self.train_data.to_csv(os.path.join(self.data_dir, self.official_train_data_all_filename))
        logging.info(f"Successfully split TEST({len(self.train_data)}).")

    def augmentation_all(self):
        if os.path.isfile(os.path.join(self.data_dir, self.augmentation_data_all_filename)):
            logging.info(
                f"Using cached balanced dataset from {os.path.join(self.data_dir, self.augmentation_data_all_filename)}")
            self.train_data = pd.read_csv(os.path.join(self.data_dir, self.augmentation_data_all_filename))
            logging.info(
                f"Cached dataset: Positive/Negative {len(self.train_data[self.train_data.label == 1])}/{len(self.train_data[self.train_data.label == 0])}")
            # self.train_data = self.train_data.drop_duplicates()
            # logging.info(
            #    f"No duplicated dataset: Positive/Negative {len(self.train_data[self.train_data.label == 1])}/{len(self.train_data[self.train_data.label == 0])}")
            # self.train_data.to_csv(os.path.join(self.data_dir, self.augmentation_data_filename))
            return
        positive_class = self.train_data[self.train_data.label == 1]
        negative_class = self.train_data[self.train_data.label == 0]
        positive_label_len = len(positive_class)
        negative_label_len = len(negative_class)
        target_label_size = 2 * max(positive_label_len, negative_label_len)
        target_label = 1

        sources = positive_class["text"].to_list()
        print(sources[:5])
        target_langs = ["fr", "es", "it", "pt", "ro", "ca", "gl", "la"]
        model = BackTranslate()
        for round in range(int(target_label_size / positive_label_len) - 1):
            target_lang = target_langs[round % len(target_langs)]
            logging.info(f"Positive Data augumentation round {round}, intermediate language {target_lang}")
            translated = model.back_translate(sources, target_lang=target_lang)
            insertion = []
            for text in translated:
                insertion.append({
                    'par_id': 0,
                    'text': text,
                    'label': target_label
                })
            logging.info(f"Positive Generated {len(insertion)} ({target_label}) samples")
            positive_class = positive_class.append(insertion)

        target_label = 0

        for round in range(int(target_label_size / negative_label_len) - 1):
            target_lang = target_langs[round % len(target_langs)]
            logging.info(f"Negative Data augumentation round {round}, intermediate language {target_lang}")
            translated = model.back_translate(sources, target_lang=target_lang)
            insertion = []
            for text in translated:
                insertion.append({
                    'par_id': 0,
                    'text': text,
                    'label': target_label
                })
            logging.info(f"Negative Generated {len(insertion)} ({target_label}) samples")
            negative_class = negative_class.append(insertion)

        logging.info(f"After augumentation: {len(positive_class)}/{len(negative_class)} samples")
        self.train_data = pd.concat([positive_class, negative_class])
        self.train_data.to_csv(os.path.join(self.data_dir, self.augmentation_data_all_filename))
        self.train_data = self.train_data.sample(frac=1, axis=1).reset_index(drop=True)

    def process(self, input_name, output_name):
        if os.path.isfile(os.path.join(self.data_dir, output_name)):
            logging.info(
                f"[process] output file exists: {os.path.join(self.data_dir, output_name)}")
            return
        if os.path.isfile(os.path.join(self.data_dir, input_name)):

            logging.info(
                f"[process] input from {os.path.join(self.data_dir, input_name)}")
            if input_name[-3:] == 'tsv':
                self.train_data = pd.read_csv(os.path.join(self.data_dir, input_name), sep='\t',
                                              names=['par_id', 'art_id', 'keyword', 'country', 'text'])
            else:
                self.train_data = pd.read_csv(os.path.join(self.data_dir, input_name))
            # self.train_data = self.train_data.drop_duplicates()
            # logging.info(
            #    f"No duplicated dataset: Positive/Negative {len(self.train_data[self.train_data.label == 1])}/{len(self.train_data[self.train_data.label == 0])}")
            # self.train_data.to_csv(os.path.join(self.data_dir, self.augmentation_data_filename))
            print((self.train_data.head()))
            count = 0
            for idx, row in self.train_data.iterrows():
                if not isinstance(row['text'], str):
                    continue
                if len(row['text']) > 512:
                    count += 1
                    self.train_data.loc[idx, 'text'] = row['text'][:128] + row['text'][-384:]
                    # row['text'] = row['text'][:128] + row['text'][-384:]
            print(f'[process] before truncation, count = {count}\n')
            count = 0
            for idx, row in self.train_data.iterrows():
                if not isinstance(row['text'], str):
                    continue
                if len(row['text']) > 512:
                    count += 1
            print(f'[process] after truncation, count = {count}\n')
            print((self.train_data.head()))
            if 'Unnamed: 0' in self.train_data.columns:
                self.train_data = self.train_data.drop(columns=['Unnamed: 0'])
            print((self.train_data.head()))
            self.train_data.to_csv(os.path.join(self.data_dir, output_name))
        # if os.path.isfile(os.path.join(self.data_dir, self.augmentation_data_all_filename)):
        #     logging.info(
        #         f"[process] Using cached balanced dataset from {os.path.join(self.data_dir, self.augmentation_data_all_filename)}")
        #     self.train_data = pd.read_csv(os.path.join(self.data_dir, self.augmentation_data_all_filename))
        #     logging.info(
        #         f"[process] Cached dataset: Positive/Negative {len(self.train_data[self.train_data.label == 1])}/{len(self.train_data[self.train_data.label == 0])}")
        #     # self.train_data = self.train_data.drop_duplicates()
        #     # logging.info(
        #     #    f"No duplicated dataset: Positive/Negative {len(self.train_data[self.train_data.label == 1])}/{len(self.train_data[self.train_data.label == 0])}")
        #     # self.train_data.to_csv(os.path.join(self.data_dir, self.augmentation_data_filename))
        #     print((self.train_data.head()))
        #     count = 0
        #     for idx, row in self.train_data.iterrows():
        #         if len(row['text']) > 512:
        #             count += 1
        #             self.train_data.loc[idx, 'text'] = row['text'][:128] + row['text'][-384:]
        #             # row['text'] = row['text'][:128] + row['text'][-384:]
        #     print(f'[process] before truncation, count = {count}\n')
        #     count = 0
        #     for idx, row in self.train_data.iterrows():
        #         if len(row['text']) > 512:
        #             count += 1
        #     print(f'[process] after truncation, count = {count}\n')
        #     print((self.train_data.head()))
        #     self.train_data = self.train_data.drop(columns=['Unnamed: 0'])
        #     print((self.train_data.head()))
        #     self.train_data.to_csv(os.path.join(self.data_dir, self.augmentation_data_all_filename))
