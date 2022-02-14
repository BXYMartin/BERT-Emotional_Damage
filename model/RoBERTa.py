from simpletransformers.classification import ClassificationModel, ClassificationArgs, MultiLabelClassificationModel, MultiLabelClassificationArgs
import pandas as pd
import os
import logging
import torch
from collections import Counter
from ast import literal_eval
from loader.base import BaseLoader


class RoBERTa:
    def __init__(self, loader: BaseLoader):
        self.data_loader = loader
        self.model_args = ClassificationArgs(num_train_epochs=5,
                                             best_model_dir=os.path.join(loader.storage_folder, "output", "best_model"),
                                             cache_dir=os.path.join(loader.storage_folder, "output", "cache"),
                                             output_dir=os.path.join(loader.storage_folder, "output"),
                                             use_multiprocessing=True,
                                             save_best_model=True,
                                             overwrite_output_dir=True)
        self.model = ClassificationModel("roberta",
                                          'roberta-base',
                                          args=self.model_args,
                                          num_labels=2,
                                          use_cuda=True)

    def train(self):
        # Train model
        self.model.train_model(self.data_loader.train_data[['text', 'label']])

    def predict(self):
        self.prediction, _ = self.model.predict(self.data_loader.test_data.text.tolist())
