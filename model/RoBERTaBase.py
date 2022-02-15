from loader.base import BaseLoader
from transformers import BertForSequenceClassification, AdamW, BertConfig, RobertaTokenizer, RobertaModel, TrainingArguments, Trainer
from datasets import Dataset, load_metric
import numpy as np
import pandas as pd
import os


class RoBERTaBase:
    precision_metric = load_metric("precision", cache_dir="/vol/bitbucket/mb220/.cache")
    recall_metric = load_metric("recall", cache_dir="/vol/bitbucket/mb220/.cache")
    f1_metric = load_metric("f1", cache_dir="/vol/bitbucket/mb220/.cache")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def __init__(self, loader: BaseLoader, load_existing=False):
        self.data_loader = loader
        self.training_args = TrainingArguments("roberta_trainer",
                                               num_train_epochs=4,
                                               logging_dir=os.path.join(self.data_loader.storage_folder, "log"),
                                               logging_steps=10,
                                               load_best_model_at_end=True,
                                               evaluation_strategy="epoch",
                                               save_strategy="epoch",
                                               per_device_train_batch_size=4
                                               )
        model_name = "roberta-base"
        local_files_only = False
        if load_existing:
            model_name = os.path.join(self.data_loader.storage_folder, "output")
            local_files_only = True
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,    # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=2,           # The number of output labels--2 for binary classification.
            output_attentions=False,    # Whether the model returns attentions weights.
            output_hidden_states=False, # Whether the model returns all hidden-states.
            cache_dir="/vol/bitbucket/mb220/.cache",
            local_files_only=local_files_only
        )
        self.model.cuda()

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        precision = RoBERTaBase.precision_metric.compute(predictions=predictions, references=labels)["precision"]
        recall = RoBERTaBase.recall_metric.compute(predictions=predictions, references=labels)["recall"]
        f1 = RoBERTaBase.f1_metric.compute(predictions=predictions, references=labels)["f1"]
        return {"precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def tokenize_function(examples):
        return RoBERTaBase.tokenizer(examples['text'], padding="max_length", truncation=True)

    def train(self):
        self.train_dataset = Dataset.from_pandas(pd.DataFrame(self.data_loader.train_data))
        self.encoded_train_dataset = self.train_dataset.map(self.tokenize_function, batched=True)
        print(self.encoded_train_dataset)
        self.test_dataset = Dataset.from_pandas(pd.DataFrame(self.data_loader.test_data))
        print(self.test_dataset)
        self.encoded_test_dataset = self.test_dataset.map(self.tokenize_function, batched=True)
        self.trainer = Trainer(model=self.model, args=self.training_args, train_dataset=self.encoded_train_dataset,
                               eval_dataset=self.encoded_test_dataset, compute_metrics=self.compute_metrics)

        self.trainer.train()
        self.trainer.save_model(os.path.join(self.data_loader.storage_folder, "output"))
        # alternative saving method and folder
        self.model.save_pretrained(os.path.join(self.data_loader.storage_folder, "output"))

    def predict(self):
        self.test_dataset = Dataset.from_pandas(pd.DataFrame(self.data_loader.test_data))
        print(self.test_dataset)
        self.encoded_test_dataset = self.test_dataset.map(self.tokenize_function, batched=True)
        self.trainer = Trainer(model=self.model, args=self.training_args,
                               eval_dataset=self.encoded_test_dataset, compute_metrics=self.compute_metrics)

        result = self.trainer.evaluate()
        print(result)

    def final(self):
        self.final_dataset = Dataset.from_pandas(pd.DataFrame(self.data_loader.final_data))
        self.encoded_final_dataset = self.final_dataset.map(self.tokenize_function, batched=True)
        print(self.encoded_final_dataset)
        self.trainer = Trainer(model=self.model, args=self.training_args)
        self.prediction, _, _ = self.trainer.predict(test_dataset=self.encoded_final_dataset)
        print(self.prediction)
        self.prediction = np.argmax(self.prediction, axis=-1)
        print(self.prediction)
