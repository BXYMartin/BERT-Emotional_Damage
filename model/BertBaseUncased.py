from loader.base import BaseLoader
from transformers import BertForSequenceClassification, AdamW, BertConfig, TrainingArguments, Trainer
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from datasets import Dataset, load_metric
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import numpy as np
import pandas as pd
import os
import tqdm


class BertBaseUncased:
    train_epochs = 4
    batch_size = 8
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    def __init__(self, loader: BaseLoader, load_existing=False):
        self.data_loader = loader
        model_name = "bert-base-uncased"
        local_files_only = False
        if load_existing:
            model_name = os.path.join(self.data_loader.storage_folder, "output")
            local_files_only = True
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,    # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=2,           # The number of output labels--2 for binary classification.
            output_attentions=False,    # Whether the model returns attentions weights.
            output_hidden_states=False, # Whether the model returns all hidden-states.
            local_files_only=local_files_only
        )
        self.model.cuda()
        self.optimizer = AdamW(self.model.parameters(),
                               lr=2e-7,
                               eps=1e-8)

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision_metric = load_metric("precision")
        recall_metric = load_metric("recall")
        f1_metric = load_metric("f1")
        precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
        recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
        f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
        return {"precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def tokenize_function(examples):
        return BertBaseUncased.tokenizer(examples['text'], padding="max_length", truncation=True, add_special_tokens=True)

    def train(self):
        self.train_dataset = Dataset.from_pandas(pd.DataFrame(self.data_loader.train_data))
        self.encoded_train_dataset = self.train_dataset.map(self.tokenize_function, batched=True)
        print(self.encoded_train_dataset)
        print(self.encoded_train_dataset[0])
        self.test_dataset = Dataset.from_pandas(pd.DataFrame(self.data_loader.test_data))
        self.encoded_test_dataset = self.test_dataset.map(self.tokenize_function, batched=True)
        self.train_loader = DataLoader(self.encoded_train_dataset,
                                       sampler=RandomSampler(self.encoded_train_dataset),
                                       batch_size=self.batch_size)
        self.test_loader = DataLoader(self.encoded_test_dataset,
                                       sampler=RandomSampler(self.encoded_test_dataset),
                                       batch_size=self.batch_size)
        self.total_steps = len(self.train_loader) * self.train_epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=0,
                                                         num_training_steps=self.total_steps)
        for epoch in range(self.train_epochs):
            self.model.train()
            with tqdm.tqdm(self.train_loader, unit="batch") as tepoch:
                for i, data in enumerate(tepoch):
                    self.model.zero_grad()
                    result = self.model(torch.stack(data['input_ids']).T.cuda(),
                                        token_type_ids=None,
                                        attention_mask=torch.stack(data['attention_mask']).T.cuda(),
                                        labels=torch.tensor(data['label']).cuda(),
                                        return_dict=True)
                    loss = result.loss
                    logits = result.logits
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    if i % 5 == 0:
                        tepoch.set_description(f"Epoch {epoch}")
                        tepoch.set_postfix(Loss=loss.item())

            # Evaluation
            self.model.eval()
            labels = np.array([])
            predictions = np.array([])
            eval_loss = 0
            with tqdm.tqdm(self.test_loader, unit="batch") as tepoch:
                for i, data in enumerate(tepoch):
                    with torch.no_grad():
                        result = self.model(torch.stack(data['input_ids']).T.cuda(),
                                            token_type_ids=None,
                                            attention_mask=torch.stack(data['attention_mask']).T.cuda(),
                                            labels=torch.tensor(data['label']).cuda(),
                                            return_dict=True)
                        loss = result.loss
                        logits = result.logits
                        eval_loss += loss.item()
                        onehot = np.argmax(logits.detach().cpu().numpy(), axis=-1)
                        labels = np.concatenate([labels, data['label'].numpy()])
                        predictions = np.concatenate([predictions, onehot])
                        tepoch.set_description(f"Evaluation {epoch}")
                        tepoch.set_postfix(Loss=loss.item())
            self.data_loader.eval(labels, predictions)
        self.model.save_pretrained(os.path.join(self.data_loader.storage_folder, "output"))

    def predict(self):
        self.test_dataset = Dataset.from_pandas(pd.DataFrame(self.data_loader.test_data))
        self.encoded_test_dataset = self.test_dataset.map(self.tokenize_function, batched=True)
        self.test_loader = DataLoader(self.encoded_test_dataset,
                                      sampler=RandomSampler(self.encoded_test_dataset),
                                      batch_size=self.batch_size)
        self.model.eval()
        labels = np.array([])
        predictions = np.array([])
        eval_loss = 0
        with tqdm.tqdm(self.test_loader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                with torch.no_grad():
                    result = self.model(torch.stack(data['input_ids']).T.cuda(),
                                        token_type_ids=None,
                                        attention_mask=torch.stack(data['attention_mask']).T.cuda(),
                                        labels=torch.tensor(data['label']).cuda(),
                                        return_dict=True)
                    loss = result.loss
                    logits = result.logits
                    eval_loss += loss.item()
                    onehot = np.argmax(logits.detach().cpu().numpy(), axis=-1)
                    labels = np.concatenate([labels, data['label'].numpy()])
                    predictions = np.concatenate([predictions, onehot])
                    tepoch.set_description(f"Prediction")
                    tepoch.set_postfix(Loss=loss.item())
        self.data_loader.eval(labels, predictions)

    def final(self):
        self.final_dataset = Dataset.from_pandas(pd.DataFrame(self.data_loader.final_data))
        self.encoded_final_dataset = self.final_dataset.map(self.tokenize_function, batched=True)
        self.final_loader = DataLoader(self.encoded_final_dataset,
                                      sampler=SequentialSampler(self.encoded_final_dataset),
                                      batch_size=self.batch_size)
        self.model.eval()
        predictions = np.array([])
        with tqdm.tqdm(self.final_loader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                with torch.no_grad():
                    result = self.model(torch.stack(data['input_ids']).T.cuda(),
                                        token_type_ids=None,
                                        attention_mask=torch.stack(data['attention_mask']).T.cuda(),
                                        return_dict=True)
                    loss = result.loss
                    logits = result.logits
                    onehot = np.argmax(logits.detach().cpu().numpy(), axis=-1)
                    predictions = np.concatenate([predictions, onehot])
                    tepoch.set_description(f"Final")
                    tepoch.set_postfix(Loss=loss.item())
        self.data_loader.final(predictions)
        self.prediction = predictions
        print(predictions)
