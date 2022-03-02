import transformers

from loader.base import BaseLoader
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, AdamW, BertConfig, TrainingArguments, \
    Trainer
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from datasets import Dataset, load_metric
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import numpy as np
import pandas as pd
import os
import tqdm


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


class DebertaV2XXLarge:
    train_epochs = 3
    eval_while_training = True
    eval_step_size = 1200
    batch_size = 2
    tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v2-xxlarge")

    def __init__(self, loader: BaseLoader, load_existing=False):
        self.data_loader = loader
        model_name = "microsoft/deberta-v2-xxlarge"
        local_files_only = False
        if load_existing:
            model_name = os.path.join(self.data_loader.storage_folder, "output")
            local_files_only = True
        self.model = DebertaV2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
            
            local_files_only=local_files_only
        )
        self.model.half()
        self.model.cuda()

        # def get_parameters(model, model_init_lr, multiplier, classifier_lr):
        #     parameters = []
        #     lr = model_init_lr
        #     for layer in range(12, -1, -1):
        #         layer_params = {
        #             'params': [p for n, p in model.named_parameters() if f'encoder.layer.{layer}.' in n],
        #             'lr': lr
        #         }
        #         parameters.append(layer_params)
        #         lr *= multiplier
        #     classifier_params = {
        #         'params': [p for n, p in model.named_parameters() if 'layer_norm' in n or 'linear' in n
        #                    or 'pooling' in n],
        #         'lr': classifier_lr
        #     }
        #     parameters.append(classifier_params)
        #     return parameters
        #
        # parameters = get_parameters(self.model, 2e-5, 0.95, 1e-4)
        # self.optimizer = AdamW(parameters)
        self.optimizer = AdamW(self.model.parameters(),
                               lr=2e-5, eps=1e-4)

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
        return DebertaV2XXLarge.tokenizer(examples['text'],
                                              add_special_tokens=True,
                                              padding='max_length',
                                              max_length=512,
                                              return_attention_mask=True,
                                              truncation=True)

    def train(self):
        self.train_dataset = Dataset.from_pandas(pd.DataFrame(self.data_loader.train_data))
        # self.encoded_train_dataset = self.train_dataset.map(self.tokenize_function, batched=True)
        self.encoded_train_dataset = self.train_dataset.map(self.tokenize_function, batched=True)
        print(f'len(self.encoded_train_dataset) = {len(self.encoded_train_dataset)}')
        print(f'len(self.data_loader.train_data) = {len(self.data_loader.train_data)}')
        # print(self.encoded_train_dataset[0])
        self.test_dataset = Dataset.from_pandas(pd.DataFrame(self.data_loader.test_data))
        # self.encoded_test_dataset = self.test_dataset.map(self.tokenize_function, batched=True)
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
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    if i % 5 == 0:
                        tepoch.set_description(f"Epoch {epoch}")
                        tepoch.set_postfix(Loss=loss.item())
                    if i % self.eval_step_size == 0 and epoch >= 0 and i != 0:
                        # Performing eval in the middle of training

                        # Evaluation
                        self.model.eval()
                        labels = np.array([])
                        predictions = np.array([])
                        eval_loss = 0
                        # total_eval_accuracy = 0
                        with tqdm.tqdm(self.test_loader, unit="batch") as tepoch:
                            for _, data in enumerate(tepoch):
                                with torch.no_grad():
                                    result = self.model(torch.stack(data['input_ids']).T.cuda(),
                                                        token_type_ids=None,
                                                        attention_mask=torch.stack(data['attention_mask']).T.cuda(),
                                                        labels=torch.tensor(data['label']).cuda(),
                                                        return_dict=True)
                                    loss = result.loss
                                    logits = result.logits
                                    label_ids = torch.tensor(data['label']).numpy()
                                    # total_eval_accuracy += flat_accuracy(logits, label_ids)
                                    eval_loss += loss.item()
                                    onehot = np.argmax(logits.detach().cpu().numpy(), axis=-1)
                                    labels = np.concatenate([labels, data['label'].numpy()])
                                    predictions = np.concatenate([predictions, onehot])
                                    tepoch.set_description(f"Evaluation {epoch}")
                                    tepoch.set_postfix(Loss=loss.item())
                        # avg_val_accuracy = total_eval_accuracy / len(self.test_loader)
                        # avg_val_loss = eval_loss / len(self.test_loader)

                        self.data_loader.eval(labels, predictions)
                        self.final("{}-{}".format(epoch, i))
                        self.model.save_pretrained(os.path.join(self.data_loader.storage_folder, "output", "checkpoint-{}-{}".format(epoch, i)))



            # Evaluation
            self.model.eval()
            labels = np.array([])
            predictions = np.array([])
            eval_loss = 0
            # total_eval_accuracy = 0
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
                        label_ids = torch.tensor(data['label']).numpy()
                        # total_eval_accuracy += flat_accuracy(logits, label_ids)
                        eval_loss += loss.item()
                        onehot = np.argmax(logits.detach().cpu().numpy(), axis=-1)
                        labels = np.concatenate([labels, data['label'].numpy()])
                        predictions = np.concatenate([predictions, onehot])
                        tepoch.set_description(f"Evaluation {epoch}")
                        tepoch.set_postfix(Loss=loss.item())
            # avg_val_accuracy = total_eval_accuracy / len(self.test_loader)
            # avg_val_loss = eval_loss / len(self.test_loader)

            self.data_loader.eval(labels, predictions)
            self.final(epoch)
            self.model.save_pretrained(os.path.join(self.data_loader.storage_folder, "output", "checkpoint-{}".format(epoch)))

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

    def final(self, epoch_num=''):
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
                    tepoch.set_postfix(Loss=loss)
        self.prediction = predictions
        self.data_loader.final(predictions, epoch_num)
        print(predictions)
