import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from torch.utils.data import Dataset
from datasets import load_dataset
from datasets import load_metric
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import math
import os
import urllib.request
from functools import partial
from urllib.error import HTTPError
from torch import cuda
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

def get_model():

    # Set up GPU

    device = 'cuda' if cuda.is_available() else 'cpu'

    # load model and tokenizer
    roberta = "roberta-base"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    labels = ['Negative', 'Neutral', 'Positive']

    return model, tokenizer, labels



def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
  
   logits, labels = eval_pred
#    labels = ['Negative', 'Neutral', 'Positive']

   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
   return {"accuracy": accuracy, "f1": f1}

def main():
    #
    dataset = load_dataset("sst", "default")
    df_train = pd.DataFrame(dataset['train'])
    df_validation = pd.DataFrame(dataset['validation'])
    df_test = pd.DataFrame(dataset['test'])

    # Shows train / validation / test split
    print(df_train)

    device = 'cuda' if cuda.is_available() else 'cpu'

    # load model and tokenizer
    roberta = "roberta-base"

    model = AutoModelForSequenceClassification.from_pretrained(roberta, num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained(roberta)

    dataset = dataset.remove_columns("tokens")
    dataset = dataset.remove_columns("tree")

    # rating = ["foo"] * len(dataset)
    # dataset = dataset.add_column("rating", rating)

    print(dataset[1])

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # def convert_label(dataset):
    #     'positive' if (dataset["label"] > 0.5) else 'negative'
    #     return dataset

    # train_dataset = train_dataset.map(convert_label)
    # test_dataset = test_dataset.map(convert_label)



    print(train_dataset["label"])

    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # print(TrainingArguments.size(), labels.size())

    repo_name = "results"
 
    training_args = TrainingArguments(
    output_dir= repo_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    push_to_hub=True,
    )
    
    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    )

    # target = torch.unsqueeze(target)

    trainer.train()

    trainer.evaluate()