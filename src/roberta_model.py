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
import evaluate

# defines evaluation metrics
def compute_metrics(eval_pred):
    metric1 = evaluate.load("precision")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("f1")
    metric4 = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = metric1.compute(predictions=predictions, references=labels, average="weighted")["precision"]
    recall = metric2.compute(predictions=predictions, references=labels, average="weighted")["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    accuracy = metric4.compute(predictions=predictions, references=labels)["accuracy"]

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

def get_model():
    # if type == "train":
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
    # elif type == "load":
    #     model = AutoModelForSequenceClassification.from_pretrained('saved_model')

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    labels = ['very negative', 'negative', 'neutral', 'positive', 'very positive']

    return model, tokenizer, labels

def create_model():
    
    dataset = load_dataset("SetFit/sst5", "default")
    print(dataset)

    model, tokenizer, _ = get_model()

    device = 'cuda' if cuda.is_available() else 'cpu'

    model.to(device)

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    test_dataset = dataset["test"]


    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_validation = validation_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    repo_name = "results"
 
    training_args = TrainingArguments(
    output_dir= repo_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    weight_decay=0.01,
    save_total_limit = 2,
    evaluation_strategy = "epoch",
    save_strategy = "no",
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

    predictions = trainer.predict(tokenized_validation)
    preds = np.argmax(predictions.predictions, axis=-1)

    # metric = evaluate.load("glue", "mrpc")
    # metric.compute(predictions=preds, references=predictions.label_ids)
    trainer.train()
    trainer.evaluate()
    trainer.save_model("saved_model")

    return trainer

# def train_model():
#     trainer = create_model("train")

#     trainer.train()



#     trainer.evaluate()

#     trainer.save_model("saved_model")
    

# def load_model():
#     trainer = create_model("load")

#     trainer.evaluate()

def main():
    create_model()