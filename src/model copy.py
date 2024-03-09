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
   load_accuracy = evaluate.load("accuracy", trust_remote_code=True)
   load_f1 = evaluate.load("f1")
  
   logits, labels = eval_pred

   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
   print("\n" + "Accuracy:" + str(accuracy) + ", F1:" + str(f1))
   return {"accuracy": accuracy, "f1": f1}

def get_model(type):
    if type == "train":
        model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
    elif type == "load":
        model = AutoModelForSequenceClassification.from_pretrained('saved_model')

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    labels = ['Negative', 'Slightly Negative', 'Neutral', 'Slightly Positive', 'Positive']

    return model, tokenizer, labels

def create_model(type):
    
    dataset = load_dataset("SetFit/sst5", "default")

    device = 'cuda' if cuda.is_available() else 'cpu'

    model, tokenizer, _ = get_model(type)

    model.to(device)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]


    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    repo_name = "results"
 
    training_args = TrainingArguments(
    output_dir= repo_name,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit = 2,
    save_strategy = "no",
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

    return trainer

def train_model():
    trainer = create_model("train")

    trainer.train()

    trainer.evaluate()

    trainer.save_model("saved_model")
    

def load_model():
    trainer = create_model("load")

    trainer.evaluate()

def main():
    load_model()