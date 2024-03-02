import numpy as np 
import pandas as pd
import re
import string
from tqdm.notebook import tqdm
import plotly.express as px
import plotly.graph_objects as go

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

dataset = pd.read_csv('../input/nlp-getting-started/train.csv')

dataset_copy = dataset.copy()

dataset = dataset_copy.sample(frac=0.80, random_state=0)
val_dataset = dataset_copy.drop(dataset.index)

max_len = None # Max lenght of the text for input
batch_size = 32
epochs = 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset creator for Pytorch
class DatasetCreator(Dataset):
    def __init__(self, processed_data, train):
        self.data = processed_data
        self.train = train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        line = self.data.iloc[index]
        if self.train:
            return {'text': line['clean_tweet'], 'label': line['target']}
        else:
            return {'text': line['clean_tweet'], 'label': 0}

# Class to tokenize and process the text for input to the dataloader    
class GPT2_collator(object):
    def __init__(self, tokenizer, max_seq_len=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        return
    
    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        labels = [int(sequence['label']) for sequence in sequences]
        inputs = self.tokenizer(text=texts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=self.max_seq_len)
        inputs.update({'labels': torch.tensor(labels)})       
        return inputs

# Function to remove hashtags....etc. Intended to improve performance   
def pre_process(dataset):
    # Remove hashtag
    dataset['clean_tweet'] = dataset['text']#.apply(lambda x: x.replace('#', ' ')) 
    # Remove websites
    #dataset['clean_tweet'] = dataset['clean_tweet'].apply(lambda x: re.sub(r"http\S+", " ", x))
    # Remove emails
    #dataset['clean_tweet'] = dataset['clean_tweet'].apply(lambda x: re.sub(r"\S*@\S*\s?", " ", x))
    # Remove punctuation
    #dataset['clean_tweet'] = dataset['clean_tweet'].apply(lambda x: re.sub(r"[^a-zA-Z]", " ", x))
    # Remove 'RT'
    #dataset['clean_tweet'] = dataset['clean_tweet'].apply(lambda x: re.sub(r"RT\s+", " ", x))
    return dataset

# Function for training
def train(dataloader, optimizer, scheduler, device):
    global model
    model.train()
    predictions_labels = []
    true_labels = []
    total_loss = 0
    
    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss, logits = outputs[:2]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    avg_epoch_loss = total_loss / len(dataloader)
    return predictions_labels, true_labels, avg_epoch_loss

# Function for validation 
def validate(dataloader, device):
    global model
    model.eval()
    predictions_labels = []
    true_labels = []
    total_loss = 0
    
    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            total_loss += loss.item()
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    avg_epoch_loss = total_loss / len(dataloader)
    return predictions_labels, true_labels, avg_epoch_loss

def predict(dataloader, device):
    global model
    model.eval()
    predictions_labels = []
    
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            _, logits = outputs[:2]
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    return predictions_labels

print('Loading gpt-2 model')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path='gpt2', num_labels=2)

print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path='gpt2', config=model_config)
model.resize_token_embeddings(len(tokenizer)) 
model.config.pad_token_id = model.config.eos_token_id
model.to(device)

gpt2_collator = GPT2_collator(tokenizer=tokenizer, max_seq_len=max_len)

# Prepare training data
processed_data = pre_process(dataset)
train_data = DatasetCreator(processed_data, train=True)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=gpt2_collator)

# Prepare validation data
val_processed = pre_process(val_dataset)
val_data = DatasetCreator(val_processed, train=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=gpt2_collator)

optimizer = AdamW(model.parameters(), lr = 5e-5, eps = 1e-8, weight_decay=0.01)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
loss = []
accuracy = []
val_loss_list = []
val_accuracy_list = []

for epoch in tqdm(range(epochs)):
    train_labels, true_labels, train_loss = train(train_dataloader, optimizer, scheduler, device)    
    train_acc = accuracy_score(true_labels, train_labels) 
    print('epoch: %.2f train accuracy %.2f' % (epoch, train_acc))
    loss.append(train_loss)
    accuracy.append(train_acc)

    val_labels, val_true_labels, val_loss = validate(val_dataloader, device)
    val_acc= accuracy_score(val_true_labels, val_labels)
    print('epoch: %.2f validation accuracy %.2f' % (epoch, val_acc))
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_acc)

    # Print train and validation loss
fig_loss = go.Figure()

fig_loss.add_trace(go.Scatter(x=[*range(0,len(loss),1)], y=loss,
                              mode='lines',
                              name='train_loss'))
fig_loss.add_trace(go.Scatter(x=[*range(0,len(loss),1)], y=val_loss_list,
                              mode='lines',
                              name='validation loss'))

# Print train and validation accuracy
fig_acc = go.Figure()

fig_acc.add_trace(go.Scatter(x=[*range(0,len(accuracy),1)], y=accuracy,
                              mode='lines',
                              name='train accuracy'))
fig_acc.add_trace(go.Scatter(x=[*range(0,len(accuracy),1)], y=val_accuracy_list,
                              mode='lines',
                              name='validation accuracy'))

fig_loss.show()
fig_acc.show()

test = pd.read_csv('../input/nlp-getting-started/test.csv')
test.reset_index()
test_dataset = pre_process(test)
test_dataset = DatasetCreator(test_dataset, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_collator)

submission = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
submission['target'] = pd.Series(predict(test_dataloader, device))
submission.to_csv('submission.csv', index=False)