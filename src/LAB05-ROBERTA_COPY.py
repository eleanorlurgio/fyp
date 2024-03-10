import torch
import torchtext
from transformers import AutoTokenizer, AutoModelForSequenceClassification


import collections

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
import transformers

# Set a fixed value for the random seed to ensure reproducible results
SEED = 1234
# Determine whether a CUDA-compatible GPU is available, and use it if so; otherwise, use the CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Apply the fixed random seed to PyTorch to ensure consistent initialization and random operations
torch.manual_seed(SEED)
# Ensure that any operations performed by cuDNN (a GPU-acceleration library used by PyTorch) are deterministic,
# which can help in reproducing results but may reduce performance
torch.backends.cudnn.deterministic = True


print("PyTorch Version: ", torch.__version__)
print("torchtext Version: ", torchtext.__version__)
print(f"Using {'GPU' if str(DEVICE) == 'cuda' else 'CPU'}.")



tokenizer = AutoTokenizer.from_pretrained("roberta-base")

max_input_length = tokenizer.max_model_input_sizes['roberta-base']

from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

# # Define a class TransformerTokenizer that inherits from torch.nn.Module
# class TransformerTokenizer(torch.nn.Module):
#     # The constructor takes a tokenizer object as input
#     def __init__(self, tokenizer):
#         super().__init__()  # Initialize the superclass (torch.nn.Module)
#         self.tokenizer = tokenizer  # Store the tokenizer object for later use
    
#     # Define the forward method, which will be called to tokenize input text
#     def forward(self, input):
#         # If the input is a list (presumably of strings), iterate over the list
#         if isinstance(input, list):
#             tokens = [] 
#             for text in input:  # Iterate over each string in the input list
#                 # Tokenize the current string and append the list of tokens to the tokens list
#                 tokens.append(self.tokenizer.tokenize(text))
#             return tokens  # Return the list of lists of tokens
#         # If the input is a single string
#         elif isinstance(input, str):
#             return self.tokenizer.tokenize(input)
#         raise ValueError(f"Type {type(input)} is not supported.")
        
# # Create a vocabulary object from the tokenizer's vocabulary, setting minimum frequency to 0
# # This includes all tokens in the tokenizer's vocabulary in the vocab object
# tokenizer_vocab = vocab(tokenizer.vocab, min_freq=0)

import torchtext.transforms as T

# text_transform = T.Sequential(
#     TransformerTokenizer(tokenizer),  # Tokenize
#     T.VocabTransform(tokenizer_vocab),  # Conver to vocab IDs
#     T.Truncate(max_input_length - 2),  # Cut to max length to add BOS and EOS token
#     T.AddToken(token=tokenizer_vocab["<s>"], begin=True),  # BOS token
#     T.AddToken(token=tokenizer_vocab["</s>"], begin=False),  # EOS token
#     T.ToTensor(padding_value=tokenizer_vocab["<pad>"]),  # Convert to tensor and pad
# )

from torchtext.datasets import IMDB
from torchtext.data.functional import to_map_style_dataset
from datasets import load_dataset

# using the split version of the dataset
dataset = load_dataset("dair-ai/emotion", "split")

train_data = dataset["train"]
valid_data = dataset["validation"]
test_data = dataset["test"]

# from torch.utils.data import random_split

print("Full train data:", len(train_data))
print("Full val data:", len(valid_data))
print("Full test data:", len(test_data))

tokenizer.convert_ids_to_tokens(tokenizer.encode("hello world"))

def tokenize_and_numericalize_example(example, tokenizer):
    ids = tokenizer(example["text"], truncation=True)["input_ids"]
    return {"ids": ids}

train_data = train_data.map(
    tokenize_and_numericalize_example, fn_kwargs={"tokenizer": tokenizer}
)
test_data = test_data.map(
    tokenize_and_numericalize_example, fn_kwargs={"tokenizer": tokenizer}
)

print(train_data[0])

pad_index = tokenizer.pad_token_id

test_size = 0.25

train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]

train_data = train_data.with_format(type="torch", columns=["ids", "label"])
valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
test_data = test_data.with_format(type="torch", columns=["ids", "label"])


def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "label": batch_label}
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader


batch_size = 32

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)


class Transformer(nn.Module):
    def __init__(self, transformer, output_dim, freeze):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, ids):
        # ids = [batch size, seq len]
        output = self.transformer(ids, output_attentions=True)
        hidden = output.last_hidden_state
        # hidden = [batch size, seq len, hidden dim]
        attention = output.attentions[-1]
        # attention = [batch size, n heads, seq len, seq len]
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(torch.tanh(cls_hidden))
        # prediction = [batch size, output dim]
        return prediction


transformer = transformers.AutoModel.from_pretrained("roberta-base")

output_dim = len(train_data["label"].unique())
freeze = False

model = Transformer(transformer, output_dim, freeze)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")


optimizer = optim.Adam(model.parameters(), lr=1e-5)



criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
criterion = criterion.to(device)

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


n_epochs = 10
best_valid_loss = float("inf")

metrics = collections.defaultdict(list)

# for epoch in range(n_epochs):
#     train_loss, train_acc = train(
#         train_data_loader, model, criterion, optimizer, device
#     )
#     valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, device)
#     metrics["train_losses"].append(train_loss)
#     metrics["train_accs"].append(train_acc)
#     metrics["valid_losses"].append(valid_loss)
#     metrics["valid_accs"].append(valid_acc)
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), "transformer.pt")
#     print(f"epoch: {epoch}")
#     print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
#     print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")





model.load_state_dict(torch.load("transformer.pt"))

test_loss, test_acc = evaluate(test_data_loader, model, criterion, device)

print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")



def predict_sentiment(text, model, tokenizer, device):
    ids = tokenizer(text)["input_ids"]
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability



text = "This film is terrible!"

print(predict_sentiment(text, model, tokenizer, device))

text = "This film is great!"

print(predict_sentiment(text, model, tokenizer, device))

text = "This film is not terrible, it's great!"

print(predict_sentiment(text, model, tokenizer, device))