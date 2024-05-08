###################################
# Experiment 2 - Model 3
# Model: DistilBERT (distilbert-base-uncased)
# Data: Counterfactually augmented
# Tokenizer: distilbert-base-uncased
# Batch size: 32
# Optimizer: Adam
# Learning rate: 1e-5
# Loss function: CrossEntropyLoss
###################################

import augment
import collections
import math
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import scipy
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
import transformers
from transformers import AutoTokenizer
from transformers import XLNetTokenizer, XLNetModel

SEED = 1234 # set random seed to a fixed value to ensure reproducible results

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # determine whether a CUDA-compatible GPU is available, and use it if so; otherwise, use the CPU

torch.manual_seed(SEED) # apply the fixed random seed to PyTorch to ensure consistent initialization and random operations

torch.backends.cudnn.deterministic = True # ensure that any operations performed by cuDNN are deterministic,helping in reproducing results but may reduce performance

print("PyTorch Version: ", torch.__version__)
print("torchtext Version: ", torchtext.__version__)
print("transformers Version: ", transformers.__version__)
print(f"Using {'GPU' if str(DEVICE) == 'cuda' else 'CPU'}.")

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased') # load distilbert-base-uncased autotokenizer

max_input_length = tokenizer.max_model_input_sizes['distilbert-base-uncased'] # max input size of distilbert-base-uncased

dataset = load_dataset("dair-ai/emotion", "split") # load the split version of the emotion dataset

train_data = augment.augment_data()
valid_data = dataset["validation"]
test_data = dataset["test"]

print("Full train data:", len(train_data))
print("Full val data:", len(valid_data))
print("Full test data:", len(test_data))

tokenizer.convert_ids_to_tokens(tokenizer.encode("hello world"))

def tokenize_and_numericalize(example, tokenizer): # tokenize and turn into ids
    ids = tokenizer(example["text"], truncation=True)["input_ids"]
    return {"ids": ids}

train_data = train_data.map(
    tokenize_and_numericalize, fn_kwargs={"tokenizer": tokenizer}
)
test_data = test_data.map(
    tokenize_and_numericalize, fn_kwargs={"tokenizer": tokenizer}
)

pad_index = tokenizer.pad_token_id

test_size = 0.25

train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]

train_data = train_data.with_format(type="torch", columns=["ids", "label"])
valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
test_data = test_data.with_format(type="torch", columns=["ids", "label"])


def get_collate_fn(pad_index): # collate batches
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

def get_data_loader(dataset, batch_size, pad_index, shuffle=False): # configure data loader
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader


batch_size = 32 # set batch size

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True) # create train data loader
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index) # create validation data loader
test_data_loader = get_data_loader(test_data, batch_size, pad_index) # create test data loader

class Transformer(nn.Module): # define transformer model
    def __init__(self, transformer, output_dim, freeze):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim) # add linear layer
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, ids): # define forward function
        output = self.transformer(ids, output_attentions=True)
        hidden = output.last_hidden_state
        attention = output.attentions[-1]
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(torch.tanh(cls_hidden))
        return prediction

transformer = transformers.AutoModel.from_pretrained("distilbert-base-uncased") # import pretrained distilbert-base-uncased model

output_dim = len(train_data["label"].unique())
freeze = False

model = Transformer(transformer, output_dim, freeze) # create model using pretrained distilbert-base-uncased model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_parameters(model):,} trainable parameters") # display number of trainable parameters

optimizer = optim.Adam(model.parameters(), lr=1e-5) # define optimizer and learning rate
criterion = nn.CrossEntropyLoss() # define loss function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # find GPU else use CPU

model = model.to(device) # push model to GPU
criterion = criterion.to(device) # push loss function to GPU

def Average(list): # util to find average of a list
    if len(list) != 0:
        return sum(list) / len(list)
    else:
        return 0

def predict_sentiment(text, model, tokenizer, device): # use model to predict sentiment of a given text
    ids = tokenizer(text)["input_ids"]
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)

    prediction = model(tensor).squeeze(dim=0) # predictions
    predicted_class = prediction.argmax(dim=-1).item() # highest predicted class

    probability = torch.softmax(prediction, dim=-1) # probability distribution
    predicted_probability = probability[predicted_class].item() # highest probability

    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"] # convert predicted_class to its text label

    if predicted_class == 0:
        predicted_class = labels[0]
    elif predicted_class == 1:
        predicted_class = labels[1]
    elif predicted_class == 2:
        predicted_class = labels[2]
    elif predicted_class == 3:
        predicted_class = labels[3]
    elif predicted_class == 4:
        predicted_class = labels[4]
    elif predicted_class == 5:
        predicted_class = labels[5]

    plt.clf()

    plt.figure(figsize=(4.5, 4)) # create graph to show probability distribution
    plt.bar(labels, probability.cpu().detach().numpy(), color = "darkblue", edgecolor = "black", width = 1)
    plt.title("Probability Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Probability")

    plt.savefig('probability_distribution.png') # save graph

    plt.close()

    return probability.cpu().detach().numpy(), predicted_class, predicted_probability

def plot_distributions(distribution_1, distribution_2): # plot graphs of two probability distributions
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    plt.clf()

    plt.figure(figsize=(4.5, 4))
    plt.bar(labels, distribution_1, color = "darkblue", edgecolor = "black", width = 1)
    plt.title("Probability Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Probability")

    plt.savefig('probability_distribution_1.png')

    plt.clf()

    plt.figure(figsize=(4.5, 4))
    plt.bar(labels, distribution_2, color = "darkblue", edgecolor = "black", width = 1)
    plt.title("Probability Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Probability")

    plt.savefig('probability_distribution_2.png')

    plt.close()

def get_wasserstein(distribution_1, distribution_2): # get wasserstein distance between two probability distributions
    wasserstein = wasserstein_distance(np.arange(6), np.arange(6), distribution_1, distribution_2)
    return wasserstein

def get_jensenshannon(distribution_1, distribution_2): # get jensen shannon distance between two probability distributions
    jensenshannon = distance.jensenshannon(distribution_1, distribution_2)
    return jensenshannon

def get_accuracy(prediction, label): # calculate accuracy metric
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

def get_precision_score(prediction, label): # calculate precision metric
    predicted_classes = prediction.argmax(dim=-1)
    return precision_score(label.cpu(), predicted_classes.cpu(), average="weighted", zero_division=1.0)

def get_recall_score(prediction, label): # calculate recall metric
    predicted_classes = prediction.argmax(dim=-1)
    return recall_score(label.cpu(), predicted_classes.cpu(), average="weighted", zero_division=1.0)

def get_f1_score(prediction, label): # calculate f1 metric
    predicted_classes = prediction.argmax(dim=-1)
    return f1_score(label.cpu(), predicted_classes.cpu(), average="weighted", zero_division=1.0)

def get_confusion_matrix(prediction, label): # generate confusion matrix
    predicted_classes = prediction.argmax(dim=-1)
    return confusion_matrix(label.cpu(), predicted_classes.cpu(), labels=[0, 1, 2, 3, 4, 5])

def get_incorrect_predictions(ids, prediction, label): # get wrong predictions
    predicted_classes = prediction.argmax(dim=-1)
    for input, prediction, label in zip(ids, predicted_classes, label):
        if prediction != label:
            text = tokenizer.decode(input.cpu().detach().numpy())
            text = text.replace('[PAD]', '')
            text = text.replace('[CLS]', '')
            text = text.replace('[SEP]', '')
            text = text.strip()
            print(text, 'has been classified as ', prediction.cpu().detach().numpy(), 'and should be ', label.cpu().detach().numpy()) 

def get_bias(): # evaluate bias of model
    df = pd.read_csv('datasets\Equity-Evaluation-Corpus.csv', usecols=["Sentence", "Template", "Person", "Gender", "Race", "Emotion", "Emotion word"]) # load EEC dataset
    eec = df.to_numpy() # convert dataframe to numpy array

    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"] # all six emotion labels present in the EEC dataset

    male_sentence = [] # stores all sentences with male label
    female_sentence = [] # stores all sentences with female label

    male_probability = [] # stores all probability distributions generated for sentences with male label
    female_probability = [] # stores all probability distributions generated for sentences with female label

    male_predicted_class = [] # stores all predicted classes generated for sentences with male label
    female_predicted_class = [] # stores all predicted classes generated for sentences with female label

    male_predicted_probability = [] # stores all predicted probabilities generated for sentences with male label
    female_predicted_probability = [] # stores all predicted probabilities generated for sentences with female label

    bias_scores = [] # stores the counterfactual bias scores generated between each pair of sentences

    for i in range(0, eec[:,0].size): # loop through each sentence of the dataset
        sentence = eec[i,0]
        gender = eec[i,3]

        probability_distribution, predicted_class, predicted_probability = predict_sentiment(sentence, model, tokenizer, device) # predict sentiment of sentence

        if gender == "male":
            male_sentence.append(sentence)
            male_probability.append(probability_distribution)
            male_predicted_class.append(predicted_class)
            male_predicted_probability.append(predicted_probability)
        elif gender == "female":
            female_sentence.append(sentence)
            female_probability.append(probability_distribution)      
            female_predicted_class.append(predicted_class)
            female_predicted_probability.append(predicted_probability)

    for i in range(0, len(male_sentence)): # for each male sentence, calculate the counterfactual bias score between it and its female counterpart
        bias_scores.append(get_jensenshannon(male_probability[i], female_probability[i]))

    df = pd.DataFrame() # create dataframe to store all predictions and bias results

    df.insert(0, "Sentence (Male)", male_sentence, True)
    df.insert(1, "Sentence (Female)", female_sentence, True)
    df.insert(2, "Predicted Class (Male)", male_predicted_class, True)
    df.insert(3, "Predicted Probability (Male)", male_predicted_probability, True)
    df.insert(4, "Predicted Class (Female)", female_predicted_class, True)
    df.insert(5, "Predicted Probability (Female)", female_predicted_probability, True)
    df.insert(6, "Bias Score", bias_scores, True)

    df.sort_values("Bias Score", axis=0, ascending=False,inplace=True, na_position='first') # sort sentences from most to least biased

    output_dir = Path("results/experiment_2/model_3") # set output directory
    output_dir.mkdir(parents=True, exist_ok=True) # make output directory if it doesn't exist

    df.to_csv(output_dir / "bias_scores.csv", sep=',', index=False, encoding='utf-8') # save dataframe to csv file

    df_count = pd.DataFrame(index=range(6),columns=range(3)) # make dataframe to store the class counts for male and female

    male_class_count = [male_predicted_class.count('sadness'), male_predicted_class.count('joy'), 
        male_predicted_class.count('love'), male_predicted_class.count('anger'), 
        male_predicted_class.count('fear'), male_predicted_class.count('surprise')] # count how many male sentences were predicted for each class

    female_class_count = [female_predicted_class.count('sadness'), female_predicted_class.count('joy'), 
        female_predicted_class.count('love'), female_predicted_class.count('anger'), 
        female_predicted_class.count('fear'), female_predicted_class.count('surprise')] # count how many female sentences were predicted for each class

    df_count.insert(0, "Class", labels, True)
    df_count.insert(1, "Male Predicted Class Count", male_class_count, True)
    df_count.insert(2, "Female Predicted Class Count", female_class_count, True)

    df_count.to_csv(output_dir / "bias_class_counts.csv", sep=',', index=False, encoding='utf-8') # save dataframe to csv file

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
    epoch_precision = []
    epoch_recall = []
    epoch_f1 = []
    matrices = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            get_incorrect_predictions(ids, prediction, label)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            precision = get_precision_score(prediction, label)
            recall = get_recall_score(prediction, label)
            f1_score = get_f1_score(prediction, label)
            confusion_matrix = get_confusion_matrix(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
            epoch_precision.append(precision.item())
            epoch_recall.append(recall.item())
            epoch_f1.append(f1_score.item())
            matrices.append(confusion_matrix)
    return np.mean(epoch_losses), np.mean(epoch_accs), np.mean(precision), np.mean(recall), np.mean(f1_score), np.sum(matrices, axis=0)


def train_model():
    n_epochs = 20
    best_valid_loss = float("inf")

    metrics = collections.defaultdict(list)

    for epoch in range(n_epochs):
        train_loss, train_acc = train(
            train_data_loader, model, criterion, optimizer, device
        )
        valid_loss, valid_acc, precision, recall, f1_score, _ = evaluate(valid_data_loader, model, criterion, device)
        metrics["train_losses"].append(train_loss)
        metrics["train_accs"].append(train_acc)
        metrics["valid_losses"].append(valid_loss)
        metrics["valid_accs"].append(valid_acc)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1_score"].append(f1_score)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "experiment_2_model_3.pt")
        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")
        print(f"precision: {precision}, recall: {recall}, f1_score: {f1_score}")

    df = pd.DataFrame() # create dataframe to save all results

    df.insert(0, "Train Loss", metrics["train_losses"], True)
    df.insert(1, "Train Accuracy", metrics["train_accs"], True)
    df.insert(2, "Valid Loss", metrics["valid_losses"], True)
    df.insert(3, "Valid Accuracy", metrics["valid_accs"], True)
    df.insert(4, "Precision", metrics["precision"], True)
    df.insert(5, "Recall", metrics["recall"], True)
    df.insert(6, "F1 Score", metrics["f1_score"], True)

    output_dir = Path("results/experiment_2/model_3")
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "results.csv", sep=',', index=False, encoding='utf-8') # save dataframe to csv file

    fig = plt.figure(figsize=(10, 6)) # create graph showing accuracy and loss metrics
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics["train_accs"], label="train accuracy")
    ax.plot(metrics["valid_accs"], label="valid accuracy")
    ax.plot(metrics["train_losses"], label="train loss")
    ax.plot(metrics["valid_losses"], label="valid loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy/loss")
    ax.set_xticks(range(n_epochs))
    ax.legend()
    ax.grid()

    plt.savefig(output_dir / 'accuracy_loss.png') # save graph

    plt.clf()

    fig = plt.figure(figsize=(10, 6)) # create graph showing precision, recall and f1_score
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics["precision"], label="precision")
    ax.plot(metrics["recall"], label="recall")
    ax.plot(metrics["f1_score"], label="f1_score")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_xticks(range(n_epochs))
    ax.legend()
    ax.grid()

    plt.savefig(output_dir / 'f1_score.png') # save graph


def load_model():
    model.load_state_dict(torch.load("experiment_2_model_3.pt")) # load previously trained model from memory

    metrics = collections.defaultdict(list)

    test_loss, test_acc, precision, recall, f1_score, confusion_matrix = evaluate(test_data_loader, model, criterion, device) # evaluate model
    metrics["test_losses"].append(test_loss)
    metrics["test_accs"].append(test_acc)
    metrics["precision"].append(precision)
    metrics["recall"].append(recall)
    metrics["f1_score"].append(f1_score)
    print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}") # show test loss and test accuracy
    print(f"precision: {precision:.3f}, recall: {recall:.3f}, f1_score: {f1_score:.3f}") # show precision, recall and f1_score
    print(confusion_matrix)

    get_bias() # evaluate bias of model

    text = "I feel sadness."
    distribution_1, _, _ = predict_sentiment(text, model, tokenizer, device)

    text = "I feel joy."
    distribution_2, _, _ = predict_sentiment(text, model, tokenizer, device)

    plot_distributions(distribution_1, distribution_2)

    get_wasserstein(distribution_1, distribution_2)

    get_jensenshannon(distribution_1, distribution_2)


def main():
    train_model()
    load_model()

main()
    
