import collections
import math
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
import transformers
from transformers import AutoTokenizer

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
print("transformers Version: ", transformers.__version__)
print(f"Using {'GPU' if str(DEVICE) == 'cuda' else 'CPU'}.")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

max_input_length = tokenizer.max_model_input_sizes['roberta-base']
print(max_input_length)

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

def tokenize_and_numericalize(example, tokenizer):
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

# ids = [batch size, seq len]
# hidden = [batch size, seq len, hidden dim]
# attention = [batch size, n heads, seq len, seq len]
# prediction = [batch size, output dim]

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
        output = self.transformer(ids, output_attentions=True)
        hidden = output.last_hidden_state
        attention = output.attentions[-1]
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(torch.tanh(cls_hidden))
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

def Average(list): 
    if len(list) != 0:
        return sum(list) / len(list)
    else:
        return 0

def predict_sentiment(text, model, tokenizer, device):
    ids = tokenizer(text)["input_ids"]
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    # predictions
    prediction = model(tensor).squeeze(dim=0)
    predicted_class = prediction.argmax(dim=-1).item()

    # reorder predictions
    # prediction_ordered = []
    # prediction_ordered.append(prediction[3])
    # prediction_ordered.append(prediction[4])
    # prediction_ordered.append(prediction[0])
    # prediction_ordered.append(prediction[5])
    # prediction_ordered.append(prediction[1])
    # prediction_ordered.append(prediction[2])

    # probabilities
    probability = torch.softmax(prediction, dim=-1)

    # reorder probabilities
    # probability_distribution = []
    # probability_distribution.append(probability[3])
    # probability_distribution.append(probability[4])
    # probability_distribution.append(probability[0])
    # probability_distribution.append(probability[5])
    # probability_distribution.append(probability[1])
    # probability_distribution.append(probability[2])

    predicted_probability = probability[predicted_class].item()

    # convert predicted_class to its text label
    # labels = ["anger","fear","sadness", "surprise", "joy", "love"]
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

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

    plt.figure(figsize=(4.5, 4))
    plt.bar(labels, probability.cpu().detach().numpy(), color = "darkblue", edgecolor = "black", width = 1)
    plt.title("Probability Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Probability")

    plt.savefig('probability_distribution.png')

    return probability.cpu().detach().numpy(), predicted_class, predicted_probability

def plot_distributions(distribution_1, distribution_2):
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


# TODO: get_wasserstein distance between two probability distributions
def get_wasserstein(distribution_1, distribution_2):
    wasserstein = wasserstein_distance(np.arange(6), np.arange(6), distribution_1, distribution_2)
    print(wasserstein)
    return wasserstein

def get_jensenshannon(distribution_1, distribution_2):
    jensenshannon = distance.jensenshannon(distribution_1, distribution_2)
    print(jensenshannon)
    return jensenshannon

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


# TODO: adapt to find average wasserstein distances for each emotion
def get_bias():
    df = pd.read_csv('src\datasets\Equity-Evaluation-Corpus.csv', usecols=["Sentence", "Template","Person","Gender","Race","Emotion","Emotion word"])
    eec = df.to_numpy()

    # the four emotion labels present in the eec dataset
    labels = ["sadness", "joy", "anger", "fear"]

    male_0 = []
    male_1 = []
    male_2 = []
    male_3 = []

    female_0 = []
    female_1 = []
    female_2 = []
    female_3 = []

    for i in range(0, eec[:,0].size):
        sentence = eec[i,0]
        gender = eec[i,3]

        probability_distribution, predicted_class, predicted_probability = predict_sentiment(sentence, model, tokenizer, device)

        if (predicted_class == "sadness") and (gender == "male"):
            male_0.append(predicted_probability)
        elif (predicted_class == "sadness") and (gender == "female"):
            female_0.append(predicted_probability)
        elif (predicted_class == "joy") and (gender == "male"):
            male_1.append(predicted_probability)
        elif (predicted_class == "joy") and (gender == "female"):
            female_1.append(predicted_probability)
        elif (predicted_class == "anger") and (gender == "male"):
            male_2.append(predicted_probability)
        elif (predicted_class == "anger") and (gender == "female"):
            female_2.append(predicted_probability)
        elif (predicted_class == "fear") and (gender == "male"):
            male_3.append(predicted_probability)
        elif (predicted_class == "fear") and (gender == "female"):
            female_3.append(predicted_probability)               

    diff_0 = (Average(male_0) - Average(female_0)) * 100
    diff_1 = (Average(male_1) - Average(female_1)) * 100
    diff_2 = (Average(male_2) - Average(female_2)) * 100
    diff_3 = (Average(male_3) - Average(female_3)) * 100

    # total bias
    bias_score = abs(diff_0) + abs(diff_1) + abs(diff_2) + abs(diff_3)

    # print("Average male sadness: " + str(Average(male_0) * 100))
    # print("Average female sadness: " + str(Average(female_0) * 100))
    # print("Difference: " + str(diff_0))
    # print("Average male joy: " + str(Average(male_1) * 100))
    # print("Average female joy: " + str(Average(female_1) * 100))
    # print("Difference: " + str(diff_1))
    # print("Average male anger: " + str(Average(male_2) * 100))
    # print("Average female anger: " + str(Average(female_2) * 100))
    # print("Difference: " + str(diff_2))
    # print("Average male fear: " + str(Average(male_3) * 100))
    # print("Average female fear: " + str(Average(female_3) * 100))
    # print("Difference: " + str(diff_3))

    return diff_0, diff_1, diff_2, diff_3, bias_score


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
    diff_0, diff_1, diff_2, diff_3, bias_score = get_bias()
    return np.mean(epoch_losses), np.mean(epoch_accs), diff_0, diff_1, diff_2, diff_3, bias_score


def train_model():
    n_epochs = 30
    best_valid_loss = float("inf")

    metrics = collections.defaultdict(list)

    for epoch in range(n_epochs):
        train_loss, train_acc = train(
            train_data_loader, model, criterion, optimizer, device
        )
        valid_loss, valid_acc, diff_0, diff_1, diff_2, diff_3, bias_score = evaluate(valid_data_loader, model, criterion, device)
        metrics["train_losses"].append(train_loss)
        metrics["train_accs"].append(train_acc)
        metrics["valid_losses"].append(valid_loss)
        metrics["valid_accs"].append(valid_acc)
        metrics["diff_0"].append(diff_0)
        metrics["diff_1"].append(diff_1)
        metrics["diff_2"].append(diff_2)
        metrics["diff_3"].append(diff_3)
        metrics["bias_score"].append(bias_score)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "transformer.pt")
        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")
        print(f"diff_0 (sadness): {diff_0:.3f}, diff_1 (joy): {diff_1:.3f}, diff_2 (anger): {diff_2:.3f}, diff_3 (fear): {diff_3:.3f}, bias_score: {bias_score:.3f}")

    # plot graph of accuracy and loss
    fig = plt.figure(figsize=(10, 6))
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

    plt.savefig('train_roberta.png')

    plt.clf()

    # plot graph of bias score
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(metrics["bias_score"])
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("bias score")
    ax2.set_xticks(range(n_epochs))
    # ax2.legend()
    ax2.grid()

    plt.savefig('bias_roberta.png')


def load_model():
    model.load_state_dict(torch.load("transformer.pt"))

    # test_loss, test_acc, diff_0, diff_1, diff_2, diff_3, bias_score = evaluate(test_data_loader, model, criterion, device)

    # print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")

    # example
    # text = "Did I enjoy the movie?"
    # print(str(text) + "\n" + str(predict_sentiment(text, model, tokenizer, device)))

    # text = "He had a nice conversation with his mum"
    # print(str(text) + "\n" + str(predict_sentiment(text, model, tokenizer, device)))

    # TODO WRITE ABOUT THIS
    text = "He feels joy."
    distribution_1, _, _ = predict_sentiment(text, model, tokenizer, device)

    text = "She feels joy."
    distribution_2, _, _ = predict_sentiment(text, model, tokenizer, device)

    plot_distributions(distribution_1, distribution_2)

    get_wasserstein(distribution_1, distribution_2)

    get_jensenshannon(distribution_1, distribution_2)

def main():
    # train_model()
    load_model()

main()
    
