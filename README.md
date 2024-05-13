# Gender Bias in Natural Language Processing

GitHub link: https://github.com/eleanorlurgio/fyp

This project consists of code to fine-tune 4 models for sentiment analysis and then assess them for gender bias.

## Instructions

1. Install dependencies by running:

pip install -r requirements.txt

2. Run the appropriate python file to train and evaluate each model, for example run:

python .\src\experiment_1\model_1.py 

Summary of each file:

* experiment_1/model_1 = BERT model
* experiment_1/model_2 = RoBERTa model
* experiment_1/model_3 = DistilBERT model
* experiment_1/model_4 = XLNet model
* experiment_2/model_1 = BERT model with augmented dataset (utilising the augment.py file)
* experiment_2/model_2 = RoBERTa model with augmented dataset (utilising the augment.py file)
* experiment_2/model_3 = DistilBERT model with augmented dataset (utilising the augment.py file)
* experiment_2/model_4 = XLNet model with augmented dataset (utilising the augment.py file)


Note - each file's 'main' function looks like this:

```
def main():
    train_model()
    load_model()
```

To only load the model without training from the beginning, comment or remove the train_model() line

3. View results inside of results folder