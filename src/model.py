import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

def get_model():
    # load model and tokenizer
    roberta = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    labels = ['Negative', 'Neutral', 'Positive']

    return model, tokenizer, labels