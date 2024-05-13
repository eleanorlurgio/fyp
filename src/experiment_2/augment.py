##############################################
# Augment dataset with counterfactual examples
##############################################

import collections
import math
import datasets
from datasets import concatenate_datasets
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re
import scipy
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import string
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
import transformers
from transformers import AutoTokenizer
from transformers import XLNetTokenizer, XLNetModel



def augment_data():

    dataset = load_dataset("dair-ai/emotion", "split") # load the split version of the emotion dataset

    train_data = dataset["train"]
    valid_data = dataset["validation"]
    test_data = dataset["test"]

    def alter_data(examples):
        outputs = ''

        text = examples['text'] # get training sample text

        original = text
        text = text.ljust(len(text)+1) # pad text with space at the end
        text = text.rjust(len(text)+1) # pad text with space at the beginning
        text = text.replace(' he ', ' X ').replace(' she ', ' he ').replace(' X ', ' she ')
        text = text.replace(' hes ', ' X ').replace(' shes ', ' hes ').replace(' X ', ' shes ')
        text = text.replace(' him ', ' X ').replace(' his ', ' Y ').replace(' her ', ' his ').replace(' X ', ' her ').replace(' Y ', ' her ')
        text = text.replace(' himself ', ' X ').replace(' herself ', ' himself ').replace(' X ', ' herself ')
        text = text.replace(' man ', ' X ').replace(' woman ', ' man ').replace(' X ', ' woman ')
        text = text.replace(' men ', ' X ').replace(' women ', ' men ').replace(' X ', ' women ')
        text = text.replace(' boy ', ' X ').replace(' girl ', ' boy ').replace(' X ', ' girl ')
        text = text.replace(' boys ', ' X ').replace(' girls ', ' boys ').replace(' X ', ' girls ')
        text = text.replace(' guy ', ' X ').replace(' girl ', ' guy ').replace(' X ', ' girl ')
        text = text.replace(' guys ', ' X ').replace(' girls ', ' guys ').replace(' X ', ' girls ')
        text = text.replace(' gentleman ', ' X ').replace(' lady ', ' gentleman ').replace(' X ', ' lady ')
        text = text.replace(' gentlemen ', ' X ').replace(' ladies ', ' gentlemen ').replace(' X ', ' ladies ')
        text = text.replace(' brother ', ' X ').replace(' sister ', ' brother ').replace(' X ', ' sister ')
        text = text.replace(' brothers ', ' X ').replace(' sisters ', ' brothers ').replace(' X ', ' sisters ')
        text = text.replace(' son ', ' X ').replace(' daughter ', ' son ').replace(' X ', ' daughter ')
        text = text.replace(' sons ', ' X ').replace(' daughters ', ' sons ').replace(' X ', ' daughters ')
        text = text.replace(' husband ', ' X ').replace(' wife ', ' husband ').replace(' X ', ' wife ')
        text = text.replace(' husbands ', ' X ').replace(' wives ', ' husbands ').replace(' X ', ' wives ')
        text = text.replace(' boyfriend ', ' X ').replace(' girlfriend ', ' boyfriend ').replace(' X ', ' girlfriend ')
        text = text.replace(' boyfriends ', ' X ').replace(' girlfriends ', ' boyfriends ').replace(' X ', ' girlfriends ')
        text = text.replace(' bf ', ' X ').replace(' gf ', ' bf ').replace(' X ', ' gf ')
        text = text.replace(' bfs ', ' X ').replace(' gfs ', ' bfs ').replace(' X ', ' gfs ')
        text = text.replace(' father ', ' X ').replace(' mother ', ' father ').replace(' X ', ' mother ')
        text = text.replace(' fathers ', ' X ').replace(' mothers ', ' fathers ').replace(' X ', ' mothers ')
        text = text.replace(' dad ', ' X ').replace(' mom ', ' dad ').replace(' X ', ' mom ')
        text = text.replace(' dads ', ' X ').replace(' moms ', ' dads ').replace(' X ', ' moms ')
        text = text.replace(' uncle ', ' X ').replace(' aunt ', ' uncle ').replace(' X ', ' aunt ')
        text = text.replace(' uncles ', ' X ').replace(' aunts ', ' uncles ').replace(' X ', ' aunts ')
        text = text.replace(' aunty ', ' X ').replace(' uncle ', ' aunty ').replace(' X ', ' uncle ')

        text = text.replace(text[0], "", 1) # remove first character (space) from the sentence
        text = text[:-1] # remove last character (space) from the sentence

        if (original != text):
            outputs += text # add altered text to the new dataset
        else:
            outputs += original # add the original text to the new dataset
        
        return {'text': outputs} # return new altered dataset

    counterfactual_train_data = train_data.map(alter_data)

    total_train_data = concatenate_datasets([train_data, counterfactual_train_data]) # concatenate all original training samples with all new training samples

    train_data_df = pd.DataFrame(total_train_data) # convert dataset object to dataframe

    train_data_df = train_data_df.drop_duplicates() # remove any repeat rows in the dataset

    augmented_train_data = datasets.Dataset.from_pandas(train_data_df) # convert dataframe back to dataset object

    return augmented_train_data