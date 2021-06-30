import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import json
import openai

from gpt3 import GPT
from gpt3 import Example

# import config
from sample_config import api_key, train_size_per_class

from datasets import load_dataset

#DATASET GENERATION
dataset = load_dataset("financial_phrasebank", 'sentences_allagree', split='train')
dataset.set_format(type='pandas')

X, y = dataset['sentence'].to_numpy(), dataset['label'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
# print(np.bincount(y_train), np.bincount(y_test))
# print(type(y_test), type(X_test))
# X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

##TRAINING DATA
X_train2, y_train2 = ["" for x in range(train_size_per_class*3)], np.empty((train_size_per_class*3), dtype=int)
index = 0
for i in range(0, 3):
    count = 0
    while count<train_size_per_class:
        if y_train[index]==i:
            X_train2[count+(i*train_size_per_class)], y_train2[count+(i*train_size_per_class)] = X_train[index], y_train[index]
            count = count+1
        index = index+1
X_train2 = np.array(X_train2)
print(X_train2.shape)

temp_train = list(zip(X_train2, y_train2))
random.shuffle(temp_train) #Randomize the training set

labels = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
num_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}