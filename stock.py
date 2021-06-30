import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import json
import openai

from gpt3 import GPT
from gpt3 import Example

# import config
from sample_config import api_key, train_size_per_class, test_size

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
# print(X_train2)

temp_train = list(zip(X_train2, y_train2))
random.shuffle(temp_train) #Randomize the training set

labels = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
num_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

#CREATE MODEL, AND ADD TRAINING DATA

# Reading the API key from a file
service_key = config.api_key
openai.api_key = service_key

gpt = GPT(engine="davinci", temperature=0.5, max_tokens=3)
# gpt.add_example(Example('''According to Gran , the company has no plans to
# move all production to Russia , although that is
# where the company is growing ''', 'Neutral'))
for t in temp_train:
    gpt.add_example(Example(t[0], num_labels[t[1]]))

# print(gpt.get_all_examples())

#TEST MODEL

accuracy, total = 0, 0
f = open("testing.txt", "at")
for index in range(0, test_size):
    total = total+1
    # print(y_test[index])
    # print(len(X_test[index]))
    prompt = X_test[index]
    output = gpt.submit_request(prompt)
    test_label = output.choices[0].text
    print(test_label, y_test[index])
    f.write(prompt+ ': '+ str(y_test[index])+ 'vs'+ test_label.replace('output: ', '').strip()+'\n')
    try:
        if labels[test_label.replace('output: ', '').strip()]==y_test[index]:
            print('Accurate for', y_test[index])
            accuracy = accuracy+1
    except:
        print('Error')
        pass
f.close()

print(accuracy*100/total)