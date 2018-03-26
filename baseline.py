import pandas as pd
import jieba
import numpy as np
import re

train_file = 'data/train_first.csv'
train_data = pd.read_csv(train_file)
# print(train_data.head())
data = train_data.values
texts = []
labels = []
for line in data:
    string = line[1].replace('，','').replace('。','').replace('～','').replace(' ','').replace('！','').\
        replace('<br/>','').replace('；','').replace('）','').replace('（','').replace('.','').\
        replace('“','').replace('”','').replace(',','').replace('【','').replace('】','').replace('~','').\
        replace('\xa0','').replace('《','').replace('》','').strip()
    words = list(jieba.cut(string))
    texts.append(words)
    labels.append(line[2])

from keras.preprocessing.text import Tokenizer

