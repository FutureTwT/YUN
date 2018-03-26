from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np

train = pd.read_csv('data/train_first.csv')
test = pd.read_csv('data/predict_first.csv')
title = train['Discuss'].values
label = train['Discuss'].values
# split train and test
X_train, X_test, y_train, y_test = train_test_split(title, label, test_size=0.1, random_state=42)

vect = TfidfVectorizer(stop_words='english',
                       token_pattern=r'\b\w{2,}\b',
                       min_df=1, max_df=0.1,
                       ngram_range=(1,2))
mnb = MultinomialNB(alpha=2)
svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42)
mnb_pipeline = make_pipeline(vect, mnb)
svm_pipeline = make_pipeline(vect, svm)
mnb_cv = cross_val_score(mnb_pipeline, title, label, scoring='accuracy', cv=10, n_jobs=1)
svm_cv = cross_val_score(svm_pipeline, title, label, scoring='accuracy', cv=10, n_jobs=1)
print('\nMultinomialNB Classifier\'s Accuracy: %0.5f\n' % mnb_cv.mean())

print('\nSVM Classifier\'s Accuracy: %0.5f\n' % svm_cv.mean())



