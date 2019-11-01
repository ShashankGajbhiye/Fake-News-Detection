# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:06:16 2019

@author: Shashank
"""

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the Dataset
dataset = pd.read_csv('news.csv')
labels = dataset.label
labels.head()

# Exploring the Dataset
dataset.head()
dataset = dataset.rename(columns = {'Unnamed:0' : 'Count'})
dataset.isnull().any()
dataset['label'].value_counts()

# Splitting the Dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset['text'], labels, test_size = 0.2, random_state = 7)

# Initializing the TfIdfVactorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Fitting the Classifier to the Training Set
from sklearn.linear_model import PassiveAggressiveClassifier
classifier = PassiveAggressiveClassifier(max_iter = 50)
classifier.fit(tfidf_train, y_train)

# Predicting the Test Set Results
y_pred = classifier.predict(tfidf_test)

# Calculating the Accuracy
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(acc_score*100, 2)}%')

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels = ['FAKE', 'REAL'])
cm

