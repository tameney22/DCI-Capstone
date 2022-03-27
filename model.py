"""
This script is where the preprocessed data is used to train the SVM model to
perform the classification. Once trained, the model is used to predict a portion
of the data reserved for testing, which is then used to determine the model's accuracy.

REFERENCE: https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34
"""

import numpy as np
import pandas as pd
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


# Open preproccessed csv
df = pd.read_csv("preprocessed.csv", index_col=0)
print(df.head())

print("SPLITTING TRAIN-TEST")
train_x, test_x, train_y, test_y = model_selection.train_test_split(
    df["Text"], df["PublicationTitle"], test_size=0.3)

# Label encode the target variable to transform categorical data of string
# type into numerical values the model can understand

encoder = LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

# Word vectorization
# turning a collection of text documents into numerical feature vectors
# We are using Term Frequency - Inverse Document
tfidf_vect = TfidfVectorizer(max_features=5000)
tfidf_vect.fit(df["Text"])

train_x_tfidf = tfidf_vect.transform(train_x)
test_x_tfidf = tfidf_vect.transform(test_x)

# print(tfidf_vect.vocabulary_)

# Fit the training dataset to the classifier
print("TRAINING THE MODEL")
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(train_x_tfidf, train_y)

print("RUNNING TEST PREDICTIONS")
predictions = SVM.predict(test_x_tfidf)

# Calculate accuracy score
accuracy = accuracy_score(test_y, predictions)
print("Accuracy:", str(accuracy * 100) + "%")
