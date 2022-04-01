"""
This script is where the preprocessed data is used to train the SVM model to
perform the classification. I am using Stratified K-Fold Cross Validation to 
prevent bias and/or any imbalance that could affect the model's accuracy.

REFERENCE: https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34
"""

import numpy as np
import pandas as pd
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold


# Open preproccessed csv
df = pd.read_csv("preprocessed.csv", index_col=0)
print(df.head())

print("SPLITTING TRAIN-TEST")
x = df["Text"]
y = df["PublicationTitle"]

train_x, test_x, train_y, test_y = model_selection.train_test_split(
    df["Text"], df["PublicationTitle"], test_size=0.3)

# Label encode the target variable to transform categorical data of string
# type into numerical values the model can understand

encoder = LabelEncoder()
# train_y = encoder.fit_transform(train_y)
# test_y = encoder.fit_transform(test_y)

# Word vectorization
# turning a collection of text documents into numerical feature vectors
# We are using Term Frequency - Inverse Document
tfidf_vect = TfidfVectorizer(max_features=5000)
tfidf_vect.fit(df["Text"])

# train_x_tfidf = tfidf_vect.transform(train_x)
# test_x_tfidf = tfidf_vect.transform(test_x)

x_tfidf = tfidf_vect.transform(df["Text"])
y = encoder.fit_transform(y)


# print(tfidf_vect.vocabulary_)

# Fit the training dataset to the classifier
print("TRAINING THE MODEL")
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
accuracies = []

fold = 1
for train_idx, test_idx in skf.split(x, y):
    print("Working on fold", fold)
    x_train_fold, x_test_fold = x_tfidf[train_idx], x_tfidf[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    SVM.fit(x_train_fold, y_train_fold)
    acc = SVM.score(x_test_fold, y_test_fold)
    print("Acc", fold, ":", acc)
    accuracies.append(acc)
    fold += 1


print("ACCURACIES:", accuracies)
print("Max Accuracy:", np.max(accuracies))
print("Min Accuracy:", np.min(accuracies))
print("Mean of Accuracies:", np.mean(accuracies))
print("STD of Accuracies:", np.std(accuracies))

# print("RUNNING TEST PREDICTIONS")
# predictions = SVM.predict(test_x_tfidf)

# # Calculate accuracy score
# accuracy = accuracy_score(test_y, predictions)
# print("Accuracy:", str(accuracy * 100) + "%")
