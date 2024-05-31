import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,mean_squared_error
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB



df=pd.read_csv("perfume_dataset_merge.csv")

df=df[df['department']!='Unisex']
X = df[['base_note', 'middle_note']]
y = df['department']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train['base_note'] + ' ' + X_train['middle_note'])
X_test_vec = vectorizer.transform(X_test['base_note'] + ' ' + X_test['middle_note'])



# Create individual models
naive_bayes = MultinomialNB()
logistic_regression = LogisticRegression(max_iter=200)

# Create an ensemble using a Voting Classifier
ensemble = VotingClassifier(estimators=[('naive_bayes', naive_bayes), ('logistic_regression', logistic_regression)], voting='hard')

# Train the ensemble
ensemble.fit(X_train_vec, y_train)

# Make predictions
predictions = ensemble.predict(X_test_vec)

# Evaluate ensemble performance
accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

print("Ensemble Accuracy:", accuracy)


