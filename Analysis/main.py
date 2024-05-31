import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,mean_squared_error
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import itertools
from joblib import dump,load



df=pd.read_csv("perfume_dataset_merge.csv")
df=df[df['department']!='Kids Unisex']

#classifying between male,female and unisex

label_encoder = LabelEncoder()
label_encoder1=LabelEncoder()

df['base_note_filtered'] = df['base_note'].apply(lambda x:re.split(r',| ,|, | and | And ',x))
df['middle_note_filtered'] = df['middle_note'].apply(lambda x:re.split(r',| ,|, | and | And ',x))
df['department_encoded']=label_encoder1.fit_transform(df['department'].apply(lambda x:x.lower()))

df['base_note_filtered']=df['base_note_filtered'].apply(lambda x:[y.replace(" ","") for y in x])
df['middle_note_filtered']=df['middle_note_filtered'].apply(lambda x:[y.replace(" ","") for y in x])

base_set=",".join([y for x in df['base_note_filtered'] for y in x])
mid_set=",".join([y for x in df['middle_note_filtered'] for y in x])
main_notes=(base_set+mid_set).split(',')
unique_set=set(main_notes)
unique = {item for item in unique_set if item != ''}

label_encoder.fit_transform(main_notes)
df['base_note_encoded']=df['base_note_filtered'].apply(lambda x:label_encoder.transform(x))
df['middle_note_encoded']=df['middle_note_filtered'].apply(lambda x:label_encoder.transform(x))


X = df[['base_note_encoded','middle_note_encoded']]
y = df['department_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xn=df[['base_note','middle_note']]
yn=df['department_encoded']


xn_train,xn_test,yn_train,yn_test=train_test_split(xn, yn, test_size=0.2, random_state=42)
'''
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(xn_train['base_note'] + ' ' + xn_train['middle_note'])
X_test_vec = vectorizer.transform(xn_test['base_note'] + ' ' + xn_test['middle_note'])


# Create and train Multinomial Naive Bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_vec, yn_train)

# Make predictions
predictions_nb = naive_bayes.predict(X_test_vec)

# Evaluate classifier performance
accuracy = accuracy_score(yn_test, predictions_nb)
'''


vectorizer = CountVectorizer()

X=vectorizer.fit_transform(df['base_note_filtered'].apply(lambda x: ' '.join(map(str, x)))+' '+df['middle_note_filtered'].apply(lambda x: ' '.join(map(str, x))))
y=np.array(df['department'])

xn_train,xn_test,yn_train,yn_test=train_test_split(X, y, test_size=0.2, random_state=42)

# model=MultinomialNB()
# model.fit(xn_train,yn_train)

#saved model loading
model = load('multinomial_model.joblib')

predictions=model.predict(xn_test)
accuracy=accuracy_score(yn_test,predictions)
#print(accuracy)
feature_log_probs = model.feature_log_prob_
feature_names = vectorizer.get_feature_names_out()

#save
#dump(model, 'multinomial_model.joblib')


# Use the loaded model for predictions

# Identify and display the top 3 most significant features for each class
top_features = {}
for class_label, class_log_probs in enumerate(feature_log_probs):
    class_feature_importance = [(feature_names[i], class_log_probs[i]) for i in range(len(feature_names))]
    class_feature_importance.sort(key=lambda x: x[1], reverse=True)
    top_features[class_label] = class_feature_importance[:3]

# Display the top 3 most significant features for each class with actual target names
for class_label, features in top_features.items():
    target_name = model.classes_[class_label]
    print(f"\nTop 3 most significant features for class '{class_label}' ('{target_name}'):")
    for feature, log_prob in features:
        print(f"{feature}: {log_prob}")






