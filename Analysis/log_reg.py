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

vectorizer = CountVectorizer()

X=vectorizer.fit_transform(df['base_note_filtered'].apply(lambda x: ' '.join(map(str, x)))+' '+df['middle_note_filtered'].apply(lambda x: ' '.join(map(str, x))))
y=np.array(df['department'])

xn_train,xn_test,yn_train,yn_test=train_test_split(X, y, test_size=0.2, random_state=42)
model=LogisticRegression()
model.fit(xn_train,yn_train)
predictions=model.predict(xn_test)
accuracy=accuracy_score(yn_test,predictions)

coefficients = model.coef_
feature_names = vectorizer.get_feature_names_out()

top_features = {}
for class_label, class_coeffs in enumerate(coefficients):
    class_feature_importance = [(feature_names[i], abs(class_coeffs[i])) for i in range(len(feature_names))]
    class_feature_importance.sort(key=lambda x: x[1], reverse=True)
    top_features[class_label] = class_feature_importance[:10]

# Display the top 5 most significant features for each class
for class_label, features in top_features.items():
    print(f"\nTop 5 most significant features for class '{class_label}' ('{model.classes_[class_label]}'):")
    for feature, importance in features:
        print(f"{feature}: {importance}")