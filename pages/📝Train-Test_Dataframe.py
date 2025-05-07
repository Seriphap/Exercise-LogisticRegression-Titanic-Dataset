import streamlit as st
import pandas as pd

st.set_page_config(page_title="Train-Test Dataframe", page_icon="📝")

url = 'https://drive.google.com/uc?id=1w1XQ8RzDDOLrUpBoE9oqRSPLO6QILEog'
df = pd.read_csv(url)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# List of columns to keep
columns_to_keep = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']

# Select the columns from the original DataFrame
df_cleaned = df[columns_to_keep]

# Remove rows where Age is less than or equal to 0 or is NaN (blank data)
df_cleaned = df_cleaned[(df_cleaned['Age'] > 0) & (df_cleaned['Age'].notna())]

# Convert 'Age' column to integers
df_cleaned['Age'] = df_cleaned['Age'].astype(int)

# Remove rows where Fare is 0
df_cleaned = df_cleaned[df_cleaned['Fare'] > 0]

# Remove rows where Embarked is blank or NaN
df_cleaned = df_cleaned[df_cleaned['Embarked'].notna()]

# Encode categorical columns 'Sex' and 'Embarked' using LabelEncoder
label_encoder = LabelEncoder()
df_cleaned['Sex'] = label_encoder.fit_transform(df_cleaned['Sex'])
df_cleaned['Embarked'] = label_encoder.fit_transform(df_cleaned['Embarked'])

# Show cleaned DataFrame
st.subheader('Titanic Dataset (Prepared for Training and Testing Model)')
st.dataframe(df_cleaned)
