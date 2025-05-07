import streamlit as st
import pandas as pd

st.set_page_config(page_title="Raw Data", page_icon="📄")

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/MyData/titanic_train.csv')
st.dataframe(df)
