import streamlit as st
import pandas as pd

st.set_page_config(page_title="Raw Data", page_icon="📄")

st.subheader('Titanic Dataset (Raw Data)')
url = 'https://drive.google.com/uc?id=1w1XQ8RzDDOLrUpBoE9oqRSPLO6QILEog'
df = pd.read_csv(url)
st.dataframe(df)

