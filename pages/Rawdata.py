import streamlit as st
import pandas as pd

st.set_page_config(page_title="Raw Data", page_icon="📄")

file_id = "1w1XQ8RzDDOLrUpBoE9oqRSPL06QILEog"
url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(url)
st.dataframe(df)
