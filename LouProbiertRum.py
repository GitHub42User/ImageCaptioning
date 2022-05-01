import streamlit as st
import pandas as pd
import numpy as np

st.write("""
# Image Captioning Group 13
Hello, we are Leon Lang, Jean Luis Fichtner and Loredana Bratu and we are the creators 
of the app ,,Image captioning“. Our app’s aim is to automatically describe an image with 
one or more natural language sentences. To generate textual description of images we will 
be using Neural Network and Deep Learning Techniques.
""")

DATE_COLUMN = 'date/time'
DATA_URL = ('https://img.alicdn.com/imgextra/i3/817462628/O1CN01eLHBGX1VHfUMBA1du_!!817462628.jpg')

def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
