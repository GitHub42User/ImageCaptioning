import streamlit as st
import pandas as pd
import numpy as np


DATE_COLUMN = 'date/time'
DATA_URL = ('https://img.alicdn.com/imgextra/i3/817462628/O1CN01eLHBGX1VHfUMBA1du_!!817462628.jpg')

def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
