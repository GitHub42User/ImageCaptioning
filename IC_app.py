import streamlit as st
import pandas as pd
import numpy as np

st.title('Image Captioning Group 13')

st.text('Hello, we are Leon Lang, Jean Luis Fichtner and Loredana Bratu and we are the')
st.text('creators of the app Image captioning“. Our app’s aim is to automatically')
st.text('describe an image with one or more natural language sentences. To generate')
st.text('textual description of images we will be using Deep Learning Techniques.')


st.write("""
#Image Captioning Group 13'

Hello, we are Leon Lang, Jean Luis Fichtner and Loredana Bratu and we are the creators 
of the app ,,Image captioning“. Our app’s aim is to automatically describe an image with 
one or more natural language sentences. To generate textual description of images we will 
be using Neural Network and Deep Learning Techniques


""")

from datasets import load_dataset

dataset = load_dataset("laion/laion2B-multi")


