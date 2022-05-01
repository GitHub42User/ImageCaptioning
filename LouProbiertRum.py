import streamlit as st
import pandas as pd
import numpy as np
st.title('Image Captioning Group 13')

with st.expander("The Team"):
  st.write("Hello, we are Leon Lang, Jean Luis Fichtner and Loredana Bratu and we are the creators of the app")
with st.expander("The Mission"):
  st.write("Our app’s aim is to automatically describe an image with one or more natural language sentences. To generate textual description of images we will be using Neural Network and Deep Learning Techniques.")
with st.expander("The Dataset"):
  st.write("Here you can see some examples from our Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg")

