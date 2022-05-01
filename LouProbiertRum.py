import streamlit as st
import pandas as pd
import numpy as np
st.title('Image Captioning Group 13')

with st.expander("The Team"):
  st.write("Hello, we are Leon Lang, Jean Luis Fichtner and Loredana Bratu and we are the creators of the app")
with st.expander("The Mission"):
  st.write("Our app’s aim is to automatically describe an image with one or more natural language sentences. To generate textual description of images we will be using Neural Network and Deep Learning Techniques.")
with st.expander("The Dataset"):
  st.write("Here you can swipe through some examples from our Dataset")



st.image('https://img.alicdn.com/imgextra/i3/817462628/O1CN01eLHBGX1VHfUMBA1du_!!817462628.jpg')
