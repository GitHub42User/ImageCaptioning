import streamlit as st
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('your_image.png')
imgplot = plt.imshow(img)
plt.show()
st.title('Uber pickups in NYC')
