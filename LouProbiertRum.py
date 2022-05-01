import streamlit as st
import pandas as pd
import numpy as np
import urllib3
import pandas as pd
from PIL import Image
import io
import json


routes = {}
df = pd.read_csv('index_file.csv')


# iterate over the rows and download the images
for index, row in df.iterrows():
    # get the image from the url
    http = urllib3.PoolManager()
    r = http.request('GET', row['url'])
    img_data = r.data
    
    # save data to image file and row['TEXT'] to json file
    filename = "filename"
    image = Image.open(io.BytesIO(img_data))
    image.save(filename + '.png')
    text = row['TEXT']
    
    # save to routes for later use
    routes[filename] = text

json.dump(routes, open("routes.json", "w"))
