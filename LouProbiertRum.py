import streamlit as st
import pandas as pd
import numpy as np
import urllib3
import pandas as pd
from PIL import Image
import io
import json


r = http.request('GET', 'http://3.bp.blogspot.com/-6uKj8avN8oc/UsvAhUlpeSI/AAAAAAAACL8/ce31UUzapow/w1200-h630-p-k-no-nu/Peugeot+308+Sedan2.jpg' )
    img_data = r.data
  
st.title('Uber pickups in NYC')
