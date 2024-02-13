import streamlit as st
from PIL import Image
import pickle
import numpy as np
import pandas as pd
import datetime
import calendar

def load_model():
    with open('junction_model_1.pkl', 'rb') as f1:
        data1 = pickle.load(f1)
    with open('junction_model_2.pkl', 'rb') as f2:
        data2 = pickle.load(f2)
    with open('junction_model_3.pkl', 'rb') as f3:
        data3 = pickle.load(f3)
    with open('junction_model_4.pkl', 'rb') as f4:
        data4 = pickle.load(f4)

    return data1, data2, data3, data4

data1, data2, data3, data4 = load_model()

m1 = data1['model']
m2 = data2['model']
m3 = data3['model']
m4 = data4['model']

e = data1['le_day']

def date_to_day(year, month, date):
    date = datetime.datetime(year, month, date)
    day = calendar.day_name[date.weekday()]
    return day

def show_predict_page():
    
    img = Image.open("https://imgur.com/a/yLEVNoZ")
    st.image(img, caption="**Traffic at Junctions**", width=1200, channels="RGB")

    # Markdown Settings
    st.markdown(
        """
        <style>
        body {
            background-color: #3498db;
            background-image: url('traffic.png');  /* Replace with your image URL */
            background-size: cover;
        }
        
        .stButton:hover {
            color: black !important; 
        }
        .stSelectbox {
            background-color: #e6e6e6; /* Set your desired select box background color */
        }
        .stSubheader {
            color: #ff4500; /* Set your desired subheader text color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Prediction of traffic at Junctions")

    st.markdown("**The Use of IOT sensors to collect the traffic data and the machine learning models for the prediction of the traffic at the different junctions**")
    st.write("""Enter the below details to get the traffic prediction""")

    year = {
        2015, 
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
        2022,
        2023,
        2024
    }

    month = {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12
    }

    date = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
    }

    hour = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    }

    junction = {
        1, 2, 3, 4 
    }

    year = st.selectbox("Year of travelling", year)
    month = st.selectbox("Month of travelling", month)
    date = st.selectbox("Date of travelling", date)
    hour = st.selectbox("Approximate hour of travelling?", hour)
    junction = st.selectbox("Junction Number", junction)

    day = date_to_day(year, month, date)
    day = e[day]

    # Apply a custom style to change the font color of the "Predict Traffic" button
    ok = st.button("Predict Traffic", key="predict_button")

    if ok:
        if junction == 1:
            prediction = m1.predict([[year, month, date, hour, day]])
            prediction = int(np.ceil(prediction[0]))
            st.subheader(f"The traffic on Junction 1 at the given date and time is approximate to be {prediction}")

        elif junction == 2:
            prediction = m2.predict([[year, month, date, hour, day]])
            prediction = int(np.ceil(prediction[0]))
            st.subheader(f"The traffic on Junction 2 at the given date and time is approximate to be {prediction}")

        elif junction == 3:
            prediction = m3.predict([[year, month, date, hour, day]])
            prediction = int(np.ceil(prediction[0]))
            st.subheader(f"The traffic on Junction 3 at the given date and time is approximate to be {prediction}")

        elif junction == 4:
            prediction = m4.predict([[year, month, date, hour, day]])
            prediction = int(np.ceil(prediction[0]))
            st.subheader(f"The traffic on Junction 4 at the given date and time is approximate to be {prediction}")
            
st.set_page_config(
        page_title="Traffic Prediction",
        page_icon="🚗",
        layout="wide",
    )
show_predict_page()
