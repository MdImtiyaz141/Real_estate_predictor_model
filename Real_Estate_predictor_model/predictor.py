import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
#from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
#from sklearn.neural_network import MLPRegressor
#import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.decomposition import PCA

st.set_page_config(page_title='Viz')

# Specify the correct path to df (1).pkl
import streamlit as st
import pickle
#C:\Users\IMTIYAZ\PycharmProjects\Real_estate_project\pages\df.pkl
file_path_1 = "C:/Users/IMTIYAZ/PycharmProjects/Real_estate_project/pages/df.pkl"
file_path_2 = "C:/Users/IMTIYAZ/PycharmProjects/Real_estate_project/pages/pipeline.pkl"

# Attempt to open the first pickle file
try:
    with open(file_path_1, 'rb') as file:
        df = pickle.load(file)
    # Proceed with further processing
    # ...
except FileNotFoundError:
    st.error("Error: File 'df (1).pkl' not found. Please make sure the file exists.")

# Attempt to open the second pickle file
try:
    with open(file_path_2, 'rb') as file:
        pipe = pickle.load(file)
    # Proceed with further processing
    # ...
except FileNotFoundError:
    st.error("Error: File 'pipeline (1).pkl' not found. Please make sure the file exists.")


data = {
    "Salon": 1, "Swimming Pool": 10, "Gymnasium": 9, "Club House": 8, "24x7 Security": 7,
    "Lift(s)": 6, "Power Backup": 5, "InterCom": 4, "CCTV Camera Security": 4,
    "Children's Play Area": 3, "Rain Water Harvesting": 3, "Car Parking": 3,
    "Landscape Garden": 3, "Jogging Track": 3, "Multipurpose Court": 3,
    "Senior Citizen Sitout": 3, "Multipurpose Hall": 2, "School": 2, "ATM": 2,
    "Wi-Fi Connectivity": 2, "Health Facilities": 2, "Amphitheatre": 2,
    "Shopping Centre": 2, "Party Lawn": 2, "Library": 2, "Medical Centre": 2,
    "Sports Facility": 2, "Indoor Games": 2, "Sewage Treatment Plant": 1,
    "Water Softener Plant": 1, "Solar Water Heating": 1, "Fire Fighting Systems": 1,
    "Gated Community": 1, "Vastu Compliant": 1, "Solar Lighting": 1,
    "Earthquake Resistant": 1, "DG Availability": 1, "24/7 Water Supply": 1,
    "24/7 Power Backup": 1, "Waste Management": 1, "Security Cabin": 1,
    "Visitors Parking": 1, "Video Door Security": 1, "Toddler Pool": 1,
    "Theatre": 1, "Table Tennis": 1, "Sun Deck": 1, "Steam Room": 1,
    "Squash Court": 1, "Spa": 1, "Skating Rink": 1, "Restaurant": 1,
    "Reflexology Park": 1, "Reading Lounge": 1, "Piped Gas": 1, "Pergola": 1,
    "Open Space": 1, "Natural Pond": 1, "Mini Theatre": 1, "Milk Booth": 1,
    "Manicured Garden": 1, "Lounge": 1, "Lawn Tennis Court": 1, "Laundry": 1,
    "Jacuzzi": 1, "Internal Street Lights": 1, "Infinity Pool": 1,
    "High Speed Elevators": 1, "Grocery Shop": 1, "Golf Course": 1,
    "Gazebo": 1, "Fountain": 1, "Football": 1, "Foosball": 1, "Food Court": 1,
    "Flower Garden": 1, "Entrance Lobby": 1, "DTH Television": 1,
    "Cricket Pitch": 1, "Creche/Day care": 1, "Conference room": 1,
    "Concierge Service": 1, "Community Hall": 1, "Clinic": 1, "Chess": 1,
    "Changing Area": 1, "Card Room": 1, "Car wash area": 1, "Cafeteria": 1,
    "Business Lounge": 1, "Bus Shelter": 1, "Bowling Alley": 1, "Billiards": 1,
    "Beach Volley Ball Court": 1, "Basketball Court": 1, "Barbecue": 1,
    "Bar/Chill-Out Lounge": 1, "Banquet Hall": 1, "Badminton Court": 1,
    "Air Hockey": 1, "Aerobics Centre": 1, "Volley Ball Court": 5,
    "Waiting Lounge": 3, "Theater Home": 5, "Yoga/Meditation Area": 6,
    "Sauna": 8, "Terrace Garden": 5, "Property Staff": 5, "RO System": 6,
    "Power Back up Lift": 5, "Pool Table": 4, "Garbage Disposal": 3,
    "Cigar Lounge": 8, "Paved Compound": 5
}
df2 = pd.DataFrame(list(data.items()), columns=['Amenity', 'Count'])

#st.dataframe(df)
st.text('This is the predictor model for various apartments in the city of hyderabad')
st.text('Please note that real values may vary since the real estate institution is dynamic')
st.text('We have considered the data for around 400 various Apartments in the city of Hyderabad')

st.text(' ')
st.text('Facilities may include')
st.dataframe(df2)

st.header('Enter the inputs')

facilities = float(st.selectbox('Please the average facilities score basics to luxurious',sorted(list(df['facilities'].unique()))))

advantages = float(st.selectbox('Advantages: average distances at which school, hospital availability in the vicinity',sorted(list(np.round(df['advantages']).unique()))))

BHK = st.selectbox('please select the BHK',sorted(list(df['BHK'].unique())))
SBA = float(st.selectbox('Please input the Super Built-up Area',sorted(list(round(df['SBA']).unique()))))
floors = float(st.selectbox('please input the number of floors',sorted(list(df['floors'].unique()))))
units = float(st.selectbox('please select the unites',sorted(list(df['units'].unique()))))
tpa = float(st.selectbox('please select the total project area in acres',sorted(list(round(df['tpa']).unique()))))
towers = float(st.selectbox('please select the towers',sorted(list(df['towers'].unique()))))
address = st.selectbox('please select the adress',list(df['address'].str.lower().unique()))
if st.button('predict'):
    data = [[facilities, advantages, BHK, SBA, floors, units, tpa, towers, address]]
    columns = ['facilities', 'advantages', 'BHK', 'SBA', 'floors', 'units', 'tpa', 'towers','address']
    test_df = pd.DataFrame(data,columns=columns)
    st.dataframe(test_df)
    base_price = np.expm1(pipe.predict((test_df)))[0]
    lower_limit = base_price - 0.11
    upper_limit = base_price + 0.11
    st.text("Price can range in between: {} Cr - {} Cr".format(round(lower_limit,2),round(upper_limit,2)))


