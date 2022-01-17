import pandas as pd
import streamlit as st
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def get_data():
    return pd.read_csv("data.csv")

def apply_model(param, predict):

    df = get_data()

    x = df.drop(columns=[" Tensile Strength (MPa)"," Elongation (%)"," Reduction in Area (%)"]).values
    y = df[predict].values


    
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, shuffle = True, random_state=2)
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    

    pred = regressor.predict(param)

    return pred







st.title("Determinar composição")



# ----SIDEBAR-----
st.sidebar.title("Definir parâmetros")

In1 =  st.sidebar.number_input("Carbon C", min_value=0.0,max_value=1.0,step=0.05)
In2 =  st.sidebar.number_input("Silicon Si", min_value=0.0,max_value=1.0,step=0.05)
In3 =  st.sidebar.number_input("Manganeses N", min_value=0.0,max_value=1.0,step=0.05)
In4 =  st.sidebar.number_input("Phosphore P", min_value=0.0,max_value=1.0,step=0.05)
In5 =  st.sidebar.number_input("Sulfur S", min_value=0.0,max_value=1.0,step=0.05)
In6 =  st.sidebar.number_input("Nickel Ni", min_value=0.0,max_value=1.0,step=0.05)
In7 =  st.sidebar.number_input("Cromium Cr", min_value=0.0,max_value=1.0,step=0.05)
In8 =  st.sidebar.number_input("Molybdenum Mo", min_value=0.0,max_value=1.0,step=0.05)
In9 =  st.sidebar.number_input("Copper Cu", min_value=0.0,max_value=1.0,step=0.05)
In10 = st.sidebar.number_input("Vanadium V", min_value=0.0,max_value=1.0,step=0.05)
In11 = st.sidebar.number_input("Alluminium Al", min_value=0.0,max_value=1.0,step=0.05)
In12 = st.sidebar.number_input("Nytrogen N", min_value=0.0,max_value=1.0,step=0.05)
In13 = st.sidebar.number_input("Carbon equivalent Ceq", min_value=0.0,max_value=1.0,step=0.05)
In14 = st.sidebar.number_input("Nb + Ta", min_value=0.0,max_value=1.0,step=0.05)
In15 = st.sidebar.number_input("Temperature (°C)", min_value=0.0,max_value=750.0,step=20.0)
In16 = st.sidebar.selectbox("Predict:",[" Tensile Strength (MPa)"," Elongation (%)"," Reduction in Area (%)"])

param = [In1,In2,In3,In4,In5,In6,In7,In8,In9,In10,In11,In12,In13,In14,In15]
# talvez colocar um if de impurezas

btn_predict = st.sidebar.button("Prever")

if btn_predict:
    pred = apply_model(param, In16)
    st.write(pred)