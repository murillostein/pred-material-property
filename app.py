import pandas as pd
import streamlit as st
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# function to get the dataset
def get_data():
    return pd.read_csv("data_clean.csv")


# function to apply all the models listed
def apply_model(param, predict):

    df = get_data()

    x = df[[' C', ' Si', ' Mn', ' P', ' S', ' Ni', ' Cr', ' Mo',' Cu', 'V', ' Al', ' N', 'Ceq', 'Nb + Ta', ' Temperature (°C)']].values
    y = df[predict].values
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.65, random_state=42)
    
    models = ['Linear Regression', 'Decision Tree Regressor','Random Forest Regressor']

    # df to store the results from each model
    results = pd.DataFrame()

    # loop to iterate over all the models listed
    for i in range(0, len(models)):

        if i == 0:
            regressor = LinearRegression()

        if i == 1:
            regressor = DecisionTreeRegressor()

        if i == 2:
            regressor = RandomForestRegressor()
        
        # fit the model
        regressor.fit(X_train, y_train)

        pred = regressor.predict(X_test)
        mae_i, r2_i =  mean_absolute_error(y_test, pred), r2_score(y_test, pred)

        results = results.append({'Modelo': models[i],'R2': r2_i,'MAE': mae_i, 'reg': regressor}, ignore_index=True)

    return results


# --------MAIN------

st.title("Determine alloy properties")

st.write("This app aims to demonstrate an application of Artificial Intelligence on Metallurgical and Mechanical Engineering industry by providing an efficient and accurate tool to predict alloy properties based on its composition.")
st.write("The process of testing a large number of materials with different compositions in a laboratory is a slow and expesive task. By utlizing this app, you can reduce significatively both problems and still have high accuracy (~95%).")
st.write("Select the component weight percentage on the lateral section and the property you want to predict.")

st.write('*Dataset:* https://www.kaggle.com/rohannemade/mechanical-properties-of-low-alloy-steels')



# -------SIDEBAR----------
st.sidebar.title("Define component percentage")
st.sidebar.markdown("")
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


st.sidebar.markdown("")
st.sidebar.subheader("Value to predict")
In16 = st.sidebar.selectbox('',[" Tensile Strength (MPa)"," Elongation (%)"," Reduction in Area (%)"])

param = [In1,In2,In3,In4,In5,In6,In7,In8,In9,In10,In11,In12,In13,In14,In15]
# talvez colocar um if de impurezas

df = pd.DataFrame(param,[' C', ' Si', ' Mn', ' P', ' S', ' Ni', ' Cr', ' Mo',' Cu', 'V', ' Al', ' N', 'Ceq', 'Nb + Ta', ' Temperature (°C)'])

btn_predict = st.sidebar.button("Predict")


st.sidebar.markdown('') 
st.sidebar.markdown("**Murillo Stein**")
st.sidebar.markdown("https://www.linkedin.com/in/murillo-stein/")

if btn_predict:
    results = apply_model(df, In16)


    st.header("Results")

    st.subheader("Model accuracy")
    st.write(results[['Modelo','MAE','R2']].set_index('Modelo'))


    best_model = results['reg'][results['R2'] == results['R2'].max()].values
    model = best_model[0]


    df = [list(df[0])]
    prediction = model.predict(df)
    st.write('Prediction')
    st.write('The model utilized was ', model, ' and the prediction is' , prediction.item())