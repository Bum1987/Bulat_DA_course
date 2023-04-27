import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

df = pd.read_csv(r'C:\Users\pride\Downloads\ML_LEC3\housing.csv')

@st.cache(suppress_st_warning=True)
def train_model(train_size):
    X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1),
                                                        df['MEDV'],
                                                        test_size=1-train_size,
                                                        random_state=2100)
    st.write('Разделили данные и передали в обучение')
    regr_model = XGBRegressor()
    regr_model.fit(X_train, y_train)
    pred = regr_model.predict(X_test)
    st.write('Обучили модель, MAE = ' + str(mean_absolute_error(y_test, pred)))

train_size = st.slider("Выберите размер обучающей выборки", 0.1, 0.9, 0.8, 0.05, key='train_size')
if st.button('Обучить модель'):
    train_model(train_size)

if st.button('Отобразить первые пять строк'):
    st.write(df.head())