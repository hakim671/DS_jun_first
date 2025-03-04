import streamlit as st
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_excel('final_prj.xlsx')
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

model = LogisticRegression(random_state=42, C = 0.03, max_iter = 120, penalty = 'l2', solver = 'saga')
model.fit(X_train, y_train)

age = st.number_input(label='Возраст', value=40, step=5)
gender = st.selectbox(label='Пол', options=['male','female'], index=0)
edu_lvl = st.selectbox(label='Образование', options=['High_School','Middle_School','Postgraduate','Primary','University'], index=3)
marital_status = st.selectbox(label='Семейный статус', options=['Divorced','Married','Single','Widowed'], index=2)
occupation = st.selectbox(label='Профессия', options=['Employed','Retired','Student','Unemployed'], index=2)
income_lvl = st.selectbox(label='Уровень дохода', options=['High', 'Low', 'Medium'], index=2)



