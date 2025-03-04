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

# Ввод данных от пользователя
age = st.number_input(label='Возраст', value=40, step=5)
gender = st.selectbox(label='Пол', options=['male', 'female'], index=0)
edu_lvl = st.selectbox(label='Образование', options=['High_School', 'Middle_School', 'Postgraduate', 'Primary', 'University'], index=3)
marital_status = st.selectbox(label='Семейный статус', options=['Divorced', 'Married', 'Single', 'Widowed'], index=2)
occupation = st.selectbox(label='Профессия', options=['Employed', 'Retired', 'Student', 'Unemployed'], index=2)
income_lvl = st.selectbox(label='Уровень дохода', options=['High', 'Low', 'Medium'], index=2)
live_area = st.selectbox(label='Место проживания', options=['city', 'village'], index=0)
Family_History = st.selectbox(label='Есть в роду шизофреник?', options=[0, 1], index=0)
Substance_use = st.selectbox(label='Употребляешь вредные вещества?', options=['No', 'Yes'], index=0)
Suicide_Attempt = st.selectbox(label='Попытки самоубийства?', options=['No', 'Yes'], index=0)
Social_Support = st.selectbox(label='Поддержка окружающих', options=['High', 'Low', 'Medium'], index=2)
Stress_Factors = st.selectbox(label='Уровень стресса', options=['High', 'Low', 'Medium'], index=2)

# Создание входных данных в формате словаря
input_data = {
    'age': age,
    'Family_History': Family_History,
    f'gender_{gender}': 1,
    f'edu_lvl_{edu_lvl}': 1,
    f'marital_status_{marital_status}': 1,
    f'occupation_{occupation}': 1,
    f'income_lvl_{income_lvl}': 1,
    f'live_area_{live_area}': 1,
    f'Substance_use_{Substance_use}': 1,
    f'Suicide_Attempt_{Suicide_Attempt}': 1,
    f'Social_Support_{Social_Support}': 1,
    f'Stress_Factors_{Stress_Factors}': 1
}

# Преобразование в DataFrame и дополнение отсутствующих столбцов
input_df = pd.DataFrame([input_data])
input_df = pd.get_dummies(input_df)

# Добавление отсутствующих столбцов (заполняем 0, если их нет в input_df)
for col in X_train.columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Удаление лишних столбцов (если есть)
input_df = input_df[X_train.columns]

if st.button("Прогноз"):
  model.predict(df_inp)
