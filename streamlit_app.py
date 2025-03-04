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
input_data = {'age': age, 'Family_History': Family_History}

# Заполняем one-hot encoding признаков
categorical_features = ['gender', 'edu_lvl', 'marital_status', 'occupation', 'income_lvl', 'live_area', 'Substance_use', 'Suicide_Attempt', 'Social_Support', 'Stress_Factors']
for feature in categorical_features:
    column_name = f'{feature}_{eval(feature)}'  # Генерация правильного имени колонки
    input_data[column_name] = 1

# Преобразование в DataFrame и дополнение отсутствующих столбцов
input_df = pd.DataFrame([input_data])
input_df = pd.get_dummies(input_df)

# Добавляем недостающие колонки и сортируем
for col in X_train.columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[X_train.columns]

# Прогноз
if st.button('Прогноз'):
    prediction = model.predict(input_df)
    st.write(f'Предсказание: {"Положительный диагноз" if prediction[0] == 1 else "Отрицательный диагноз"}')
