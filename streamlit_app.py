import streamlit as st
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<p class="title">Прогноз шизофрении</p>', unsafe_allow_html=True)


st.markdown("### Введите свои данные и получите прогноз.")

@st.cache_data
def load_data():
    df = pd.read_excel('final_prj.xlsx')
    return df

df = load_data()

X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

model = LogisticRegression(random_state=42, C = 0.03, max_iter = 120, penalty = 'l2', solver = 'saga')
model.fit(X_train, y_train)

# Ввод данных от пользователя
with st.sidebar:
    st.header("Введите данные пациента")
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

input_df = pd.DataFrame({
    'age': [age],
    'Family_History': [Family_History],
    'gender_female': [0],
    'gender_male': [0],
    'edu_lvl_High_School': [0],
    'edu_lvl_Middle_School': [0],
    'edu_lvl_Postgraduate': [0],
    'edu_lvl_Primary': [0],
    'edu_lvl_University': [0],
    'marital_status_Divorced': [0],
    'marital_status_Married': [0],
    'marital_status_Single': [0],
    'marital_status_Widowed': [0],
    'occupation_Employed': [0],
    'occupation_Retired': [0],
    'occupation_Student': [0],
    'occupation_Unemployed': [0],
    'income_lvl_High': [0],
    'income_lvl_Low': [0],
    'income_lvl_Medium': [0],
    'live_area_city': [0],
    'live_area_village': [0],
    'Substance_use_No': [0],
    'Substance_use_Yes': [0],
    'Suicide_Attempt_No': [0],
    'Suicide_Attempt_Yes': [0],
    'Social_Support_High': [0],
    'Social_Support_Low': [0],
    'Social_Support_Medium': [0],
    'Stress_Factors_High': [0],
    'Stress_Factors_Low': [0],
    'Stress_Factors_Medium': [0]
})


# Gender
if gender == 'male':
    input_df['gender_male'] = 1
else:
    input_df['gender_female'] = 1

# Education level
input_df['edu_lvl_High_School'] = 1 if edu_lvl == 'High_School' else 0
input_df['edu_lvl_Middle_School'] = 1 if edu_lvl == 'Middle_School' else 0
input_df['edu_lvl_Postgraduate'] = 1 if edu_lvl == 'Postgraduate' else 0
input_df['edu_lvl_Primary'] = 1 if edu_lvl == 'Primary' else 0
input_df['edu_lvl_University'] = 1 if edu_lvl == 'University' else 0

# Marital status
input_df['marital_status_Divorced'] = 1 if marital_status == 'Divorced' else 0
input_df['marital_status_Married'] = 1 if marital_status == 'Married' else 0
input_df['marital_status_Single'] = 1 if marital_status == 'Single' else 0
input_df['marital_status_Widowed'] = 1 if marital_status == 'Widowed' else 0

# Occupation
input_df['occupation_Employed'] = 1 if occupation == 'Employed' else 0
input_df['occupation_Retired'] = 1 if occupation == 'Retired' else 0
input_df['occupation_Student'] = 1 if occupation == 'Student' else 0
input_df['occupation_Unemployed'] = 1 if occupation == 'Unemployed' else 0

# Income level
input_df['income_lvl_High'] = 1 if income_lvl == 'High' else 0
input_df['income_lvl_Low'] = 1 if income_lvl == 'Low' else 0
input_df['income_lvl_Medium'] = 1 if income_lvl == 'Medium' else 0

# Living area
input_df['live_area_city'] = 1 if live_area == 'city' else 0
input_df['live_area_village'] = 1 if live_area == 'village' else 0

# Family History
input_df['Family_History'] = Family_History

# Substance use
input_df['Substance_use_Yes'] = 1 if Substance_use == 'Yes' else 0
input_df['Substance_use_No'] = 1 if Substance_use == 'No' else 0

# Suicide Attempt
input_df['Suicide_Attempt_Yes'] = 1 if Suicide_Attempt == 'Yes' else 0
input_df['Suicide_Attempt_No'] = 1 if Suicide_Attempt == 'No' else 0

# Social Support
input_df['Social_Support_High'] = 1 if Social_Support == 'High' else 0
input_df['Social_Support_Low'] = 1 if Social_Support == 'Low' else 0
input_df['Social_Support_Medium'] = 1 if Social_Support == 'Medium' else 0

# Stress Factors
input_df['Stress_Factors_High'] = 1 if Stress_Factors == 'High' else 0
input_df['Stress_Factors_Low'] = 1 if Stress_Factors == 'Low' else 0
input_df['Stress_Factors_Medium'] = 1 if Stress_Factors == 'Medium' else 0

# Присваиваем возраст отдельно
input_df['age'] = age

if st.button("Прогноз"):
    y_score = model.predict_proba(input_df)[:, 1]
    if y_score >= 0.35:
        st.error("⚠️ **Высокая вероятность заболевания!** Обратитесь к специалисту.")
    else:
        st.success("✅ **Вы здоровы!** Вероятность заболевания низкая.")
