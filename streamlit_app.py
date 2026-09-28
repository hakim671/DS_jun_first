import streamlit as st
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="Оценка риска шизофрении",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Design system
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --bg: #F1F5F2;
        --surface: #FFFFFF;
        --ink: #1B2B26;
        --ink-muted: #5B6E67;
        --accent: #2E6B5E;
        --accent-soft: #DCEAE5;
        --risk-high: #A8462F;
        --risk-low: #3F7A56;
        --border: #D7E0DB;
    }

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--ink);
    }

    .stApp { background: var(--bg); }

    .block-container {
        max-width: 980px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        font-family: 'Fraunces', serif;
        color: var(--ink);
        font-weight: 500;
    }

    p, li, label { color: var(--ink); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 2.2rem; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--ink-muted);
        margin-top: 1.8rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }
    [data-testid="stWidgetLabel"] p {
        font-size: 0.84rem;
        color: var(--ink-muted);
    }

    /* Inputs */
    .stNumberInput input, .stTextInput input {
        border-radius: 4px;
        border: 1px solid var(--border);
    }
    div[data-baseweb="select"] > div {
        border-radius: 4px;
        border-color: var(--border) !important;
    }

    /* Buttons */
    .stButton > button {
        background: var(--accent);
        color: #FFFFFF;
        border: none;
        border-radius: 4px;
        padding: 0.6rem 1.7rem;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        transition: background 0.15s ease;
    }
    .stButton > button:hover { background: #1E4A40; color: #FFFFFF; }
    .stButton > button:focus { box-shadow: 0 0 0 2px var(--accent-soft); }

    /* Callout */
    .callout {
        border-left: 3px solid var(--accent);
        background: var(--accent-soft);
        padding: 0.9rem 1.2rem;
        font-size: 0.92rem;
        color: var(--ink-muted);
        margin: 1.3rem 0 1.6rem 0;
    }

    .hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 2.2rem 0;
    }

    /* Result */
    .result-label {
        font-size: 1.05rem;
        color: var(--ink-muted);
        margin-bottom: 0.2rem;
    }
    .result-number {
        font-family: 'Fraunces', serif;
        font-size: 4.4rem;
        font-weight: 500;
        line-height: 1;
        margin: 0;
    }
    .risk-bar-track {
        width: 100%;
        height: 8px;
        background: var(--border);
        border-radius: 4px;
        overflow: hidden;
        margin: 1.1rem 0;
    }
    .risk-bar-fill { height: 100%; border-radius: 4px; }
    .result-note {
        font-size: 0.95rem;
        color: var(--ink-muted);
        max-width: 640px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

HERO_SVG = """
<svg viewBox="0 0 320 320" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
  <g stroke="#2E6B5E" stroke-width="1.4" fill="none" opacity="0.55">
    <line x1="60" y1="80" x2="120" y2="50"/>
    <line x1="120" y1="50" x2="180" y2="70"/>
    <line x1="180" y1="70" x2="230" y2="110"/>
    <line x1="60" y1="80" x2="110" y2="100"/>
    <line x1="110" y1="100" x2="90" y2="150"/>
    <line x1="120" y1="50" x2="160" y2="140"/>
    <line x1="180" y1="70" x2="160" y2="140"/>
    <line x1="230" y1="110" x2="220" y2="180"/>
    <line x1="90" y1="150" x2="160" y2="140"/>
    <line x1="160" y1="140" x2="220" y2="180"/>
    <line x1="90" y1="150" x2="80" y2="240"/>
    <line x1="160" y1="140" x2="140" y2="220"/>
    <line x1="220" y1="180" x2="190" y2="250"/>
    <line x1="140" y1="220" x2="80" y2="240"/>
    <line x1="140" y1="220" x2="190" y2="250"/>
    <line x1="220" y1="180" x2="250" y2="200"/>
  </g>
  <g fill="#2E6B5E">
    <circle cx="60" cy="80" r="5"/>
    <circle cx="120" cy="50" r="5"/>
    <circle cx="180" cy="70" r="5"/>
    <circle cx="230" cy="110" r="5"/>
    <circle cx="90" cy="150" r="5"/>
    <circle cx="220" cy="180" r="5"/>
    <circle cx="140" cy="220" r="5"/>
    <circle cx="80" cy="240" r="5"/>
    <circle cx="190" cy="250" r="5"/>
    <circle cx="250" cy="200" r="5"/>
    <circle cx="110" cy="100" r="5"/>
  </g>
  <circle cx="160" cy="140" r="9" fill="#A8462F"/>
</svg>
"""


@st.cache_data
def load_data():
    df = pd.read_excel("fnl_prj.xlsx")
    return df


@st.cache_resource
def get_model():
    df = load_data()
    X = df.drop(["diagnosis"], axis=1)
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.05, random_state=42, stratify=y
    )

    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    model = LogisticRegression(
        random_state=42, C=0.03, max_iter=120, penalty="l2", solver="saga"
    )
    model.fit(X_train, y_train)
    return model


model = get_model()

# ------------------------------------------------------------------
# Hero
# ------------------------------------------------------------------
hero_col, art_col = st.columns([3, 2], gap="large")

with hero_col:
    st.markdown("# Оценка риска шизофрении")
    st.markdown(
        "Модель логистической регрессии оценивает вероятность диагноза "
        "по демографическим и клиническим данным пациента, введённым слева."
    )
    st.markdown(
        """
        <div class="callout">
        Результат носит ориентировочный характер и не является медицинским
        диагнозом. Для точной оценки состояния обратитесь к психиатру.
        </div>
        """,
        unsafe_allow_html=True,
    )

with art_col:
    st.markdown(HERO_SVG, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Sidebar — patient data
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Данные пациента")

    st.markdown("## Демография")
    age = st.number_input(label="Возраст", value=40, step=5)
    gender = st.selectbox(label="Пол", options=["male", "female"], index=0)
    edu_lvl = st.selectbox(
        label="Образование",
        options=["High_School", "Middle_School", "Postgraduate", "Primary", "Undergraduate"],
        index=3,
    )
    marital_status = st.selectbox(
        label="Семейный статус",
        options=["Divorced", "Married", "Single", "Widowed"],
        index=2,
    )
    occupation = st.selectbox(
        label="Профессия",
        options=["Employed", "Retired", "Student", "Unemployed"],
        index=2,
    )
    income_lvl = st.selectbox(label="Уровень дохода", options=["High", "Low", "Medium"], index=2)
    live_area = st.selectbox(label="Место проживания", options=["city", "village"], index=0)

    st.markdown("## Клинические факторы")
    Family_History = st.selectbox(label="Есть в роду шизофреник?", options=["No", "Yes"], index=0)
    Substance_use = st.selectbox(label="Употребляет вредные вещества?", options=["No", "Yes"], index=0)
    Suicide_Attempt = st.selectbox(label="Попытки самоубийства?", options=["No", "Yes"], index=0)
    Social_Support = st.selectbox(label="Поддержка окружающих", options=["High", "Low", "Medium"], index=2)
    Stress_Factors = st.selectbox(label="Уровень стресса", options=["High", "Low", "Medium"], index=2)

# ------------------------------------------------------------------
# Build model input (unchanged logic)
# ------------------------------------------------------------------
input_df = pd.DataFrame({
    'age': [age],
    'Family_History': [0],
    'gender_female': [0],
    'gender_male': [0],
    'edu_lvl_High_School': [0],
    'edu_lvl_Middle_School': [0],
    'edu_lvl_Postgraduate': [0],
    'edu_lvl_Primary': [0],
    'edu_lvl_Undergraduate': [0],
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

if gender == 'male':
    input_df['gender_male'] = 1
else:
    input_df['gender_female'] = 1

input_df['edu_lvl_High_School'] = 1 if edu_lvl == 'High_School' else 0
input_df['edu_lvl_Middle_School'] = 1 if edu_lvl == 'Middle_School' else 0
input_df['edu_lvl_Postgraduate'] = 1 if edu_lvl == 'Postgraduate' else 0
input_df['edu_lvl_Primary'] = 1 if edu_lvl == 'Primary' else 0
input_df['edu_lvl_Undergraduate'] = 1 if edu_lvl == 'Undergraduate' else 0

input_df['marital_status_Divorced'] = 1 if marital_status == 'Divorced' else 0
input_df['marital_status_Married'] = 1 if marital_status == 'Married' else 0
input_df['marital_status_Single'] = 1 if marital_status == 'Single' else 0
input_df['marital_status_Widowed'] = 1 if marital_status == 'Widowed' else 0

input_df['occupation_Employed'] = 1 if occupation == 'Employed' else 0
input_df['occupation_Retired'] = 1 if occupation == 'Retired' else 0
input_df['occupation_Student'] = 1 if occupation == 'Student' else 0
input_df['occupation_Unemployed'] = 1 if occupation == 'Unemployed' else 0

input_df['income_lvl_High'] = 1 if income_lvl == 'High' else 0
input_df['income_lvl_Low'] = 1 if income_lvl == 'Low' else 0
input_df['income_lvl_Medium'] = 1 if income_lvl == 'Medium' else 0

input_df['live_area_city'] = 1 if live_area == 'city' else 0
input_df['live_area_village'] = 1 if live_area == 'village' else 0

input_df['Family_History'] = 1 if Family_History == "Yes" else 0

input_df['Substance_use_Yes'] = 1 if Substance_use == 'Yes' else 0
input_df['Substance_use_No'] = 1 if Substance_use == 'No' else 0

input_df['Suicide_Attempt_Yes'] = 1 if Suicide_Attempt == 'Yes' else 0
input_df['Suicide_Attempt_No'] = 1 if Suicide_Attempt == 'No' else 0

input_df['Social_Support_High'] = 1 if Social_Support == 'High' else 0
input_df['Social_Support_Low'] = 1 if Social_Support == 'Low' else 0
input_df['Social_Support_Medium'] = 1 if Social_Support == 'Medium' else 0

input_df['Stress_Factors_High'] = 1 if Stress_Factors == 'High' else 0
input_df['Stress_Factors_Low'] = 1 if Stress_Factors == 'Low' else 0
input_df['Stress_Factors_Medium'] = 1 if Stress_Factors == 'Medium' else 0

input_df['age'] = age

# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
st.markdown('<hr class="hr" />', unsafe_allow_html=True)

if st.button("Оценить риск"):
    y_score = model.predict_proba(input_df)[:, 1][0]
    percent = round(y_score * 100, 1)
    is_high = y_score >= 0.35

    color = "var(--risk-high)" if is_high else "var(--risk-low)"
    label = "Повышенный риск" if is_high else "Низкий риск"
    note = (
        "Показатели указывают на повышенную вероятность. "
        "Рекомендуется очная консультация психиатра для дальнейшей оценки."
        if is_high else
        "Показатели не указывают на повышенную вероятность. "
        "При наличии тревожных симптомов консультация специалиста всё же уместна."
    )

    st.markdown(
        f"""
        <div class="result-label">{label}</div>
        <p class="result-number">{percent}%</p>
        <div class="risk-bar-track">
            <div class="risk-bar-fill" style="width:{percent}%; background:{color};"></div>
        </div>
        <p class="result-note">{note}</p>
        """,
        unsafe_allow_html=True,
    )
