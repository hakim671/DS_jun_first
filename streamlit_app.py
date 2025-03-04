import streamlit as st
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_excel('final_prj.xlsx')
