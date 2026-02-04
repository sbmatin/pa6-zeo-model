import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="PA6-Zeolite Optimizer", layout="wide")

st.title("PA6-Zeolite Composite Performance Predictor")
st.markdown("""
This tool predicts the mechanical and tribological properties of Zeolite-filled Polyamide-6 composites 
based on experimental data from the study: **'Tribological Behaviour of Zeolite Filled PA6 Composites'**.
""")

X = np.array([0, 20, 40, 50, 60]).reshape(-1, 1)
y_hardness = np.array([85.40, 87.82, 90.37, 92.18, 95.54])
y_uts = np.array([54.51, 56.90, 65.70, 53.16, 42.72])

model_h = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X, y_hardness)
model_u = make_pipeline(PolynomialFeatures(3), LinearRegression()).fit(X, y_uts)

st.sidebar.header("User Input Parameters")
zeolite_pct = st.sidebar.slider("Select Zeolite Content (wt%)", 0, 60, 30)

h_pred = model_h.predict([[zeolite_pct]])[0]
u_pred = model_u.predict([[zeolite_pct]])[0]

col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted Hardness", f"{h_pred:.2f} Shore D")
    
with col2:
    st.metric("Predicted Tensile Strength (UTS)", f"{u_pred:.2f} MPa")

st.subheader("Performance Trends")
X_seq = np.linspace(0, 60, 100).reshape(-1, 1)

fig, ax1 = plt.subplots(figsize=(10, 4))
ax2 = ax1.twinx()

ax1.plot(X_seq, model_h.predict(X_seq), color='blue', label='Hardness Prediction')
ax1.scatter(X, y_hardness, color='blue', alpha=0.5)
ax1.set_ylabel('Hardness (Shore D)', color='blue')

ax2.plot(X_seq, model_u.predict(X_seq), color='red', label='UTS Prediction')
ax2.scatter(X, y_uts, color='red', alpha=0.5)
ax2.set_ylabel('UTS (MPa)', color='red')

ax1.axvline(zeolite_pct, color='green', linestyle='--', label='Your Selection')

st.pyplot(fig)

st.info(f" **Analysis:** At {zeolite_pct}wt%, the material is in the {'Optimal' if 20<=zeolite_pct<=40 else 'Sub-optimal'} window for engineering applications.")

