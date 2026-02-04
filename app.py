import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="PA6-Zeolite Optimizer", layout="wide")

st.title("PA6-Zeolite Performance Predictor")
st.markdown("Predicting Mechanical and Tribological properties based on filler weight percentage.")

# X = Zeolite Content
X = np.array([0, 20, 40, 50, 60]).reshape(-1, 1)
X_wear = np.array([0, 20, 40, 50]).reshape(-1, 1)

y_hardness = np.array([85.40, 87.82, 90.37, 92.18, 95.54])
y_uts = np.array([54.51, 56.90, 65.70, 53.16, 42.72])
y_wear = np.array([2.97, 10.28, 39.51, 63.78]) # 10^-9 mm3/Nm

model_h = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X, y_hardness)
model_u = make_pipeline(PolynomialFeatures(3), LinearRegression()).fit(X, y_uts)
model_w = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X_wear, y_wear)

st.sidebar.header("Material Composition")
zeolite_pct = st.sidebar.slider("Zeolite Content (wt%)", 0, 60, 30)

h_pred = model_h.predict([[zeolite_pct]])[0]
u_pred = model_u.predict([[zeolite_pct]])[0]

if zeolite_pct <= 55:
    w_pred = model_w.predict([[zeolite_pct]])[0]
    w_display = f"{w_pred:.2f}"
else:
    w_display = "N/A (Critical Failure)"

col1, col2, col3 = st.columns(3)
col1.metric("Hardness", f"{h_pred:.2f} Shore D")
col2.metric("Tensile Strength", f"{u_pred:.2f} MPa")
col3.metric("Spec. Wear Rate", f"{w_display}", delta="Units: 10⁻⁹ mm³/Nm", delta_color="inverse")

st.subheader("Performance Analysis & Optimization Window")
X_seq = np.linspace(0, 60, 100).reshape(-1, 1)

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

ax1.plot(X_seq, model_u.predict(X_seq), color='red', label='UTS (Strength)', linewidth=2)
ax1.plot(X_seq, model_h.predict(X_seq), color='blue', linestyle='--', label='Hardness')
ax1.set_ylabel("Mechanical Properties (MPa / Shore D)")

X_seq_w = np.linspace(0, 55, 100).reshape(-1, 1)
ax2.plot(X_seq_w, model_w.predict(X_seq_w), color='green', label='Wear Rate', linewidth=3)
ax2.set_ylabel("Wear Rate ($10^{-9} mm^3/Nm$)", color='green')

ax1.axvspan(20, 40, color='yellow', alpha=0.2, label='Optimal Performance Window')
ax1.axvline(zeolite_pct, color='black', linewidth=1)

fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85))
st.pyplot(fig)

if 20 <= zeolite_pct <= 40:
    st.success("✅ **Optimal Window:** This loading maintains high UTS while controlling wear escalation.")
elif zeolite_pct > 40:
    st.error("⚠️ **Warning:** Particle agglomeration likely. Wear rate increases drastically due to abrasive third-body particles.")
else:
    st.info("ℹ️ **Low Loading:** Material behaves similarly to neat PA6 with minor reinforcement.")

from sklearn.metrics import r2_score

y_h_true = np.array([85.40, 87.82, 90.37, 92.18, 95.54])
y_u_true = np.array([54.51, 56.90, 65.70, 53.16, 42.72])

y_h_pred = model_h.predict(X)
y_u_pred = model_u.predict(X)

r2_hardness = r2_score(y_h_true, y_h_pred)
r2_uts = r2_score(y_u_true, y_u_pred)

print(f"R² for Hardness: {r2_hardness:.4f}")
print(f"R² for UTS:      {r2_uts:.4f}")

comparison_df = pd.DataFrame({
    'Zeolite wt%': [0, 20, 40, 50, 60],
    'Exp. UTS (MPa)': y_u_true,
    'Model UTS (MPa)': np.round(y_u_pred, 2),
    'Error (%)': np.round(np.abs((y_u_true - y_u_pred)/y_u_true)*100, 2)
})
print("\nValidation Table for UTS:")
print(comparison_df)

from sklearn.metrics import r2_score

y_h_true = np.array([85.40, 87.82, 90.37, 92.18, 95.54])
y_u_true = np.array([54.51, 56.90, 65.70, 53.16, 42.72])

y_h_model = model_h.predict(X)
y_u_model = model_u.predict(X)

r2_h = r2_score(y_h_true, y_h_model)
r2_u = r2_score(y_u_true, y_u_model)

st.sidebar.markdown("### Model Validation")
st.sidebar.write(f"Hardness $R^2$: **{r2_h:.4f}**")
st.sidebar.write(f"UTS $R^2$: **{r2_u:.4f}**")

with st.expander("View Data Comparison Table (Model vs. Experimental)"):
    comparison_df = pd.DataFrame({
        'Zeolite (wt%)': [0, 20, 40, 50, 60],
        'Exp. UTS (MPa)': y_u_true,
        'Model UTS (MPa)': np.round(y_u_model, 2),
        'Accuracy (%)': np.round(100 - (np.abs(y_u_true - y_u_model)/y_u_true)*100, 2)
    })
    st.table(comparison_df)

st.sidebar.markdown("---")
st.sidebar.markdown("### Developed by")
st.sidebar.markdown("**Saddam Bin Matin**")
st.sidebar.markdown("[LinkedIn Profile](https://linkedin.com/in/saddam-bin-matin)")
st.sidebar.markdown("[Research Paper](https://doi.org/10.5281/zenodo.18215181)")
st.sidebar.markdown("---")





