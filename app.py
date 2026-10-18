import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


# ------------------------------------------------------------
# 1. EXPERIMENTAL DATA
# ------------------------------------------------------------
# Content is mass percent (wt%), not volume percent.
H_X = np.array([0, 20, 40, 50, 60, 70], dtype=float)
H_Y = np.array([85.40, 87.82, 90.37, 92.18, 95.54, 97.88])
H_SD = np.array([0.306, 0.410, 0.794, 0.308, 1.731, 0.114])
H_N = np.array([10] * 6)  # indentations, not independent batches

T_X = np.array([0, 20, 40, 50, 60], dtype=float)
T_Y = np.array([
    54.51467558365336,
    56.902208020720245,
    65.70447368293817,
    53.15598005694136,
    42.71860057543122,
])
T_SD = np.array([0.01192335, 7.34418581, 1.17964393, 5.03316895, 0.24514176])
T_N = np.array([2, 3, 3, 3, 3])  # retained valid tensile specimens

# Thesis Table 4.1, organized as (loading wt%, sliding distance m, diameter mm).
SCAR_ROWS = np.array([
    [0, 5000, 0.330], [0, 7500, 0.380],
    [20, 5000, 0.450], [20, 7500, 0.490],
    [40, 5000, 0.630], [40, 7500, 0.700],
    [50, 5000, 0.710], [50, 7500, 0.766],
], dtype=float)

# NOMINAL geometry/load: verify the steel ball diameter against lab records.
BALL_RADIUS_MM = 4.0           # 8-mm nominal steel ball diameter (assumed)
NORMAL_FORCE_N = 10.0         # thesis nominal normal force


def spherical_cap_volume(diameter_mm):
    """Wear volume removed from a spherical steel-ball tip, mm^3."""
    a = np.asarray(diameter_mm, dtype=float) / 2.0
    h = BALL_RADIUS_MM - np.sqrt(BALL_RADIUS_MM**2 - a**2)
    return np.pi * h**2 * (3 * BALL_RADIUS_MM - h) / 3.0


W_Z = SCAR_ROWS[:, 0]
W_L = SCAR_ROWS[:, 1]
W_D = SCAR_ROWS[:, 2]
# Scale the path length in km for numerical conditioning. This is still a
# two-feature model: zeolite content AND sliding distance.
W_X = np.column_stack([W_Z, W_L / 1000.0])
W_VOLUME = spherical_cap_volume(W_D)
# Values represented in convenient units of 10^-9 mm^3/(N*m).
W_Y = W_VOLUME / (NORMAL_FORCE_N * W_L) * 1e9


# ------------------------------------------------------------
# 2. CANDIDATE MODELS
# ------------------------------------------------------------
def quadratic_regression():
    return make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
    )


def quadratic_wear_regression():
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        LinearRegression(),
    )


def rbf_svr():
    # Same fixed parameters as the documented pilot analysis. Scaling is
    # learned INSIDE each training fold, preventing preprocessing leakage.
    return TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", C=2.0, epsilon=0.1, gamma="scale"),
        ),
        transformer=StandardScaler(),
    )


def positive_wear_svr():
    # Modeling log(wear rate) ensures a strictly positive wear estimate.
    return TransformedTargetRegressor(
        regressor=make_pipeline(
            StandardScaler(),
            SVR(kernel="linear", C=2.0, epsilon=0.1),
        ),
        func=np.log,
        inverse_func=np.exp,
        check_inverse=True,
    )


DATASETS = {
    "Hardness": {
        "X": H_X.reshape(-1, 1), "y": H_Y, "groups": None,
        "unit": "Shore D",
        "models": {
            "Quadratic regression": quadratic_regression,
            "RBF-SVR (fixed)": rbf_svr,
        },
        "default": "Quadratic regression",
    },
    "Tensile strength": {
        "X": T_X.reshape(-1, 1), "y": T_Y, "groups": None,
        "unit": "MPa",
        "models": {
            "RBF-SVR (fixed)": rbf_svr,
            "Quadratic regression": quadratic_regression,
        },
        "default": "RBF-SVR (fixed)",
    },
    "Steel-ball wear": {
        "X": W_X, "y": W_Y, "groups": W_Z,
        "unit": "×10⁻⁹ mm³/(N·m)",
        "models": {
            "Linear SVR (log wear)": positive_wear_svr,
            "Quadratic regression": quadratic_wear_regression,
            "RBF-SVR (fixed)": rbf_svr,
        },
        "default": "Linear SVR (log wear)",
    },
}


def cross_validated_predictions(dataset, model_factory):
    """Leave one composition out; for wear, remove both distances together."""
    X, y, groups = dataset["X"], dataset["y"], dataset["groups"]
    cv = LeaveOneGroupOut() if groups is not None else LeaveOneOut()
    y_cv = np.empty(len(y), dtype=float)
    for train_idx, test_idx in cv.split(X, y, groups):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        y_cv[test_idx] = model.predict(X[test_idx])
    return y_cv


def scores(actual, predicted):
    return {
        "LOCO R²": float(r2_score(actual, predicted)),
        "LOCO RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "LOCO MAE": float(mean_absolute_error(actual, predicted)),
    }


@st.cache_resource(show_spinner=False)
def fitted_models():
    fitted = {}
    for property_name, dataset in DATASETS.items():
        fitted[property_name] = {}
        for name, factory in dataset["models"].items():
            fitted[property_name][name] = factory().fit(dataset["X"], dataset["y"])
    return fitted


@st.cache_data(show_spinner=False)
def validation_results():
    results, held_out = [], {}
    for property_name, dataset in DATASETS.items():
        held_out[property_name] = {}
        for name, factory in dataset["models"].items():
            predicted = cross_validated_predictions(dataset, factory)
            held_out[property_name][name] = predicted
            values = scores(dataset["y"], predicted)
            results.append({"Property": property_name, "Model": name, **values})
    return pd.DataFrame(results), held_out


def predict_one(fitted, property_name, model_name, zeolite, distance_m):
    limits = {"Hardness": 70, "Tensile strength": 60, "Steel-ball wear": 50}
    if zeolite > limits[property_name]:
        return None
    X = (np.array([[zeolite, distance_m / 1000.0]])
         if property_name == "Steel-ball wear"
         else np.array([[zeolite]]))
    value = float(fitted[property_name][model_name].predict(X)[0])
    if not np.isfinite(value) or value < 0:
        # In particular, do not report a negative quadratic wear prediction.
        return None
    return value


def property_plot(property_name, model, selected_zeolite, distance_m):
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    domain = {"Hardness": 70, "Tensile strength": 60, "Steel-ball wear": 50}[property_name]
    x_line = np.linspace(0, domain, 201)
    if property_name == "Hardness":
        ax.errorbar(H_X, H_Y, yerr=H_SD, fmt="o", capsize=3,
                    label="Measured mean ± SD")
        ax.plot(x_line, model.predict(x_line.reshape(-1, 1)), label="Model fit")
        ax.set_ylabel("Shore D")
    elif property_name == "Tensile strength":
        ax.errorbar(T_X, T_Y, yerr=T_SD, fmt="o", capsize=3,
                    label="Measured mean ± SD")
        ax.plot(x_line, model.predict(x_line.reshape(-1, 1)), label="Model fit")
        ax.set_ylabel("Tensile strength (MPa)")
    else:
        for dist in (5000, 7500):
            vals = model.predict(np.column_stack([x_line, np.full_like(x_line, dist / 1000)]))
            # Hide physically invalid negative quadratic predictions (not clamp to zero).
            vals = np.where(vals >= 0, vals, np.nan)
            ax.plot(x_line, vals, label=f"Model, {dist:,} m",
                    linewidth=2.4 if dist == distance_m else 1.3,
                    linestyle="-" if dist == distance_m else "--")
            ix = W_L == dist
            ax.scatter(W_Z[ix], W_Y[ix], s=45, label=f"Measured, {dist:,} m")
        ax.set_ylabel("Steel-ball specific wear (×10⁻⁹ mm³/[N·m])")
    ax.axvline(selected_zeolite, linestyle=":", alpha=0.65, label="Chosen loading")
    ax.set_xlabel("Zeolite loading (wt%)")
    ax.set_xlim(0, domain)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def comparison_table(property_name, model, cv_prediction):
    dataset = DATASETS[property_name]
    fitted_prediction = model.predict(dataset["X"])
    out = {
        "Zeolite (wt%)": H_X if property_name == "Hardness"
                           else T_X if property_name == "Tensile strength" else W_Z,
    }
    if property_name == "Steel-ball wear":
        out["Distance (m)"] = W_L.astype(int)
        out["Scar diameter (mm)"] = W_D
    out["Measured"] = dataset["y"]
    if property_name != "Steel-ball wear":
        out["Sample SD"] = H_SD if property_name == "Hardness" else T_SD
        out["n"] = H_N if property_name == "Hardness" else T_N
    out["Fitted"] = fitted_prediction
    out["Held-out prediction"] = cv_prediction
    return pd.DataFrame(out).round(3)


# ------------------------------------------------------------
# 3. STREAMLIT DASHBOARD
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="PA6–Zeolite Property Explorer", layout="wide")
    st.title("PA6–Zeolite Property Explorer")
    st.caption("Experimental composites · exploratory regression and support vector regression")
    st.info(
        "These models interpolate a very small experimental dataset. They have **not** "
        "been independently validated and do not constitute a digital twin. "
        "Predictions are estimates, not substitutes for experiments."
    )

    st.sidebar.header("Material composition")
    zeolite = st.sidebar.slider("Zeolite content (wt%)", 0, 70, 30, step=1)
    distance = st.sidebar.selectbox("Sliding distance (tribology)", [5000, 7500],
                                    format_func=lambda v: f"{v:,} m")
    st.sidebar.subheader("Choose a model")
    chosen = {}
    for property_name, dataset in DATASETS.items():
        choices = list(dataset["models"])
        chosen[property_name] = st.sidebar.selectbox(
            property_name, choices, index=choices.index(dataset["default"])
        )

    fitted = fitted_models()
    score_table, holdouts = validation_results()
    predictions = {
        prop: predict_one(fitted, prop, chosen[prop], zeolite, distance)
        for prop in DATASETS
    }

    c1, c2, c3 = st.columns(3)
    with c1:
        val = predictions["Hardness"]
        st.metric("Shore D hardness", f"{val:.2f}" if val is not None else "N/A")
        st.caption("0–70 wt% examined")
    with c2:
        val = predictions["Tensile strength"]
        st.metric("Tensile strength (MPa)", f"{val:.2f}" if val is not None else "N/A")
        st.caption("0–60 wt% examined")
    with c3:
        val = predictions["Steel-ball wear"]
        st.metric("Steel-ball specific wear (×10⁻⁹)",
                  f"{val:.2f}" if val is not None else "N/A")
        st.caption(f"0–50 wt%, {distance:,} m · mm³/(N·m)")

    if zeolite > 50:
        st.warning("No tribometer disc was tested above 50 wt%: wear prediction is unavailable.")
    if zeolite > 60:
        st.warning("No tensile-strength result is available at 70 wt%: only hardness is shown.")
    if zeolite not in [0, 20, 40, 50, 60, 70]:
        st.caption("Selected composition is between tested formulations: this is model interpolation.")

    st.subheader("Property curves and laboratory data")
    tab_h, tab_t, tab_w, tab_cv = st.tabs([
        "Hardness", "Tensile strength", "Steel-ball wear", "Model validation"
    ])
    for tab, prop in [(tab_h, "Hardness"), (tab_t, "Tensile strength"),
                      (tab_w, "Steel-ball wear")]:
        with tab:
            name = chosen[prop]
            model = fitted[prop][name]
            st.markdown(f"**Selected model:** {name}")
            fig = property_plot(prop, model, zeolite, distance)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption("Fitted curve uses all available composition means. "
                       "The table separately shows predictions for withheld compositions.")
            st.dataframe(comparison_table(prop, model, holdouts[prop][name]),
                         hide_index=True, use_container_width=True)
            if prop == "Steel-ball wear":
                st.warning(
                    "Wear is measured on the **steel-ball counterface**, not the PA6 disc. "
                    "Values are provisional because they use an assumed 8 mm ball diameter "
                    "and a nominal 10 N normal load. The ball diameter should be checked "
                    "against the laboratory instrument records."
                )
                st.caption("The eight original scar diameters come from thesis Table 4.1. "
                           "Wear volume uses spherical-cap geometry and k = V/(F×L).")

    with tab_cv:
        st.markdown("**Leave-one-composition-out evaluation**")
        st.write(
            "Hardness and tensile strength: withhold one composition mean at a time. "
            "Counterface wear: withhold BOTH sliding distances at the same composition "
            "to avoid leakage across related measurements. Scaling is refitted within each fold."
        )
        st.dataframe(score_table.round(3), hide_index=True, use_container_width=True)
        st.caption("R² below zero means the model performs worse than predicting the "
                   "mean of all measured values. RMSE/MAE use each property's displayed units.")
        st.warning(
            "Model types and fixed SVR settings were examined using these same small datasets. "
            "Therefore, these are internal exploratory validation results, not estimates "
            "from a completely independent test set."
        )

    with st.expander("Experimental coverage, exclusions and data sources"):
        st.write(
            "**Hardness:** 0, 20, 40, 50, 60 and 70 wt%; ten indents per composition. "
            "A 70 wt% tensile-shaped specimen was produced for hardness measurements, "
            "but no suitable tribometer disc could be made."
        )
        st.write(
            "**Tensile strength:** 0, 20, 40, 50 and 60 wt%; valid tests retained "
            "(n = 2, 3, 3, 3, 3). Specimens affected by improperly tightened "
            "machine grips were excluded."
        )
        st.write(
            "**Tribology:** 0, 20, 40 and 50 wt%; steel-ball scar diameter at "
            "5000 and 7500 m. The eight measurements are not independent replicates "
            "at the same composition and distance."
        )
        st.write(
            "Data: original Shore D Hardness.xlsx, Tensile Measurements.xlsx "
            "('Tensile Modulus' sheet), and master's thesis Table 4.1."
        )
        st.markdown(
            "[Research preprint](https://doi.org/10.5281/zenodo.18496332) | "
            "[Source code](https://github.com/sbmatin/pa6-zeo-model)"
        )

    st.sidebar.divider()
    st.sidebar.markdown("**Developed by Saddam Bin Matin**")
    st.sidebar.markdown("[LinkedIn](https://linkedin.com/in/saddam-bin-matin)")
    st.sidebar.markdown("[GitHub repository](https://github.com/sbmatin/pa6-zeo-model)")


if __name__ == "__main__":
    main()
