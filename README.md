# PA6–Zeolite Property Explorer

An experimental-data explorer and **exploratory** predictive-modelling dashboard for PA6 composites filled with natural zeolite. The Streamlit application compares conventional quadratic regression with Support Vector Regression (SVR).

**Live app:** https://pa6-zeo-model.streamlit.app/

## Predictions and measured ranges

| Property | Measured zeolite contents (wt%) | Default model | Notes |
|---|---|---|---|
| Shore D hardness | 0, 20, 40, 50, 60, 70 | Quadratic regression | 10 indents at each composition |
| Ultimate tensile strength | 0, 20, 40, 50, 60 | RBF-SVR with fixed parameters | Retained valid tensile specimens: n = 2, 3, 3, 3, 3 |
| Steel-ball counterface specific wear | 0, 20, 40, 50 | Linear SVR on log(wear rate) | Both 5000 m and 7500 m tests are represented |

The sliding wear values refer to the **steel-ball counterface**, **not** the wear of the PA6 composite specimen.

## Validation

- **Hardness and tensile:** leave-one-composition-out cross-validation, applied to the composition means.
- **Counterface wear:** leave-one-composition-out *grouped* cross-validation: the 5000 m and 7500 m records for a composition are both withheld in the same fold.
- Reported RMSE, MAE and R² use held-out predictions. Ordinary training fit is not described as prediction accuracy.
- SVR hyperparameters are fixed and documented in `app.py`, as explored during a pilot study. Model and hyperparameter choices have been explored on these same small data, so the dashboard is **not independently validated**, and is **not** a digital twin.

## Important limitations

1. The steel ball was modelled using **nominal 8 mm diameter** and **10 N normal force**; the ball size needs checking against original laboratory records. The wear-rate values should remain provisional until this is verified.
2. The eight steel-ball scar diameters are historical optical measurements from thesis Table 4.1; the calibrated TIFF images were used for qualitative auditing, not to replace them.
3. At 70 wt% zeolite a tensile-shaped specimen could be molded and used for hardness testing, but a suitable tribometer disc could not be produced. Tensile strength was not reported at 70 wt%.
4. The code does not extrapolate beyond each property's tested zeolite range. Values between compositions are modelled *interpolations* that need experimental verification.
5. The data are very small and some tests were excluded because of known grip-tightening errors. The results do not establish an experimentally validated optimum composition.

## Run locally

```bash
git clone https://github.com/sbmatin/pa6-zeo-model.git
cd pa6-zeo-model
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` — data, model definitions, cross-validation and dashboard.
- `requirements.txt` — deployment dependencies.
- `README.md` — documentation and scientific limitations.

## Source and citation

Laboratory data: Saddam Bin Matin, Master's thesis (University of Miskolc), original hardness and tensile spreadsheets, and thesis Table 4.1 (steel-ball wear scar diameters).

Related preprint: https://doi.org/10.5281/zenodo.18496332

Developer: **Saddam Bin Matin** · [LinkedIn](https://linkedin.com/in/saddam-bin-matin)
