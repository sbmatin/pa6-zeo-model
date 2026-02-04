**Predictive Modeling for Zeolite-Filled PA6 Composites**

**Overview:**
This repository contains a data-driven predictive model designed to estimate the mechanical and tribological performance of Polyamide-6 (PA6) composites reinforced with natural zeolite.

The goal of this project is to provide a Digital Twin for material design, allowing researchers to predict properties like Tensile Strength and Wear Rate without the need for exhaustive and expensive laboratory testing.

**Key Predictions:**

Shore D Hardness: Modeling the monotonic stiffening effect of ceramic fillers.

Ultimate Tensile Strength (UTS): Predicting the non-monotonic peak performance loading.

Specific Wear Rate: Mapping the transition from adhesive to abrasive wear mechanisms.

**Scientific Context:**

This model is based on experimental data from the manuscript:

"Tribological Behaviour of Zeolite Filled PA6 Composites" > Authors: Matin, S.B., et al.

**Key Finding:** The study identifies an "Optimal Performance Window" between 20 wt% and 40 wt% zeolite loading, where the composite achieves maximum structural integrity and balanced wear resistance.

**Technical Stack:**

Language: Python 3.x

Modeling: Scikit-Learn (Polynomial Regression)

Deployment: Streamlit Cloud

Visualization: Matplotlib / Seaborn

Why Polynomial Regression?
Material properties in polymer composites often exhibit non-linear behaviors (e.g., strength increases with filler loading until agglomeration causes a sharp drop). We utilized 2nd and 3rd-degree polynomial features to accurately capture these performance peaks which standard linear models would fail to identify.

**How to Use**

1. Interactive Web Dashboard
The easiest way to explore the model is through the live Streamlit Dashboard. Use the slider to select a Zeolite weight percentage and view real-time property predictions.

2. Local Installation
To run the model locally, follow these steps:

Clone the repository: git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

Install dependencies: pip install -r requirements.txt

Run the app: streamlit run app.py

**Data & Validation:**
The model was validated against experimental results using the Coefficient of Determination ($R^2$):
- **Hardness Accuracy:** 98.90% ($R^2 = 0.9890$)
- **UTS Accuracy:** 90.28% ($R^2 = 0.9028$)

This validation proves the model is a reliable tool for predicting composite behavior within the 0-60 wt% zeolite loading range.

**Contact & Citation:**
If you use this model in your research, please cite the original paper: Matin, S. B. (2026). TRIBOLOGICAL BEHAVIOUR OF ZEOLITE FILLED POLYAMIDE-6 COMPOSITES. Zenodo. https://doi.org/10.5281/zenodo.18215181

Developed by: SADDAM BIN MATIN

Connect with me on linkedin.com/in/saddam-bin-matin


