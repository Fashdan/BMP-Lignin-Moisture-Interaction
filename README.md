BMP Prediction & XAI Analysis
This repository contains the Python code and machine learning workflow associated with the manuscript:
"Explainable AI Reveals a Critical Interplay Between Lignin and Moisture in Limiting Biochemical Methane Potential"

Overview
The study utilizes Machine Learning (Random Forest) to predict the Biochemical Methane Potential (BMP) of diverse agricultural feedstocks. Beyond prediction, the project employs Explainable AI (SHAP) to uncover biochemical drivers, specifically identifying a robust interaction between Dry Matter (moisture) content and Lignin recalcitrance.
Contents

bmp_model_analysis.py: The main script that performs data preprocessing, trains the Random Forest model, calculates performance metrics (R2, RMSE), and generates SHAP visualization plots.

requirements.txt: List of Python dependencies.

Key Features
Data Filtering: Focuses on solid and semi-solid feedstocks (Dry Matter ≥ 15%).
Model: Random Forest Regressor optimized for diverse biomass types.
Interpretation: - SHAP Summary plots to validate biochemical principles (e.g., Lignin inhibition).
SHAP Dependence plots to visualize the Lignin-Moisture interaction hypothesis.

Usage
Install Dependencies:
pip install -r requirements.txt

Prepare Data:
Ensure your dataset (CSV) is placed in the project directory. Update the DATA_PATH variable in bmp_model_analysis.py if your filename differs from table_complete.csv.

Run Analysis:
python bmp_model_analysis.py


Citation
If you use this code or the associated findings, please cite the manuscript:
Fasheun, D.O., Omage, F.B., Ferreira-Leitão, V.S. (2025). Explainable AI Reveals a Critical Interplay Between Lignin and Moisture in Limiting Biochemical Methane Potential.