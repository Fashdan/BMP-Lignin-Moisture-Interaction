# -*- coding: utf-8 -*-
"""
Biochemical Methane Potential (BMP) Prediction and XAI Analysis
===============================================================

This script reproduces the machine learning analysis and Explainable AI (XAI) 
results presented in the manuscript:
"Explainable AI Reveals a Critical Interplay Between Lignin and Moisture 
in Limiting Biochemical Methane Potential".

It performs the following:
1. Loads and cleans agricultural feedstock data.
2. Filters for solid/semi-solid feedstocks (Dry Matter >= 15%).
3. Trains a Random Forest Regressor (the champion model).
4. Evaluates performance using 5-Fold Cross-Validation.
5. Generates SHAP plots:
   - Figure 1A: Global Feature Importance (Bar Chart)
   - Figure 1B: Global Feature Impact (Beeswarm Plot)
   - Figure 2: Interaction Dependence Plot (Lignin vs. DM)
   - Figure 3: Stability Analysis (Aggregated Interaction over 5 Folds)

Author: Daniel O. Fasheun
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error

# --- Configuration & Constants ---

# File Configuration
DATA_PATH = "C:/Users/danie/Downloads/BMP_Project_Fasheun_2025/table_complete.csv"  # Path to the Lallement et al. (2023) dataset

# Feature Definitions
NUMERICAL_FEATURES = [
    'DM Mean (% FM)', 
    'VS Mean (% FM)', 
    'C/N', 
    'Carbon Mean (% DM)',
    'Hydrogen Mean (% DM)', 
    'Nitrogen Mean (% DM)', 
    'Sulfur Mean (% DM)',
    'Oxygen Mean (% DM)', 
    'Cellulose Mean (g/100g DM)',
    'Hemicelluloses Mean (g/100g DM)', 
    'Lignin Mean (g/100g DM)'
]

TARGET_COL_NAME = 'BMP_DM (Nm3 CH4/t DM)'

# Model Hyperparameters
RF_PARAMS = {
    'n_estimators': 50,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# Plotting Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)


def load_and_preprocess(filepath):
    """
    Loads dataset, calculates target BMP, and filters for DM >= 15%.
    """
    try:
        df = pd.read_csv(filepath, na_values='---', thousands=',')
        print(f"Data Loaded: {df.shape[0]} samples.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please check the path.")

    # Target Creation: BMP_DM = BMP_VS * (VS/DM ratio)
    # Adjust column names below if raw CSV headers differ slightly
    bmp_vs_col = 'BMP exp Mean (Nm3 CH4/t VS)'
    vs_dm_ratio_col = 'VS/DM Mean (% FM)'
    
    # Ensure numeric types
    cols_to_numeric = NUMERICAL_FEATURES + [bmp_vs_col, vs_dm_ratio_col]
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate Target
    df[TARGET_COL_NAME] = df[bmp_vs_col] * df[vs_dm_ratio_col]

    # Drop missing essential values
    df = df.dropna(subset=NUMERICAL_FEATURES + [TARGET_COL_NAME])

    # Filter: Dry Matter >= 15% (Solid/Semi-solid feedstocks)
    df_filtered = df[df['DM Mean (% FM)'] >= 15].copy()
    
    print(f"Filtered Data (DM >= 15%): {df_filtered.shape[0]} samples.")
    
    return df_filtered[NUMERICAL_FEATURES], df_filtered[TARGET_COL_NAME]


def evaluate_model(X, y):
    """
    Performs 5-Fold Cross-Validation and prints performance metrics.
    """
    # Preprocessing Pipeline
    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(**RF_PARAMS))
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calculate R2
    r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
    
    # Calculate RMSE
    rmse_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='neg_root_mean_squared_error')
    rmse_scores = -rmse_scores  # Convert back to positive

    print("\n" + "="*40)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("="*40)
    print(f"Mean R2:   {np.mean(r2_scores):.3f} (+/- {np.std(r2_scores):.3f})")
    print(f"Mean RMSE: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})")
    print("="*40 + "\n")
    
    return pipeline


def run_shap_analysis(pipeline, X, y):
    """
    Fits the model on the full dataset and generates Figures 1A, 1B & 2.
    """
    print("Running Global SHAP analysis (Figures 1A, 1B & 2)...")
    
    # 1. Fit pipeline on full dataset
    pipeline.fit(X, y)
    
    # 2. Extract model and transform data
    model = pipeline.named_steps['model']
    preprocessor = Pipeline(steps=[
        ('imputer', pipeline.named_steps['imputer']),
        ('scaler', pipeline.named_steps['scaler'])
    ])
    
    # Transform X to match what the model sees (scaled/imputed)
    X_transformed = preprocessor.transform(X)
    X_transformed_df = pd.DataFrame(X_transformed, columns=X.columns)

    # 3. Compute SHAP values
    explainer = shap.TreeExplainer(model)
    # Handle additivity check to prevent errors with small precision diffs
    explainer.check_additivity = False 
    shap_values = explainer.shap_values(X_transformed_df)

    # --- FIGURE 1A: Global Importance (Bar Chart) ---
    print("Generating Figure 1A (Bar Chart)...")
    plt.figure(figsize=(10, 6))
    plt.title("Figure 1A: Global Feature Importance (Mean |SHAP|)")
    # plot_type='bar' creates the ranking chart
    shap.summary_plot(shap_values, X_transformed_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('Figure1A_SHAP_Bar.png', dpi=300)
    plt.show()

    # --- FIGURE 1B: Summary Plot (Beeswarm) ---
    print("Generating Figure 1B (Beeswarm)...")
    plt.figure(figsize=(10, 6))
    plt.title("Figure 1B: Feature Impact Summary")
    shap.summary_plot(shap_values, X_transformed_df, show=False)
    plt.tight_layout()
    plt.savefig('Figure1B_SHAP_Summary.png', dpi=300)
    plt.show()

    # --- FIGURE 2: Interaction Plot (Lignin vs DM) ---
    lignin_col = 'Lignin Mean (g/100g DM)'
    dm_col = 'DM Mean (% FM)'
    
    if lignin_col in X.columns:
        print(f"Generating Figure 2: Interaction {lignin_col} colored by {dm_col}")
        
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            ind=lignin_col, 
            shap_values=shap_values, 
            features=X, 
            interaction_index=dm_col,
            show=False
        )
        plt.title(f"Figure 2: Interaction {lignin_col} vs. {dm_col}")
        plt.tight_layout()
        plt.savefig('Figure2_Lignin_DM_Interaction.png', dpi=300)
        plt.show()


def run_stability_analysis(X, y):
    """
    Generates Figure 3: Stability Analysis across 5 folds.
    Aggregates SHAP results from 5 CV folds and colors by INTERACTION VALUE.
    """
    print("Running Stability Analysis (Figure 3)...")
    
    lignin_col = 'Lignin Mean (g/100g DM)'
    dm_col = 'DM Mean (% FM)'
    
    if lignin_col not in X.columns or dm_col not in X.columns:
        print("Skipping Figure 3: Required columns not found.")
        return

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Storage for aggregated results
    all_lignin_values = []
    all_lignin_shap = []
    all_interaction_values = []

    # Iterate through folds
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"  Processing Fold {fold_idx + 1}/5...")
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Define and fit pipeline for this fold
        fold_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(**RF_PARAMS))
        ])
        fold_pipeline.fit(X_train, y_train)
        
        # Prepare test data for SHAP
        model = fold_pipeline.named_steps['model']
        preprocessor = Pipeline(steps=[
            ('imputer', fold_pipeline.named_steps['imputer']),
            ('scaler', fold_pipeline.named_steps['scaler'])
        ])
        X_test_transformed = preprocessor.transform(X_test)
        
        # Compute Interaction Values (Shape: samples x features x features)
        explainer = shap.TreeExplainer(model)
        
        # FIX: Set check_additivity as attribute instead of argument for interaction values
        explainer.check_additivity = False 
        shap_interaction_vals = explainer.shap_interaction_values(X_test_transformed)
        
        # Get column indices
        col_names = list(X.columns)
        lignin_idx = col_names.index(lignin_col)
        dm_idx = col_names.index(dm_col)
        
        # Extract specific values
        # 1. X-axis: Lignin Feature Value (Raw)
        fold_lignin_vals = X_test[lignin_col].values
        
        # 2. Y-axis: Total SHAP Value for Lignin (Sum of its row in interaction matrix)
        # shap_interaction_vals[sample, feat_idx, :] sums to the standard SHAP value for feat_idx
        fold_lignin_shap = shap_interaction_vals[:, lignin_idx, :].sum(axis=1)
        
        # 3. Color: Interaction Value of DM on Lignin
        # Specific value from the interaction matrix at [lignin_idx, dm_idx]
        fold_interaction_val = shap_interaction_vals[:, lignin_idx, dm_idx]
        
        # Append to aggregates
        all_lignin_values.extend(fold_lignin_vals)
        all_lignin_shap.extend(fold_lignin_shap)
        all_interaction_values.extend(fold_interaction_val)

    # --- Plotting Figure 3 ---
    plt.figure(figsize=(10, 7))
    
    sc = plt.scatter(
        all_lignin_values, 
        all_lignin_shap, 
        c=all_interaction_values,
        cmap='viridis', 
        alpha=0.7, 
        edgecolor='grey',
        linewidth=0.5,
        s=60
    )
    
    cbar = plt.colorbar(sc)
    cbar.set_label('SHAP Interaction Value of DM on Lignin', rotation=270, labelpad=20)
    
    plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    plt.xlabel(f"{lignin_col}")
    plt.ylabel("SHAP Value for Lignin")
    plt.title("Figure 3: Stability Analysis (Aggregated over 5 Folds)")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('Figure3_Stability_Analysis.png', dpi=300)
    plt.show()
    print("Figure 3 generated successfully.")


if __name__ == "__main__":
    # 1. Load Data
    X, y = load_and_preprocess(DATA_PATH)
    
    # 2. Train & Evaluate (Global)
    final_pipeline = evaluate_model(X, y)
    
    # 3. Generate Interpretability Plots
    run_shap_analysis(final_pipeline, X, y)   # Figures 1A, 1B & 2
    run_stability_analysis(X, y)              # Figure 3