# Reproducible BMP Explainable-ML Workflow

This repository contains the reproducible analysis workflow for the manuscript:

**An Explainable Machine Learning Framework for Hypothesis Generation in Biochemical Methane Potential Prediction**

The workflow analyzes compositional feedstock data, benchmarks regression models,
fits the final Random Forest model used for SHAP interpretation, and generates
the manuscript tables and figures.

## Repository Structure

```text
data/
  table_complete.csv
scripts/
  run_analysis.py
outputs/
  figures/
  tables/
```

## How To Run

Create an environment with Python 3.11, install the dependencies, and run:

```bash
pip install -r requirements.txt
python scripts/run_analysis.py
```

The command writes manuscript and supplementary figures to `outputs/figures/`
and the supplementary descriptive-statistics table to `outputs/tables/`.

## Dataset

The workflow uses a local copy of `table_complete.csv`, derived from the public
feedstock dataset reported by Lallement et al. (2023). The analysis computes BMP
per tonne of dry matter from the reported BMP per tonne volatile solids and the
VS/DM ratio. Samples with Dry Matter (DM) below 15 percent are excluded, giving
the manuscript working dataset of 127 samples.

Users should cite the original dataset/source publication when reusing the data.
