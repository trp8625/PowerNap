# PowerNap
# PowerNap: Carbon-Aware Grid Intensity Forecasting

**NYU | Tanvi Patel, Alexandra Lugo, Alex Muzila, Benny Yuan, Srivar Janna**

## Overview

PowerNap forecasts grid carbon intensity at 1, 6, and 24 hour horizons across four U.S. regions. The goal is to predict when the electrical grid is cleanest so that flexible compute jobs — like data center workloads — can be shifted to lower-carbon windows. We train and compare three model classes (Ridge Regression, Random Forest, and LSTM) and evaluate their ability to generalize across both time and geography.

## Repository Structure

```
PowerNap/
├── 01_data_engineering.ipynb         # Pull and merge ElectricityMaps + Open-Meteo data
├── 02_feature_engineering.ipynb      # Lag features, cyclical time encodings, save featured dataset
├── 03_model_training.ipynb           # Ridge and Random Forest training + evaluation
├── 04_lstm_regional.ipynb            # LSTM trained per region (chronological split)
├── 04_lstm_full.ipynb                # LSTM trained globally (leave-one-region-out split)
├── 05_evaluation.ipynb               # Cross-model comparison, plots, metrics
├── full_dataset.csv                  # Raw merged dataset (35,040 rows, 4 regions)
├── full_dataset_featured.csv         # Featured dataset with lags + time encodings
├── model_predictions/                # Saved prediction CSVs from Ridge/RF/LSTM
└── best_lstm_*.pt                    # Saved LSTM model weights
```

## Data

| Source | What it provides |
|---|---|
| ElectricityMaps | Hourly carbon intensity (gCO2eq/kWh) for CA, TX, IL, VA |
| Open-Meteo | Hourly weather (temperature, wind speed, cloud cover, solar radiation) |

- 35,040 rows covering the full year 2023
- 4 regions: California (CAISO), Texas (ERCOT), Illinois (PJM), Virginia (PJM)
- Zero null values after preprocessing

## Features

- **Lag features** — carbon intensity at 1, 2, 3, and 24 hours prior, created per region to prevent data leakage across regions
- **Cyclical time encodings** — hour, day of week, and month encoded as sin/cos pairs to capture cyclical patterns (e.g. hour 23 is close to hour 0)
- **Weather variables** — temperature, wind speed, cloud cover, solar radiation

## Models

| Model | Notes |
|---|---|
| Ridge Regression | Linear baseline with StandardScaler |
| Random Forest | Non-linear ensemble, 50 trees, max depth 8 |
| LSTM (Regional) | Per-region sequential model, 24 timestep input window |
| LSTM (Global) | Trained across all regions for leave-one-region-out evaluation |

## Evaluation Strategy

Two generalization tests are used for all models:

**Chronological Time Split** — Train on the first 70% of 2023, validate on the next 10%, test on the final 20%. Tests whether models generalize to future time periods they have never seen.

**Leave-One-Region-Out** — Train on 3 regions, test on the held-out 4th region. Rotate through all 4 regions. Tests whether models generalize to a completely unseen geographic area.

Evaluation metrics: MAE, RMSE, R², Skill Score

## How to Run

1. Run notebooks in order: `01` → `02` → `03` → `04` → `05`
2. Each notebook reads from the previous notebook's output CSV
3. `05_evaluation.ipynb` loads prediction CSVs from `model_predictions/` and produces all final plots and metrics

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
torch
```
