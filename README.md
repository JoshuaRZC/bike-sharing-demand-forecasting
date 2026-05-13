# Bike-Sharing Demand Forecasting

This repository contains the materials for a STAT 248 final project on hourly bike-sharing demand in Washington, DC. The project studies the temporal structure of Capital Bikeshare rental counts and compares several short-term forecasting approaches using historical demand, calendar information, and weather variables.

## Project Overview

Bike-sharing systems have strong within-day and within-week usage patterns. For a station-based system such as Capital Bikeshare, these hourly swings matter because inaccurate demand forecasts can translate into poor bike availability and harder system rebalancing.

The analysis focuses on the hourly rental count series, `cnt`, from the Bike Sharing Dataset. The current working pipeline rebuilds a complete hourly time index, examines daily and weekly dependence, and compares a classical ARIMA benchmark with lagged regression, LSTM, RNN, and an ARIMAX-style extension.

## Research Question

What temporal structure characterizes hourly bike-sharing demand, and how well can short-term demand be forecast using historical usage together with observed calendar and weather conditions?

## Repository Structure

```text
bike-sharing-demand-forecasting/
├── code/
│   ├── bike_sharing_demand_forecasting.ipynb
│   ├── data.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── arima.py
│   ├── lagged_regression.py
│   ├── rnn_lstm.py
│   ├── diagnostic.py
│   └── analysis.py
├── data/
│   ├── hour.csv
│   ├── day.csv
│   └── Readme.txt
├── figures/
│   └── hourly_demand_structure.png
├── instruction/
│   ├── Code Guidelines.md
│   ├── Presentation Guidelines.md
│   └── Stat 248 Final Project Guidelines.pdf
├── proposal/
│   ├── proposal.tex
│   ├── proposal.pdf
│   └── proposal.bib
├── report/
│   ├── report.tex
│   ├── report.pdf
│   └── report.bib
├── environment.yaml
└── run.sh
```

## Data

The dataset used in this project is the Bike Sharing Dataset, available on Kaggle:

<https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset>

The data contain Capital Bikeshare rental counts from Washington, DC for 2011 and 2012, along with weather and calendar variables. This project uses the hourly file, `hour.csv`, which has 17,379 observed hourly records from January 1, 2011 through December 31, 2012.

The raw hourly file is not a perfectly complete hourly series: 165 hourly timestamps are missing. The analysis reconstructs the full hourly grid, interpolates continuous weather and demand variables for missing hours, forward/back-fills weather category when needed, and rebuilds calendar variables from the timestamp. The variables `casual` and `registered` are not used as predictors because they sum directly to the response variable, `cnt`.

## Analysis Workflow

The current analysis is organized around four stages:

1. Data preparation, including timestamp construction, missing-hour reconstruction, imputation, and basic summaries.
2. Temporal structure analysis, including hourly profiles, working-day versus non-working-day comparisons, ACF/PACF checks, and first/seasonal differencing.
3. Forecasting models, including ARIMA on a 24-hour differenced series, lagged regression with weather and calendar predictors, LSTM window comparison, RNN benchmark, and an ARIMAX-style extension.
4. Model comparison and diagnostics, including validation/test MAE and RMSE, forecast plots, residual checks, and interpretation of which methods capture the main time dependence.

The main notebook, `code/bike_sharing_demand_forecasting.ipynb`, is the clean entry point for the full analysis.

## Current Results

The first complete pipeline suggests that most predictive power comes from recent demand and daily/weekly repetition. A 24-hour differenced ARIMA benchmark improves on a naive treatment of the raw hourly counts, but explicit lag features perform better. In the current run, lagged regression has the best test RMSE among the main stable models, while the LSTM with a 72-hour window has the best validation RMSE but slightly weaker test RMSE.

The ARIMAX-style extension is included as a comparison, but in the current version it underperforms the simpler lagged regression model and shows a convergence warning. This makes it useful as a cautionary benchmark rather than the main result.

## Reproducibility

The project environment is defined in `environment.yaml`. A conda environment can be created with:

```bash
conda env create -f environment.yaml
conda activate bike-sharing-demand-forecasting
```

The main analysis notebook can be executed in place with:

```bash
bash run.sh
```

The script runs:

```bash
jupyter nbconvert --to notebook --execute --inplace code/bike_sharing_demand_forecasting.ipynb
```

## Outputs

The main written materials are:

- `proposal/proposal.tex`
- `proposal/proposal.pdf`
- `report/report.tex`
- `report/report.pdf`

Figures used in the written documents and presentation are stored in `figures/`. The current main exploratory figure is `figures/hourly_demand_structure.png`.

## License

This repository is distributed under the terms of the MIT License. See `LICENSE` for details.

Use of the Bike Sharing Dataset should cite the original dataset paper listed in `data/Readme.txt`.
