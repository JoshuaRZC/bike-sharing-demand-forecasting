#!/usr/bin/env bash

cd "$(dirname "$0")/code"

conda activate bike-sharing-demand-forecasting

jupyter nbconvert --to notebook --execute --inplace bike_sharing_demand_forecasting.ipynb
