#!/bin/sh

export CONDA_ALWAYS_YES="true"

echo "Installing plotly..."
conda install -c plotly plotly > /dev/null

echo "Installing seaborn..."
conda install seaborn > /dev/null

echo "Installing xgboost..."
conda install -c conda-forge xgboost > /dev/null

unset CONDA_ALWAYS_YES 
echo "\nFinished"