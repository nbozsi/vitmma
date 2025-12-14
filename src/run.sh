#!/bin/bash
set -e

echo "Running 01_processing.py..."
python 01_data_processing.py

echo "Running 02_train.py..."
python 02_train.py

echo "Running 03_evaluation.py..."
python 03_evaluation.py

echo "Pipeline finished successfully."