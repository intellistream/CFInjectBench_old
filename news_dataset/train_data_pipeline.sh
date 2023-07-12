#!/bin/bash

set -e

echo "Step 1: Download Raw Dataset"
if [ ! -f "data.csv" ]; then
  gdown --id 1C0p5R4UThJQiVHSrYmjIvapHVE3oXzjA
  unzip data.zip
  rm data.zip
else
  echo "Data generation already performed. Skipping this step."
fi

echo "Step 2: Data Cleaning"
if [ ! -f "processed_data_full.csv" ]; then
  python3 data_clean.py
else
  echo "Data cleaning already performed. Skipping this step."
fi

echo "Step 3: Entity Extraction"
if [ ! -f "ner_data.csv" ]; then
  python3 ner_gen.py
else
  echo "Entity extraction already performed. Skipping this step."
fi

echo "Step 4: Training Data Generation"
if [ ! -f "train_data.csv" ]; then
  python3 train_data_gen.py
else
  echo "Training data generation already performed. Skipping this step."
fi

echo "Pipeline completed successfully!"
