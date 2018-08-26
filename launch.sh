#!/usr/bin/env bash
echo "Starting Data Stream ..."
python Continuous_Stream_Data.py&

echo "Starting Sentiment Stream ..."
python Continuous_Stream_Sentiment.py&

echo "Training and Prediction"
python engine.py
