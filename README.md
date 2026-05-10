# Hate Speech Detection in Social Media

A machine learning system to detect hate speech in tweets.

## Classes
- 0 = Hate Speech
- 1 = Offensive Language  
- 2 = Neither

## Dataset
Davidson Hate Speech Dataset — 24,783 English tweets

## Model
Calibrated LinearSVC with dual TF-IDF features (word + char level)

## Accuracy
- Overall Accuracy: 88.68%
- Macro F1: 0.71

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
