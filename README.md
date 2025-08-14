# Movie Review Sentiment Classification & Rating Prediction

## Overview
This project focuses on building deep learning models to:
1. **Classify** movie reviews into positive or negative sentiment (binary classification).
2. **Predict** numerical ratings of movie reviews (regression).

It uses multiple RNN architectures — **SimpleRNN**, **LSTM**, and **GRU** — trained on preprocessed movie review data. The models are evaluated using accuracy for classification and MAE, RMSE, and R² for regression.

---

## Features
- **Data Preprocessing**: Tokenisation, stopword removal, and lemmatisation applied to text data.
- **Classification Models**:  
  - SimpleRNN  
  - LSTM  
  - GRU  
- **Regression Models**:  
  - LSTM  
  - GRU  
- **Hyperparameter Tuning** for optimising learning rate, batch size, and hidden units.
- **Evaluation Metrics**:
  - Classification: Accuracy
  - Regression: MAE, RMSE, R²

---

##  Technologies Used
- Python 3
- TensorFlow / Keras
- NLTK
- scikit-learn
- NumPy, Pandas
- Matplotlib, Seaborn
