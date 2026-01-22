# River_temperature_prediction
# 🌊 River Temperature Prediction using Machine Learning

This repository focuses on predicting **river water temperature** using **machine learning and deep learning models**. River temperature is a critical parameter influencing water quality, aquatic ecosystems, and climate resilience. Accurate prediction helps in environmental monitoring and sustainable water resource management.

---

## 📌 Project Overview

The objective of this project is to build and evaluate data-driven models that can predict river water temperature using historical and environmental datasets. The workflow includes:

- Data preprocessing and feature engineering  
- Training multiple ML/DL models  
- Model evaluation and comparison  
- Deployment-ready prediction pipeline  

---

## 🧠 Models Used

The project explores and compares the following models:

- **LSTM (Long Short-Term Memory)** – for time-series prediction  
- **XGBoost Regressor**  
- **LightGBM Regressor**  
- **TabNet Regressor**

Each model is evaluated using standard regression performance metrics.

---

## 📂 Repository Structure

```text
River_temperature_prediction/
│
├── training.ipynb        # Model development, EDA, and experimentation
├── train_models.py       # Script to train and save ML/DL models
├── app.py                # Application for loading model and making predictions
├── requirements.txt      # Required Python dependencies
└── README.md             # Project documentation
