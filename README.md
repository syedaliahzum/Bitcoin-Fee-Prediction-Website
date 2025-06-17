# Bitcoin Fee Prediction Website #
This project is a machine learning-based web application that predicts Bitcoin transaction fees using transaction-specific attributes such as size, weight, number of inputs/outputs, and fee per kilobyte. It aims to help users estimate optimal fees for faster and more cost-effective confirmations.

## Overview ##
Problem: Bitcoin fees vary with network congestion, and choosing the wrong fee can lead to delays or overpayment.

Solution: A deep learning model predicts the required transaction fee using historical data.

## Machine Learning ##
Trained on over 900,000 Bitcoin transactions

### Features: ###

Transaction size

Weight

Number of inputs and outputs

Fee per KB

Data was log-transformed and scaled using StandardScaler.

### Model Performance ###
R² Score: 0.9992

Mean Absolute Error (MAE): ~21.55 sats

Accuracy within ±2%: ~98%

Web Interface (if applicable)
The optional web app allows users to:

Input transaction attributes

Instantly receive a fee prediction

Estimate potential fee savings

## Project Structure ##
transaction_data.csv – Raw dataset

new_transaction_model.keras – Trained DNN model

scaler123.pkl – Preprocessing scaler

notebooks/ – Jupyter notebooks for training and evaluation

app.py or predict.py – Script for running predictions or web interface

## Technologies Used ##
Python, TensorFlow, Keras

Pandas, NumPy, Scikit-learn

Matplotlib, Seaborn

(Optional) Streamlit or Flask for UI

## How to Run ##
Clone the repository

Install dependencies 

Load the model and scaler

Run app.py (or your prediction script)

