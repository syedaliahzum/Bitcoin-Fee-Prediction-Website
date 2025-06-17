🚀 Bitcoin Fee Prediction Website
This project presents a machine learning-powered web application that accurately predicts Bitcoin transaction fees based on transaction attributes like size, weight, input/output count, and fee rate per kilobyte. The goal is to help users optimize transaction costs by estimating appropriate fees in real-time.

🔍 Overview
Problem: Bitcoin transaction fees can fluctuate significantly, and setting an inappropriate fee can result in delayed confirmations or unnecessary cost.

Solution: This tool uses a trained deep neural network (DNN) model to predict the expected transaction fee using historical blockchain data.

🧠 Machine Learning
A deep neural network was trained using over 900,000 real Bitcoin transactions.

Features used:

Transaction size

Weight

Number of inputs/outputs

Fee per KB

All data was preprocessed with log transformation and standardized using StandardScaler.

📊 Performance
R² Score: 0.9992

MAE: ~21.55 sats

Accuracy (±2%): ~98%

🌐 Web Interface (Optional if applicable)
If hosted as a web application (e.g., via Streamlit or Flask), users can:

Input transaction details

Get predicted fee instantly

Estimate potential fee savings

📂 Files & Structure
transaction_data.csv – Raw data

new_transaction_model.keras – Trained DNN model

scaler123.pkl – Standard scaler used for preprocessing

notebooks/ – All model training and evaluation steps in Jupyter Notebooks

predict.py or app.py – Script to run the prediction engine or web app (if implemented)

📈 Technologies Used
Python, TensorFlow, Keras

Pandas, NumPy, Scikit-learn

Matplotlib, Seaborn (for analysis)

(Optional) Streamlit or Flask for frontend

✅ How to Run
Clone the repo

Install required packages from requirements.txt

Load model and scaler

Run the prediction script or web interface

