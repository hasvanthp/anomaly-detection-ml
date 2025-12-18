# anomaly-detection-ml
Machine learning–based credit card fraud detection system using XGBoost with a Streamlit dashboard.

This project is a **machine learning–based fraud detection system** developed as a **self project for my ML course**.  
The goal of this project is to understand how real-world fraud detection works using supervised learning, proper data preprocessing, and a simple interactive interface.

The system uses an **XGBoost classifier** for prediction and a **Streamlit web app** to test transactions both individually and in batch.

---

## Project Overview

Fraud detection is a common real-world machine learning problem where the data is highly imbalanced and errors can be costly.  
In this project, I focused on:

- Cleaning and preparing transaction data  
- Training a reliable ML model  
- Saving and reusing the trained model  
- Building a simple UI to test predictions  

The project is kept modular so that each step (data processing, training, inference, visualization) is easy to understand.

---

## Tech Stack

- **Python**
- **XGBoost**
- **Scikit-learn**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **Streamlit**
- **Joblib**

---

## Project Structure

fraud_detection/
│
├── app.py # Streamlit application
├── requirements.txt # Required Python packages
│
├── data/
│ └── Fraud.csv # Dataset (not included in repo)
│
├── models/
│ ├── xgboost_fraud_model.pkl
│ └── scaler.pkl
│
├── src/
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── model_inference.py
│ └── visualization.py
│
└── README.md

   

---

## Dataset Information

- Numerical transaction features (`V1` to `V28`)
- `Time` – time of transaction
- `Amount` – transaction value
- `Class` – target label  
  - `0` → Legitimate  
  - `1` → Fraud  

The dataset is anonymized and is **not uploaded** to this repository.

---

## Setup & Execution (PyCharm + Virtual Environment)

### 1. Clone the Repository
```  
git clone https://github.com/your-username/fraud-detection-ml-system.git
cd fraud-detection-ml-system
2. Create a Virtual Environment
  
 
python -m venv .venv
Activate it:

Windows
.venv\Scripts\activate

Mac / Linux
source .venv/bin/activate
PyCharm usually auto-detects the virtual environment.

3. Install Required Packages
pip install --upgrade pip
pip install -r requirements.txt

4. Train the Model (Run Once) 
python src/model_training.py
This trains the XGBoost model and saves the model and scaler inside the models/ folder.

5. Run the Streamlit Application 
streamlit run app.py
The application will open in the browser at:

arduino 
http://localhost:8501
Accessing the Application
When the app is running, Streamlit shows two URLs:

Local URL
Used on the same machine running the app.

Network URL
Can be opened on another device connected to the same Wi-Fi network.
This is useful for live demos and presentations.

Application Features
Manual transaction input with fraud probability

Batch CSV upload for multiple predictions

Automatic handling of missing or mismatched columns

Simple visual summary of predictions

Learning Outcomes
Through this project, I gained hands-on experience with:

Handling imbalanced datasets

Feature scaling and preprocessing

Training and evaluating ML models

Model persistence and reuse

Building a basic ML-powered web application

Future Improvements
Add model explainability (SHAP)

Try ensemble or anomaly detection models

Deploy the app using Docker or cloud services

Improve UI and performance for large datasets

Author
Hasvanth Reddy Ponnapureddy
CSE Student from Mohan Babu University
Machine Learning Self Project
