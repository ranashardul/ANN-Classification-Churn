# 🧠 Artificial Neural Network (ANN) Learning Project

This project is a hands-on implementation of Artificial Neural Networks (ANN) using TensorFlow and Keras to understand:

- Binary Classification (Customer Churn Prediction)
- Regression (Salary Prediction)
- Hyperparameter Tuning
- Model Deployment using Streamlit
- TensorBoard Visualization

> ⚠️ This is a learning-focused project built to understand ANN fundamentals, not a production-grade ML system.

---

## 📌 Project Overview

The project uses the **Churn_Modelling.csv** dataset to:

1. Predict whether a customer will churn (Binary Classification).
2. Predict a customer's estimated salary (Regression).
3. Experiment with different ANN architectures.
4. Visualize training using TensorBoard.
5. Deploy a simple inference app using Streamlit.

---

## 🗂️ Project Structure

```
3 - ANN Project/
│
├── app.py                         # Streamlit deployment app
├── experiments.ipynb              # Classification ANN
├── hyper_parameter_tuning_ann.ipynb
├── salary_regression.ipynb        # Regression ANN
├── Churn_Modelling.csv
├── model.h5                       # Trained classification model
├── regression_model.h5            # Trained regression model
├── scaler.pkl
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── requirements.txt
└── log/                           # TensorBoard logs
```

---

# 📊 1️⃣ Customer Churn Prediction (Binary Classification)

## Problem Statement
Predict whether a customer will exit the bank.

## Preprocessing Steps

- Dropped irrelevant columns:
  - RowNumber
  - CustomerId
  - Surname
- Label Encoded: `Gender`
- One-Hot Encoded: `Geography`
- Standard Scaled numerical features
- Train-Test Split (80/20)

---

## ANN Architecture

```python
Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

- Optimizer: Adam
- Loss: Binary Crossentropy
- Metric: Accuracy
- EarlyStopping enabled
- TensorBoard enabled

---

# 📈 2️⃣ Hyperparameter Tuning

Used:

- `GridSearchCV`
- `scikeras.wrappers.KerasClassifier`

Parameters tuned:
- Number of hidden layers
- Number of neurons
- Epochs

This was done to understand how model complexity affects performance.

---

# 📉 3️⃣ Salary Prediction (Regression)

## Objective
Predict `EstimatedSalary` using ANN.

## Model Architecture

```python
Sequential([
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # Linear activation for regression
])
```

- Optimizer: Adam
- Loss: Mean Absolute Error (MAE)
- EarlyStopping enabled
- TensorBoard logging enabled

---

# 📊 TensorBoard Visualization

Training logs are stored in:

```
log/fit/
regression_logs/fit/
```

To launch TensorBoard:

```
tensorboard --logdir log/fit
```

or inside Jupyter:

```
%load_ext tensorboard
%tensorboard --logdir log/fit
```

---

# 🚀 Streamlit Deployment

A simple interactive app is built to predict customer churn.

To run:

```
streamlit run app.py
```

The app:
- Accepts user inputs
- Applies saved encoders and scaler
- Loads trained model
- Outputs churn probability

---

# 🧰 Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- TensorBoard
- Scikeras

---

# 📚 Key Learnings

Through this project, I understood:

- How feedforward neural networks work
- Role of activation functions (ReLU, Sigmoid)
- Difference between classification and regression losses
- Importance of feature scaling
- Early stopping to prevent overfitting
- Hyperparameter tuning using GridSearch
- How to deploy an ML model using Streamlit
- Maintaining consistency between training and inference pipelines

---

# ⚠️ Limitations

- No advanced regularization (Dropout, L2)
- Limited hyperparameter search space
- Basic evaluation metrics
- Not optimized for production use

---

# 🏁 Conclusion

This project helped build foundational understanding of:

- ANN architecture design
- Model training workflow
- Hyperparameter tuning
- ML model deployment

It serves as a stepping stone toward more advanced deep learning projects.

---

## 👨‍💻 Author

Shardul Rana
