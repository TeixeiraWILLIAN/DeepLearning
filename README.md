# ECO-PANN
### Estimation of Crude Oil Properties Using Artificial Neural Networks

ECO-PANN is a machine learning application designed to estimate **physicochemical properties of crude oil** using **Artificial Neural Networks (ANNs)**.

The system combines **data preprocessing, neural network prediction, and an interactive web interface** built with **Streamlit**, allowing users to estimate oil properties directly through a browser.

The goal of this project is to provide a **fast and reliable alternative to traditional laboratory characterization methods**, which are often time-consuming and costly.

---

# Live Application

Access the deployed application:

https://ecopann.streamlit.app

The web interface allows users to input oil properties and obtain predictions instantly using the trained neural network model.

---

# Project Motivation

Characterizing crude oil properties is essential for several petroleum engineering operations, including:

- Flow assurance
- Pipeline transport
- Reservoir simulation
- Production system design

However, laboratory analyses may require significant **time, cost, and sample volume**.

Artificial Neural Networks provide an alternative approach capable of **learning nonlinear relationships between oil properties** and predicting them rapidly once trained.

---

# Technologies Used

This project was developed using the following technologies:

- Python 3.12
- Streamlit
- TensorFlow / Keras
- Scikit-learn
- Optuna
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Openpyxl

---

# Machine Learning Model

The predictive models are based on **Feedforward Artificial Neural Networks (MLP)**.

Main characteristics:

- Fully connected dense layers
- Hyperparameter optimization using **Optuna**
- Optimization algorithm: **Tree-structured Parzen Estimator (TPE)**
- Cross-validation for model evaluation
- Data preprocessing with **RobustScaler and StandardScaler**

Evaluation metrics include:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

---

# Dataset

The dataset contains **104 crude oil samples** collected from industry sources.

Oil samples include data from companies such as:

- ExxonMobil
- TotalEnergies
- PRIO
- Equinor

Available properties include examples such as:

- Density
- Viscosity
- Wax content
- Asphaltene content
- Pour point

These properties are used as **inputs or targets** depending on the prediction task.

---

# Application Interface

The application provides a simple interface where users can:

1. Insert oil properties
2. Run the trained neural network model
3. Obtain predicted values instantly

Example workflow:

User inputs properties  
↓  
Streamlit interface  
↓  
Neural network prediction  
↓  
Estimated oil property

---

# Installation

Clone the repository:

```bash
git clone https://github.com/teixeirawILLIAN/ecopann.git
cd ecopann

Recommended Python version:

Python 3.12

Create a virtual environment (recommended):

python -m venv venv

Activate the virtual environment.

Linux / Mac:

source venv/bin/activate

Windows:

venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt
Running the Application

Run the Streamlit application:

streamlit run app.py

After running the command, open the browser at:

http://localhost:8501
