# ECOP-ANN
### Estimation of Crude Oil Properties Using Artificial Neural Networks

ECOP-ANN is a machine learning application designed to estimate **physicochemical properties of crude oil** using **Artificial Neural Networks (ANNs)**.

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
- Pour point
- Wax content
- Asphaltene content
- Viscosity at 20 °C
- Viscosity at 50 °C

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
```

Recommended Python version:

Python 3.12

Create a virtual environment (recommended):

```bash
python -m venv venv
```

Activate the virtual environment.

Linux / Mac:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

After running the command, open the browser at:

```
http://localhost:8501
```

---

# Project Structure

Typical repository structure:

```
ecopann
│
├── app.py
├── model
│   └── trained_model.h5
│
├── data
│
├── src
│   ├── preprocessing.py
│   ├── prediction.py
│
├── notebooks
│
├── requirements.txt
├── README.md
```

Description:

| Folder | Description |
|------|-------------|
| `app.py` | Streamlit application interface |
| `model/` | Trained neural network models |
| `data/` | Dataset files |
| `src/` | Data preprocessing and prediction scripts |
| `notebooks/` | Development notebooks |
| `requirements.txt` | Project dependencies |

---

# Research Context

This project was developed within the **Geoenergia Lab** at **Santa Catarina State University (UDESC)**.

Research focus includes:

- Machine Learning applied to Petroleum Engineering
- Prediction of crude oil physicochemical properties
- Data-driven modeling of petroleum fluids

---

# Future Improvements

Possible future developments include:

- Expanding the dataset with additional crude oil samples
- Implementing deeper neural network architectures
- Adding uncertainty quantification
- Integrating additional petroleum fluid properties
- Deploying an API for industrial integration

---

# License

This project is intended for **research and educational purposes**.
