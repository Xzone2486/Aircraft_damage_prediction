# ✈️ Aviation Damage Severity Prediction

A Machine Learning pipeline to predict the severity of aircraft damage based on aviation incident data. This project is designed to assist aviation safety analysts, engineers, and regulatory bodies in identifying high-risk incidents using historical patterns.

## 📌 Project Overview

In aviation, understanding the extent of damage after an incident is crucial for safety, insurance, and regulatory compliance. However, manually reviewing each case can be time-consuming.  
This project uses supervised machine learning techniques to **predict aircraft damage severity** (e.g., *Minor*, *Substantial*, *Destroyed*, *None*) using **preprocessed aviation incident data**.

Developed as part of a hackathon challenge to address safety analytics in aviation.

---

## 🧠 Problem Statement

Given structured aviation incident data, the goal is to:
- Predict the **damage severity** caused by an incident.
- Analyze **important contributing features**.
- Provide **visual analytics and model insights** for safety decision-making.

---

## 🛠️ Features

- Preprocessing of raw aviation datasets
- EDA and visualization (class distribution, feature correlation, importance)
- Support for multiple ML models:
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
  - SVM
- Hyperparameter tuning using Grid Search
- Automatic model selection and performance evaluation
- Saved trained model for inference on new data
- Detailed charts and JSON-based summary reports

---

## 🗂️ Project Structure

```
aviation-damage-prediction/
│
├── aircraft_damage_prediction_model.py   # Main training and prediction script
├── AviationData.csv                      # Raw aviation data
├── AviationData_preprocessed.csv         # Cleaned feature dataset
├── visualization_plots/                  # All saved plots
│   ├── 01_dataset_overview.png
│   ├── 02_feature_analysis.png
│   ├── ...
│
├── aircraft_damage_model.pkl             # Trained model (Joblib)
├── aircraft_damage_model_backup.pkl      # Backup model (Pickle)
├── model_summary_report.json             # Final model summary report
└── README.md
```

---

## 📊 Target Labels (Severity Classes)

- **Destroyed**
- **Substantial**
- **Minor**
- **None**
- **Unknown**

These are simplified from original dataset values for better model generalization.

---

## 🚀 How to Run

### 1. Install Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 2. Place Dataset Files

Ensure the following files are in the project directory:
- `AviationData.csv`
- `AviationData_preprocessed.csv`

### 3. Run the Training Pipeline

```bash
python aircraft_damage_prediction_model.py
```

After execution:
- Trained model is saved as `aircraft_damage_model.pkl`
- Visualizations are stored in `visualization_plots/`
- Summary report is saved as `model_summary_report.json`

---

## 🔍 Predict with Trained Model

```python
from aircraft_damage_prediction_model import predict_aircraft_damage
import joblib
import pandas as pd

# Load model
model_package = joblib.load('aircraft_damage_model.pkl')

# Load new data
new_data = pd.read_csv('your_new_preprocessed_input.csv')

# Make prediction
predictions, probabilities = predict_aircraft_damage(model_package, new_data)
print(predictions)
```

---

## 📈 Evaluation Metrics

- Accuracy
- F1 Score
- Confusion Matrix
- ROC Curves (Multiclass)
- Cross-validation scores
- Class-wise prediction distribution

---

## 💡 Key Insights

- Class imbalance was handled using `class_weight='balanced'`
- Important features were visualized using feature importance plots
- Substantial improvement was seen by model optimization using GridSearchCV

---

## 🤝 Team & Hackathon Context
This project was developed as part of a hackathon focused on Aviation Data Science Applications, emphasizing safety prediction and decision-making support in the aerospace domain.

- 👨‍💻 Built by:
--Ansh Kumar Prasad — anshprasad489@gmail.com

--Kumar Ankur — ankurkumarloh0909@gmail.com

--Tarun Jaiswal — tarunjaiswal2020@gmail.com

--Hardik Gaur — hardikgaur971@gmail.com

--Sheha Chauhan

### 👨‍💻 Built by:
- [Your Name] — Data Science & AI/ML Enthusiast, VIT Bhopal

> Add more team members if applicable

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).
