# 📈 Telco Churn ML Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end, production-ready machine learning pipeline for the **DevelopersHub AI/ML Engineering Internship**. This repository demonstrates how to architect a robust data processing and classification pipeline to predict customer churn using Scikit-Learn.

## 🚀 Project Overview

Customer churn is a critical metric for subscription-based businesses. This project automates the entire machine learning lifecycle—from data preprocessing to hyperparameter tuning and model export.

### Key Features
- **Scikit-Learn `Pipeline` API**: Ensures no data leakage between training and testing data by coupling preprocessing with the estimator.
- **Robust Preprocessing**: Uses `ColumnTransformer` to apply `StandardScaler` to numerical features and `OneHotEncoder` to categorical features.
- **Automated Hyperparameter Tuning**: Leverages `GridSearchCV` to automatically find the optimal depth and tree count for a `RandomForestClassifier`.
- **Production Ready**: The final tuned model is serialized using `joblib`, making it instantly ready for inference on raw data.

## 📂 Repository Structure

```text
telco-churn-ml-pipeline/
├── main.py                # Core pipeline script (data gen, preprocessing, training, tuning, export)
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
```

## ⚙️ Installation & Setup

Ensure you have Python installed. Clone the repository and install the dependencies:

```bash
git clone https://github.com/asraserver06/telco-churn-ml-pipeline.git
cd telco-churn-ml-pipeline
pip install -r requirements.txt
```

## 💻 Usage Instructions

To execute the entire pipeline from start to finish, run:

```bash
python main.py
```

**What happens during execution?**
1. Generates a synthetic but realistic Telco Churn dataset.
2. Splits data into training and test sets.
3. Constructs and tunes the Scikit-Learn pipeline.
4. Outputs the `Accuracy` and `Classification Report` to the terminal.
5. Saves the final pipeline as `churn_pipeline.pkl`.

## 🤝 Acknowledgments
Completed as part of the **DevelopersHub Corporation AI/ML Engineering Advanced Internship Tasks** (Task 2).
