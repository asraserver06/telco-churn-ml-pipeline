# Task 2: End-to-End ML Pipeline with Scikit-learn Pipeline API

## Objective
Build a reusable and production-ready machine learning pipeline for predicting customer churn using the Telco Churn Dataset (simulated via synthetic data generation for demonstration).

## Methodology / Approach
1. **Data Preprocessing**: Constructed a Scikit-learn `ColumnTransformer` to apply `StandardScaler` to numerical features (tenure, MonthlyCharges, TotalCharges) and `OneHotEncoder` to categorical features (Contract, InternetService, PaymentMethod).
2. **Model Construction**: Combined the preprocessor and a `RandomForestClassifier` into a single Scikit-learn `Pipeline`.
3. **Hyperparameter Tuning**: Used `GridSearchCV` to find the optimal number of estimators and max depth for the random forest.
4. **Export**: Exported the fitted, tuned pipeline as `churn_pipeline.pkl` using `joblib` for future reuse in a production environment.

## Key Results / Observations
- The pipeline correctly handles both numeric scaling and categorical encoding, preventing data leakage between train and test sets.
- `GridSearchCV` successfully improved model performance.
- The exported pipeline can directly take in raw inference data without separate preprocessing steps.
