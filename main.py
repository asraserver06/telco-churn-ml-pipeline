import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def generate_synthetic_telco_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'tenure': np.random.randint(1, 72, num_samples),
        'MonthlyCharges': np.random.uniform(20, 120, num_samples),
        'TotalCharges': np.random.uniform(20, 8000, num_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], num_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], num_samples),
        'Churn': np.random.choice(['Yes', 'No'], num_samples, p=[0.26, 0.74])
    }
    df = pd.DataFrame(data)
    # Convert target to binary
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def main():
    print("Generating synthetic Telco Churn dataset...")
    df = generate_synthetic_telco_data()
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['Contract', 'InternetService', 'PaymentMethod']
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Hyperparameter tuning
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10, 20]
    }
    
    print("Training and tuning the model...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    
    # Evaluation
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Export the pipeline
    model_path = 'churn_pipeline.pkl'
    joblib.dump(best_model, model_path)
    print(f"Pipeline exported to {model_path}")

if __name__ == "__main__":
    main()
