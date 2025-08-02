import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                            classification_report, mean_squared_error, r2_score)
from imblearn.under_sampling import RandomUnderSampler

def load_local_data():
    """Load dataset from local file"""
    try:
        df = pd.read_csv('creditcard.csv')
        print("Dataset loaded successfully from local file!")
        print("\nOriginal class distribution:")
        print(df['Class'].value_counts())
        return df
    except FileNotFoundError:
        print("Error: 'creditcard.csv' not found in current directory")
        return None

def balance_and_split_data(df):
    """Balance classes and split into train/test sets"""
    if df is None:
        return None, None, None, None
    
    # Balance classes
    rus = RandomUnderSampler(random_state=42)
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_res, y_res = rus.fit_resample(X, y)
    
    print("\nBalanced class distribution:")
    print(pd.Series(y_res).value_counts())
    
    # Split data
    return train_test_split(
        X_res, y_res, test_size=0.3, random_state=42, stratify=y_res
    )

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate both regression models"""
    if X_train is None:
        return
        
    # Initialize models
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    lin_reg = LinearRegression()
    
    # Train models
    print("\nTraining Logistic Regression...")
    log_reg.fit(X_train, y_train)
    
    print("Training Linear Regression...")
    lin_reg.fit(X_train, y_train)
    
    # Evaluate Logistic Regression (Classification)
    y_pred_log = log_reg.predict(X_test)
    print("\nLogistic Regression Performance:")
    print(f"Precision: {precision_score(y_test, y_pred_log):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_log):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred_log):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_log))
    
    # Evaluate Linear Regression (Regression Metrics)
    y_pred_lin = lin_reg.predict(X_test)
    # Convert continuous predictions to binary (0/1) using 0.5 threshold
    y_pred_lin_class = (y_pred_lin >= 0.5).astype(int)
    
    print("\nLinear Regression Performance:")
    print("\nLinear Regression Classification Metrics (threshold=0.5):")
    print(f"Precision: {precision_score(y_test, y_pred_lin_class):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_lin_class):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred_lin_class):.4f}")

# Main execution
if __name__ == "__main__":
    # Step 1: Load local data file
    df = load_local_data()
    
    # Step 2: Balance and split data
    X_train, X_test, y_train, y_test = balance_and_split_data(df)
    
    # Step 3: Train and evaluate models
    train_and_evaluate_models(X_train, X_test, y_train, y_test)