import numpy as np
import pandas as pd
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Literal

## Develop a base model with Pyantic        
class TransactionModel(BaseModel):
    step: int
    type: Literal['CASH-OUT', 'PAYMENT', 'TRANSFER']
    amount: float
    prevBalance: float
    newBalance: float
    isFlaggedFraud: int
    isFraud: int

    # example schema data for model
    model_config = {
        "json_schema_extra": {
            "example": {
                "step": 45,
                "type": "CASH-OUT",
                "amount": 18000.0,
                "prevBalance": 20000.0,
                "newBalance": 2000.0,
                "isFlaggedFraud": 1,
                "isFraud": 1
            }
        }
    }
    
# Response model
class FraudResponse(BaseModel):
    isFraud: bool
    confidence: float
    triggered_rules: list[str]

# Global ML model
model= None
scaler = None
training_metrics={}

def create_model():
    """ Create and train a simple fraud detection mode with XGBoost """
    global model, scaler, training_metrics
    np.random.seed(42)
    n_samples= 5000
    # Generate a synthetic dataset
    data = {
        'step': np.random.randint(1, 744, n_samples),  # simulate 1 month of hourly transactions
        'type': np.random.choice(['CASH-OUT', 'PAYMENT', 'TRANSFER'], n_samples),
        'amount': np.random.uniform(10, 10000, n_samples),
        'prevBalance': np.random.uniform(0, 20000, n_samples),
        'newBalance': np.random.uniform(0, 20000, n_samples),
        'isFlaggedFraud': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])}
    df = pd.DataFrame(data)

    # Train-test split
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    features = df.drop(columns=['isFraud']) # Exclude non-numeric and target columns
    target = df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Handle class imbalance
    neg, pos = np.bincount(y_train)
    scale = neg / pos

    # Train XGBoost model with class weighting
    model = XGBClassifier(eval_metric='logloss', scale_pos_weight=scale)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Predict probabilties and apply custom threshold
    y_proba= model.predict_proba(X_test)[:,1]
    threshold = 0.3
    y_pred = (y_proba > threshold).astype(int)

    # Evaluate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred) 
    
    training_metrics = {
        'accuracy': round(accuracy, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1': round(f1, 3),
        'threshold': threshold,
        'class_distribution': {'negative': int(neg), 'positive': int(pos)},
        'confusion_matrix': matrix.tolist()}

# Define evalution rules
def evaluate_rules(transaction: dict):
    """ Develop business rules for fraud detection"""
    triggered_rules =[]
    score=0
    rules = [
        ("High transaction amount", transaction['amount'] > 5000, 0.3),
        ("Unusual transaction time", transaction['step'] % 24 < 6, 0.2),
        ("Flagged fraud", transaction['isFlaggedFraud'] == 1, 0.4),
        ("Large balance change", abs(transaction['newBalance'] - transaction['prevBalance']) > 10000, 0.2)
    ]
    triggered_rules = [r[0] for r in rules if r[1]]
    score = sum(r[2] for r in rules if r[1])
    return {'risk_score': min(score, 1.0), 'triggered_rules': triggered_rules}


def predict_fraud(transaction: TransactionModel):
    """ Predict fraud using the trained ML model and business rules """
    global model, scaler
    if model is None or scaler is None:
        raise ValueError("Model is not trained yet. Call create_model() first.")
    
    # Convert transaction to DataFrame
    transaction_df = pd.DataFrame([transaction.model_dump()])
    transaction_df = pd.get_dummies(transaction_df, columns=['type'], drop_first=True)
    
    # Ensure all expected columns are present
    for col in ['type_PAYMENT', 'type_TRANSFER']:
        if col not in transaction_df.columns:
            transaction_df[col] = 0
    
    features = transaction_df.drop(columns=['isFraud'], errors='ignore') # Exclude non-numeric and target columns
    features = scaler.transform(features)
    
    # Predict using the ML model
    fraud_prob = model.predict_proba(features)[0][1]
    
    # Evaluate business rules
    rules_evaluation = evaluate_rules(transaction.model_dump())
    
    # Combine results
    combined_score = (fraud_prob + rules_evaluation['risk_score']) / 2
    is_fraud = combined_score > 0.4
    
    return {
        'isFraud': bool(is_fraud),
        'confidence': float(combined_score),
        'triggered_rules': list(rules_evaluation['triggered_rules'])}

# Initialize FastAPI app
app = FastAPI(title="Financial Fraud Detection API", description="API for detecting financial fraud using a pre-trained ML model", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """ Create and train the model at startup """
    create_model()

@app.get("/")
async def read_root():
    return {"api": "Fraud Detection", "version": "1.0", "endpoints": ["/health", "/predict", "/test-scenarios"]}

@app.get("/health")  
async def health():
    """ Create a model health"""
    return {
            "status": "healthy", 
            "model_loaded": model is not None,
            "metrics": training_metrics }

@app.post("/predict", response_model=FraudResponse)
async def predict(transaction: TransactionModel):
        result = predict_fraud(transaction)
        return FraudResponse(
            isFraud=result['isFraud'],
            confidence=result['confidence'],
            triggered_rules=result['triggered_rules']
        )
    
@app.get("/test-scenarios")
async def test_scenarios():
    """Run test predictions to demonstrate the API"""
    test_cases = [{"step": 100, "type": "PAYMENT", "amount": 200, "prevBalance": 2000, "newBalance": 1800, "isFlaggedFraud": 0, "isFraud": 0},
        {"step": 35, "type": "CASH-OUT", "amount": 15000, "prevBalance": 16000, "newBalance": 1000, "isFlaggedFraud": 1, "isFraud": 1}]
    results = []
    for case in test_cases:
        transaction = TransactionModel(**case)
        result = predict_fraud(transaction)
        results.append({"input": case, "prediction": result})
    return {"test_results": results}

if __name__ == "__main__":
    create_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
