# Fraud Detection API

A lightweight real-time fraud detection service combining machine learning with business rules for financial transaction screening.

## Overview

This API demonstrates key MLOps capabilities:
* Real-time inference via FastAPI
* Hybrid ML + Rules approach for comprehensive fraud detection
* Comprehensive monitoring with health checks and performance metrics
* Production-ready error handling and validation

## Technical Architecture

**Machine Learning Model:**
- XGBoost classifier trained on transaction patterns
- Features: transaction timing, amounts, balances, and system flags
- Handles class imbalance with weighted training

**Business Rules Engine:**
- High-value transaction detection
- Unusual timing pattern identification (night transactions)
- System flag correlation
- Balance manipulation detection

**API Framework:**
- FastAPI with automatic OpenAPI documentation
- Pydantic data validation
- RESTful endpoints with proper error handling

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run API Server
```bash
python main.py
```

### 3. Test Endpoints
**API Documentation:** http://localhost:8000/docs

**Health Check:**
http://localhost:8000/health


**Test Prediction:**
- Go to http://localhost:8000/docs
- Find the /predict POST endpoint
- Click "Try it out" then "Execute"
- Use the pre-filled example or test with this fraud case:

```bash
     -H "Content-Type: application/json" \
     -d '{
       "step": 45,
       "type": "CASH-OUT", 
       "amount": 18000.0,
       "prevBalance": 20000.0,
       "newBalance": 2000.0,
       "isFraud": 1,
       "isFlaggedFraud": 1
     }'
```

**Built-in Test Cases:**
Visit http://localhost:8000/test-scenarios to see example predictions.

## API Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/` | GET | API information |
| `/health` | GET | Service health status |
| `/predict` | POST | Fraud prediction for transactions |
| `/test-scenarios` | GET | Built-in test demonstrations |

## Example Response

```json
{
  "isFraud": true,
  "confidence": 0.673,
  "triggered_rules": [
    "High transaction amount",
    "Unusual transaction time",
    "Flagged fraud"
  ]
}
```

## Model Performance

The system combines two detection approaches:
- **ML Model Score:** XGBoost probability output  
- **Business Rules Score:** Weighted rule violations
- **Final Decision:** Combined score with 0.4 threshold

Performance metrics are available via the `/health` endpoint and include:
- **Precision:** Ratio of true fraud among flagged transactions
- **Recall:** Percentage of actual fraud cases detected
- **F1-Score:** Balanced measure of precision and recall
- **Accuracy:** Overall prediction correctness

## Use Cases

Designed for:
- Real-time transaction monitoring
- Batch fraud analysis
- Risk assessment integration
- Compliance and audit support

## Development

**Project Structure:**
```
├── main.py              # Complete API implementation
├── requirements.txt     # Dependencies
├── README.md           # This documentation
└── test_fraud_api.py   # External test client (optional)
```

**Dependencies:**
- FastAPI: Modern API framework
- XGBoost: Machine learning model
- Scikit-learn: Data preprocessing
- Pandas/NumPy: Data manipulation
