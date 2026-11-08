"""CreditWise: predict loan approval for a new applicant using the saved model.

Usage:
    python predict.py

Loads model.pkl (Logistic Regression), scaler.pkl (StandardScaler) and columns.pkl
(the 27 feature names), rebuilds the same encoding used in credit_wise.ipynb, and
returns the approval decision with its probability.
"""
import pickle
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def _load(name):
    with open(HERE / name, "rb") as f:
        return pickle.load(f)


MODEL = _load("model.pkl")
SCALER = _load("scaler.pkl")
COLUMNS = _load("columns.pkl")

NUMERIC = ["Applicant_Income", "Coapplicant_Income", "Age", "Dependents", "Credit_Score",
           "Existing_Loans", "DTI_Ratio", "Savings", "Collateral_Value", "Loan_Amount",
           "Loan_Term"]
# One-hot encoded in the notebook with drop="first" (first category = baseline of all zeros).
CATEGORICAL = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area",
               "Gender", "Employer_Category"]
EDUCATION = {"Graduate": 0, "Not Graduate": 1}   # LabelEncoder order from the notebook


def to_features(applicant: dict) -> pd.DataFrame:
    """Turn one raw applicant record into the model's 27 feature columns."""
    row = {c: 0.0 for c in COLUMNS}
    for c in NUMERIC:
        row[c] = float(applicant[c])
    row["Education_Level"] = EDUCATION[applicant["Education_Level"]]
    for c in CATEGORICAL:
        col = f"{c}_{applicant[c]}".replace("-", "_")
        if col in row:              # baseline category -> stays all zeros
            row[col] = 1.0
    return pd.DataFrame([row], columns=COLUMNS)


def predict(applicant: dict) -> dict:
    X = SCALER.transform(to_features(applicant).values)
    proba = float(MODEL.predict_proba(X)[0, 1])
    return {"approved": proba >= 0.5, "probability": round(proba, 3)}


if __name__ == "__main__":
    # A real applicant from loan_approval_data.csv (row 2), who was approved.
    applicant = {
        "Applicant_Income": 7390, "Coapplicant_Income": 2106, "Age": 25, "Dependents": 2,
        "Credit_Score": 674, "Existing_Loans": 4, "DTI_Ratio": 0.2, "Savings": 13844,
        "Collateral_Value": 6908, "Loan_Amount": 27943, "Loan_Term": 72,
        "Employment_Status": "Salaried", "Marital_Status": "Single",
        "Loan_Purpose": "Business", "Property_Area": "Urban",
        "Education_Level": "Graduate", "Gender": "Female", "Employer_Category": "Government",
    }
    result = predict(applicant)
    print("Loan approved:", "YES" if result["approved"] else "NO",
          f"(probability {result['probability']:.1%})")
