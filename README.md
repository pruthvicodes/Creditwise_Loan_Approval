# CreditWise: Loan Approval Prediction

A machine-learning model that predicts whether a loan application should be **approved or rejected**, using the applicant's income, credit score, debts, savings, collateral and loan details.

Manual loan screening is slow and inconsistent: good applicants get rejected and risky ones get approved. CreditWise learns from past decisions to give a fast, consistent first assessment that a loan officer can then review.

---

## Results

Models were trained on 80% of the data and evaluated on a held-out 20% (200 applicants).

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** (selected) | **86.5%** | 0.78 | 0.77 | **0.78** |
| Naive Bayes | 86.5% | 0.80 | 0.74 | 0.77 |
| K-Nearest Neighbors (k=5) | 76.0% | 0.63 | 0.52 | 0.57 |
| KNN, tuned with GridSearchCV | 74.5% | 0.60 | 0.49 | 0.54 |

Precision, recall and F1 are for the "approved" class.

**Logistic Regression** was selected: joint-best accuracy and the best F1, and it is simple to interpret. Tuning (`C` in 0.01–10, 5-fold cross-validation) confirmed the default `C = 1`.

**Strongest signal:** credit score (correlation 0.45 with approval), far ahead of every other feature.

---

## Dataset

`loan_approval_data.csv`: **1,000 applicants, 20 columns.** About 5% of values are missing in every column.

| Group | Columns |
|---|---|
| Applicant | Age, Gender, Marital_Status, Dependents, Education_Level |
| Income & work | Applicant_Income, Coapplicant_Income, Employment_Status, Employer_Category |
| Credit & finances | Credit_Score, Existing_Loans, DTI_Ratio (debt-to-income), Savings, Collateral_Value |
| Loan | Loan_Amount, Loan_Term (months), Loan_Purpose, Property_Area |
| **Target** | **Loan_Approved** (Yes / No) |

The dataset comes from a machine-learning course exercise built around a fictional lender ("SecureTrust Bank").

---

## How it works

1. **Clean:** fill missing numbers with the column mean and missing categories with the most frequent value.
2. **Explore:** approval rate, income and credit-score distributions, box plots for outliers, a correlation heatmap.
3. **Encode:** label-encode `Education_Level` and one-hot encode the other categorical columns, giving 27 features.
4. **Split and scale:** 80/20 train-test split, then `StandardScaler` fitted on the training set only.
5. **Train and compare:** Logistic Regression, KNN and Naive Bayes, plus GridSearchCV tuning for KNN and Logistic Regression.
6. **Save:** the selected model, the scaler and the feature list are saved as `.pkl` files.
7. **Predict:** `predict.py` loads them and scores a new applicant.

---

## Project structure

```
├── credit_wise.ipynb         full analysis: cleaning, EDA, models, evaluation
├── predict.py                predict approval for a new applicant
├── loan_approval_data.csv    dataset (1,000 applicants)
├── model.pkl                 trained Logistic Regression model
├── scaler.pkl                fitted StandardScaler
├── columns.pkl               the 27 feature names, in model order
└── requirements.txt
```

---

## Getting started

```bash
git clone https://github.com/pruthvicodes/Creditwise_Loan_Approval.git
cd Creditwise_Loan_Approval
pip install -r requirements.txt
```

**Predict for a new applicant:**
```bash
python predict.py
# Loan approved: YES (probability 53.4%)
```
To score your own applicant, edit the `applicant` dictionary at the bottom of `predict.py`, or import it:
```python
from predict import predict
predict({"Applicant_Income": 7390, "Credit_Score": 674, ...})   # all 18 input fields
# {'approved': True, 'probability': 0.534}
```

**Explore the analysis:** open `credit_wise.ipynb` in Jupyter or VS Code.

> `scikit-learn` is pinned to 1.6.1, the version the saved model was trained with, so the `.pkl` files load without version warnings.

---

## Limitations and next steps

- **Missing target labels:** 50 applications have no recorded decision and were filled with the most common outcome ("No"). Dropping those rows would be more correct.
- **Imputation before the split:** missing values were filled using statistics from the whole dataset. Fitting the imputer on the training set only (for example, inside a scikit-learn `Pipeline`) would avoid small test-set leakage.
- **Small dataset:** with 1,000 rows, cross-validated scores would be more reliable than a single split.
- **Next:** try tree-based models (Random Forest, XGBoost), handle class imbalance (about 30% approved), and add a simple Streamlit form for predictions.

---

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter

## Author

**Prithviraj Chavan**, B.E. Artificial Intelligence & Data Science, Pune
[LinkedIn](https://www.linkedin.com/in/prithvirajshahajichavan) · [GitHub](https://github.com/pruthvicodes)
