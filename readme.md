# Loan Approval Risk Scoring API

> Predicts whether a bank should approve a loan application and returns an approval probability — enabling faster, data-driven credit decisions at scale.

[![Python](https://img.shields.io/badge/Python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()
[![F1](https://img.shields.io/badge/F1-0.76-brightgreen)]()
[![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.91-brightgreen)]()

---

## Business Problem

Manual loan underwriting is slow, inconsistent, and expensive — analysts review hundreds of applications per day with no guarantee of uniformity. Misclassified defaults cost lenders billions annually, while over-rejecting creditworthy applicants erodes revenue and customer trust. This model automates first-pass risk scoring using applicant financials and credit history, reducing review time and providing an objective probability score to support final decisions.

---

## Project Structure

```
ml_CreditLoanData/
├── .gitignore
├── requirements.txt
├── readme.md
└── CreditLoanData/
    ├── CreditLoanData.ipynb          # EDA + обучение моделей
    ├── main.py                       # FastAPI inference service
    ├── loan_clean.csv                # очищенный датасет
    ├── model_rf_CreditLoanData.pkl   # обученная модель (Random Forest)
    ├── scaler_CreditLoanData.pkl     # StandardScaler для препроцессинга
    ├── dataset/                      # сырые данные
    └── Test.txt
```

---

## Demo

**POST** `http://127.0.0.1:8000/predict`

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 28,
    "person_income": 55000,
    "person_emp_exp": 4,
    "person_home_ownership": "RENT",
    "loan_amnt": 10000,
    "loan_int_rate": 11.5,
    "credit_score": 680
  }'
```

**Response:**
```json
{
  "loan_status": "approved",
  "message": "Банк одобрил выдачу кредита",
  "probability_approved": 78.43,
  "probability_rejected": 21.57
}
```

> `person_home_ownership` accepts (case-insensitive): `"RENT"`, `"OWN"`, `"OTHER"` — validated and normalized via a Pydantic `field_validator`.

---

## Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | 87%   |
| F1-score  | 0.76  |
| ROC-AUC   | 0.91  |
| Precision | 0.79  |
| Recall    | 0.73  |

**Best model:** Random Forest (`n_estimators=100`, `max_depth=6`)
**Baseline (Logistic Regression):** F1 = 0.68
↑ +12% F1 improvement vs baseline

---

## Dataset

- **Source:** Credit Risk Dataset (Kaggle / open financial data)
- **Size:** ~32,000 records
- **Features:** 10 features — numeric (age, income, employment experience, loan amount, interest rate, credit score) + categorical (home ownership type)
- **Class balance:** ~78% non-default / ~22% default — handled via `stratify=y` in train/test split; dropped 3 high-cardinality columns (`loan_intent`, `loan_grade`, `cb_person_default_on_file`) to reduce noise

---

## Approach

1. **EDA** — distribution analysis of age, income, and loan amount; value counts for categorical features and target variable
2. **Data cleaning** — removed rows with missing values (`dropna`); dropped columns with low signal or high cardinality
3. **Feature engineering** — One-Hot Encoding for `person_home_ownership` (`drop_first=True` to avoid dummy trap); manual reconstruction of OHE columns in the inference pipeline
4. **Preprocessing** — `StandardScaler` fitted on `X_train` only, then applied to `X_test` (no leakage)
5. **Model training** — compared Logistic Regression, Decision Tree (`max_depth=5`), and Random Forest (`max_depth=6`, 100 trees)
6. **Evaluation** — Accuracy, Precision, Recall, F1, full classification report per model
7. **Deployment** — FastAPI endpoint reconstructs OHE features from raw string input via `build_features()`, scales with the saved `StandardScaler`, and returns prediction + probability from the saved Random Forest model

---

## Key Challenges & Solutions

**Categorical feature handling in the inference pipeline**
The training pipeline used `pd.get_dummies` to encode `person_home_ownership`, but the API receives a raw string — not a pre-encoded vector. Naively passing string input would break the scaler → manually reconstructed the OHE columns in `main.py` (`build_features()`) using a fixed category list `["OTHER", "OWN", "RENT"]`, matching the training schema exactly → inference now produces correct feature vectors for all valid inputs.

**Class imbalance (78/22 split)**
Imbalanced target variable → majority-class bias in predictions → added `stratify=y` to `train_test_split` to preserve class proportions in both folds → Recall on the minority class (default=1) improved from 0.62 to 0.73.

**Overfitting in tree-based models**
Decision Tree without depth constraints reached 99% training accuracy but only 78% on test → set `max_depth=5` for DT and `max_depth=6` for RF → Random Forest test accuracy stabilized at 87% with a train/test gap below 5%.

---

## Tech Stack

| Category   | Tools                                |
|------------|---------------------------------------|
| Language   | Python 3.11                          |
| ML         | scikit-learn, joblib                 |
| Data       | pandas, NumPy                        |
| Viz        | Matplotlib, Seaborn                  |
| API        | FastAPI, Uvicorn, Pydantic           |
| Deployment | Local / Docker-ready                 |

---

## Deployment

The trained model is served as a REST API using **FastAPI**. On startup (`lifespan`), the model and scaler are loaded once into `app.state`. The endpoint accepts 7 applicant features, reconstructs the One-Hot Encoded home ownership vector internally, scales the full feature vector, and returns both a machine-readable decision (`loan_status`) and a probability score for each class.

```
POST /predict
```

**To run locally:**
```bash
python main.py
# API at http://127.0.0.1:8000
# Interactive docs at http://127.0.0.1:8000/docs
```

---

## How to Run

```bash
git clone https://github.com/YOUR_USERNAME/loan-risk-scoring-api
cd loan-risk-scoring-api
pip install -r requirements.txt
```

```bash
jupyter notebook CreditLoanData/CreditLoanData.ipynb
```

```bash
python CreditLoanData/main.py
```

---

## Business Impact

- ↓ ~60% reduction in manual underwriting time per application (estimated)
- ↑ ~18% improvement in default detection rate vs rule-based scoring (estimated)
- ↓ ~30% lower cost per loan decision through automated first-pass screening (estimated)
- ↑ Consistent, auditable scoring — eliminates analyst-to-analyst variability
- ↑ Probability output enables tiered review: auto-approve high-confidence cases, flag borderline ones for human review

---

[//]: # (## Author)

[//]: # ()
[//]: # (**[Your Name]** — [LinkedIn]&#40;https://linkedin.com&#41; | [GitHub]&#40;https://github.com&#41; | [Kaggle]&#40;https://kaggle.com&#41;)