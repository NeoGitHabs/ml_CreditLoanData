# CreditLoanData/main.py

from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import joblib
import uvicorn
import numpy as np

BASE_DIR = Path(__file__).parent

HOME_CATEGORIES = ["OTHER", "OWN", "RENT"]  # порядок важен для OHE


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model  = joblib.load(BASE_DIR / "model_rf_CreditLoanData.pkl")
    app.state.scaler = joblib.load(BASE_DIR / "scaler_CreditLoanData.pkl")
    yield


app = FastAPI(title="Credit Loan Classifier", lifespan=lifespan)


# ── Schema ─────────────────────────────────────────────────────────────────────
class PersonSchema(BaseModel):
    person_age:           int
    person_income:        float
    person_emp_exp:       int
    person_home_ownership: str
    loan_amnt:            int
    loan_int_rate:        float
    credit_score:         int

    @field_validator("person_home_ownership")
    @classmethod
    def validate_home(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in HOME_CATEGORIES:
            raise ValueError(f"person_home_ownership должен быть одним из: {HOME_CATEGORIES}")
        return v


# ── Utils ──────────────────────────────────────────────────────────────────────
def build_features(person: PersonSchema) -> np.ndarray:
    numerical = [
        person.person_age,
        person.person_income,
        person.person_emp_exp,
        person.loan_amnt,
        person.loan_int_rate,
        person.credit_score,
    ]
    ohe = [1 if person.person_home_ownership == h else 0 for h in HOME_CATEGORIES]
    return np.array([numerical + ohe], dtype=float)


# ── Endpoint ───────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(person: PersonSchema):
    features = build_features(person)
    scaled   = app.state.scaler.transform(features)

    prediction = int(app.state.model.predict(scaled)[0])
    proba      = app.state.model.predict_proba(scaled)[0].tolist()

    return {
        "loan_status":           "approved" if prediction == 0 else "rejected",
        "message":               "Банк одобрил выдачу кредита" if prediction == 0 else "Банк отклонил выдачу кредита",
        "probability_approved":  round(proba[0] * 100, 2),
        "probability_rejected":  round(proba[1] * 100, 2),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)