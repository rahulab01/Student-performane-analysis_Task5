# Student-performane-analysis_Task5
Data analysis using python- internship task 5
# 🚢 Titanic ML Pipeline

**Data Science with Python Internship | Maincrafts Technology — Task 5**

A complete, professional machine learning pipeline built on the classic Titanic dataset. The project demonstrates a real-world ML workflow: data cleaning, preprocessing, model training, evaluation, and persistence.

---

## 📋 Objective

Build an end-to-end supervised ML pipeline that:

- Cleans and preprocesses raw data (missing value imputation, categorical encoding, numeric scaling)
- Trains a **Logistic Regression** classifier as an interpretable baseline
- Evaluates performance using accuracy, precision/recall/F1, confusion matrix, and ROC-AUC
- Validates results with **5-fold cross-validation** for reliable generalisation estimates
- Persists the trained pipeline to `model.joblib` for reuse

---

## 📁 Project Structure

```
.
├── internship_task5.ipynb   # Main notebook with full pipeline
├── Titanic-Dataset.csv      # Dataset (891 passengers, 12 columns)
├── model.joblib             # Saved trained pipeline (generated on run)
└── images/
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── cross_validation.png
    └── feature_importance.png
```

---

## 🗂️ Dataset

The standard Kaggle Titanic dataset with **891 rows and 12 columns**.

**Target variable:** `Survived` (0 = did not survive, 1 = survived)

### Features Used

| Feature | Type | Rationale |
|---|---|---|
| `Pclass` | Categorical | Proxy for socioeconomic status and cabin deck proximity to lifeboats |
| `Sex` | Categorical | The single most predictive feature — "women and children first" |
| `Age` | Numeric | Children were prioritised; ~20% missing (median imputed) |
| `SibSp` | Numeric | Number of siblings/spouses aboard |
| `Parch` | Numeric | Number of parents/children aboard |
| `Fare` | Numeric | Correlated with class and cabin location |
| `Embarked` | Categorical | Port of embarkation: S = Southampton, C = Cherbourg, Q = Queenstown |

**Excluded:** `PassengerId`, `Name`, `Ticket`, `Cabin` (77% missing)

### Missing Value Strategy

| Column | Missing | Strategy |
|---|---|---|
| `Age` | ~20% (177 rows) | Median imputation (robust to outliers, fitted on train only) |
| `Cabin` | ~77% (687 rows) | Dropped entirely |
| `Embarked` | 2 rows | Most frequent value ('S') |

---

## ⚙️ Pipeline Architecture

Built using a `sklearn.pipeline.Pipeline` wrapping a `ColumnTransformer` to prevent data leakage — all preprocessing statistics are fitted only on training data.

```
Pipeline
├── Preprocessor (ColumnTransformer)
│   ├── Numeric Pipeline  → SimpleImputer(median) → StandardScaler
│   └── Categorical Pipeline → SimpleImputer(most_frequent) → OneHotEncoder
└── Classifier → LogisticRegression(max_iter=1000)
```

**Train/Test Split:** 80/20, stratified on the target (`random_state=42`)

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | ~80.8% |
| Test ROC-AUC | ~0.860 |
| CV ROC-AUC (5-fold) | ~0.858 ± 0.022 |
| CV Accuracy (5-fold) | ~0.800 ± 0.025 |

Low standard deviation across folds confirms the model is stable and generalises well.

---

## 🔑 Key Findings

1. **Sex is the strongest predictor** — female survival rate ~74% vs ~19% for males.
2. **Pclass is the second most impactful feature** — 1st-class passengers had deck and evacuation priority; 3rd-class survival rate was only ~24%.
3. **Fare adds marginal signal** beyond Pclass (correlated with cabin location).
4. **Age has a moderate effect** — children were prioritised; the 20% missingness limits its signal.
5. **The sklearn Pipeline eliminates data leakage** — the professional standard for reproducible ML.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Running the Notebook

```bash
jupyter notebook internship_task5.ipynb
```

Ensure `Titanic-Dataset.csv` is in the same directory as the notebook before running.

### Loading the Saved Model

```python
import joblib
import pandas as pd

model = joblib.load('model.joblib')

sample = pd.DataFrame([{
    'Pclass': '1', 'Sex': 'female', 'Age': 29,
    'SibSp': 0, 'Parch': 0, 'Fare': 100.0, 'Embarked': 'S'
}])

print(model.predict(sample))          # [1]
print(model.predict_proba(sample))    # [[prob_0, prob_1]]
```

---

## 🔭 Recommended Next Steps

- **Feature engineering:** `FamilySize = SibSp + Parch`, `IsAlone` flag, title extraction from `Name` (Mr, Mrs, Miss, Master)
- **Advanced models:** Compare with Random Forest and Gradient Boosting
- **Hyperparameter tuning:** Optimise regularisation `C` with `GridSearchCV`
- **Class imbalance:** Try `class_weight='balanced'` to improve recall on survivors
- **Cabin:** Extract deck letter from the ~200 non-null `Cabin` rows

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas / numpy** — data manipulation
- **scikit-learn** — ML pipeline, modelling, evaluation
- **matplotlib / seaborn** — visualisation
- **joblib** — model serialisation

---

## 📜 License

This project was developed as part of the Maincrafts Technology Data Science Internship (Task 5).
## Author
Rahul Prajapat
