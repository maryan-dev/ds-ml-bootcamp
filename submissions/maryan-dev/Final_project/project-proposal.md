# Final Project Proposal

## 1. Certificate Name

Maryan Mohamed Adam

## 2. Project Title and Description

**Title:** Machine Learning-Based Employment Status Prediction Using Somali Labour Force Data

Employment is one of the most pressing socioeconomic challenges in Somalia, with many young people and women facing unemployment or remaining outside the labour force entirely. This project builds a machine learning model that predicts an individual's employment status (Employed, Unemployed, or Not in Labour Force) from demographic and household characteristics. The tool is intended for use by government agencies, NGOs, and development organizations who need fast, data-driven insight into labour market patterns to design and target employment interventions, rather than waiting on slow manual survey analysis.

## 3. Problem Type

**Classification** — the target column, `employment_status`, has three categories (Employed, Unemployed, Not in Labour Force), so this is a multiclass classification problem.

## 4. Dataset

- **Source:** `lfs_somalia_synthetic_2000.csv` — a synthetic Somali labour force dataset modeled on the structure of the Somali Labour Force Survey (SLFS 2019, Somalia National Bureau of Statistics / ILO: https://microdata.nbs.gov.so/index.php/catalog/57).
- **Size:** 2,000 rows, 10 columns — meets the 1,000-row minimum.
- **Target column:** `employment_status` — categorical, values are `Employed` (1,104 rows), `Unemployed` (541 rows), `Not in Labour Force` (355 rows). Classes are imbalanced, which will be addressed in the final project (class weighting / stratified sampling).
- **Main features:**
  - `age` — respondent's age in years
  - `gender` — Male / Female
  - `region` — one of 7 Somali regions (e.g., Banadir, Puntland, Somaliland)
  - `education` — Primary, Secondary, Tertiary, Vocational, or missing (~25% missing, will be imputed/flagged)
  - `marital_status` — Single, Married, Widowed, Divorced
  - `sector` — industry sector of work (e.g., Trade, Agriculture, Fishing); mostly missing for non-workers, so it will be used only as a secondary/derived feature, not a core predictor
- **Excluded features:** `hours_worked` and `monthly_wage_usd` will **not** be used as model inputs. Checking the data shows `hours_worked = 0` for every single non-Employed row and only for non-Employed rows — including either column would leak the answer directly into the model instead of letting it learn genuine patterns.

## 5. Algorithms You Plan to Train

1. **Logistic Regression** (bootcamp) — simple, interpretable multiclass baseline; good reference point before trying more complex models.
2. **Decision Tree** (bootcamp) — a single interpretable tree that shows the exact split rules (e.g., age, education, region thresholds) driving a prediction, and serves as a direct comparison point for the ensemble-based Random Forest below.
3. **Random Forest** (bootcamp) — handles the mix of categorical and numeric features well and captures nonlinear interactions (e.g., age × education × region) without heavy preprocessing.
4. **XGBoost** (self-researched) — gradient boosting typically gives the strongest accuracy on structured/tabular data like this and has built-in tools for handling class imbalance.

**Optional exploratory step (unsupervised, not one of the four compared models):** Before training the classifiers above, **K-Means clustering** will be applied to the feature set (age, education, region, gender, marital_status) to explore natural demographic segments in the population — for example, groups that combine low education with high concentration in the "Not in Labour Force" class. This is exploratory only: because K-Means has no access to `employment_status` during training, its output (cluster IDs) is not directly comparable to the three real employment classes using classification metrics like F1, so it will not be scored against Macro F1 or included as one of the four required algorithms. It will instead be evaluated separately using Silhouette Score, and used only to inform feature engineering and EDA insights for the report.

## 6. Evaluation Plan

- **Metrics:** Accuracy, Precision, Recall, and Macro F1-Score, plus a confusion matrix for each model.
- **Selection metric:** **Macro F1-Score** will decide the best model, because the three employment classes are imbalanced (over half the data is "Employed") and Macro F1 weighs all three classes equally rather than letting the majority class dominate the score.

## 7. Deployment Sketch

- **Framework:** FastAPI
- **Endpoint:** `POST /predict`
- **Accepts (JSON):**
```json
{
  "age": 28,
  "gender": "Female",
  "region": "Banadir",
  "education": "Secondary",
  "marital_status": "Single"
}
```
- **Returns (JSON):**
```json
{
  "predicted_status": "Unemployed",
  "probabilities": {
    "Employed": 0.22,
    "Unemployed": 0.61,
    "Not in Labour Force": 0.17
  },
  "model_used": "XGBoost"
}
```

## 8. Repository Plan

```
maryan-dev-final-project/
├── dataset/
│   └── lfs_somalia_synthetic_2000.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── api/
│   └── app.py
├── models/
│   └── best_model.pkl
├── README.md
└── project_paper.md
```