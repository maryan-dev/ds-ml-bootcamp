1. The Process 

Notebook B1 cleaned the raw loan data: fixed currency text, fixed Yes/No typos, filled missing values, dropped duplicates (103 → 100 rows), capped outliers, encoded Yes/No to 0/1, added two new features (DebtToIncome, IncomePerYearEmployed), scaled numbers, saved clean_loan_dataset.csv.

Notebook B2 loaded that file, split 80/20 (stratify=y, random_state=42), trained three models: Logistic Regression, Random Forest, and Decision Tree (my third pick). Checked each with Accuracy, Precision, Recall, F1, confusion matrix, plus one sanity-check row.

2. Comparison of Models

On the sanity-check row, all three models said Approved (1). Real answer was Rejected (0). All three wrong on that one row — no useful difference there.

Across the full 20-row test set, Logistic Regression and Random Forest gave the exact same confusion matrix. Decision Tree caught one more true approval, but also approved one more risky case. With only 100 rows total, this overlap makes sense — not enough data to see much difference between models.

3. Understanding Random Forest

In my own words: Random Forest trains many decision trees, each on a random slice of rows and features. Each tree votes, majority wins. One tree overfits easily. Many different trees averaged together cancel out each other's mistakes, so the forest generalizes better.

4. Third Classifier: Decision Tree

I picked Decision Tree because it's the base building block of Random Forest, so comparing them shows what the ensemble adds. It fits loan approval well — the features (credit score, employment years, collateral) split naturally into yes/no rules, similar to how a loan officer thinks.

From scikit-learn's docs: the tree picks the best feature and cutoff to split the data at each step (default: Gini impurity). Good: fully readable, you can trace the exact path to a decision. Bad: unstable, small data changes give a different tree, easy to overfit without limits.

Its metrics: highest Recall (0.923) and highest F1 (0.800) of all three models, but slightly lower Precision (0.706) — it approved more borderline cases.

5. Metrics Discussion

ModelAccuracyPrecisionRecallF1Logistic Regression0.7000.7330.8460.786Random Forest0.7000.7330.8460.786Decision Tree0.7000.7060.9230.800

All tied on Accuracy. Logistic Regression and Random Forest tied everywhere expected on a small dataset, not enough rows for the ensemble's strength to show. Decision Tree wins Recall and F1, but loses a bit on Precision: it approves more, catches more good cases, but also lets in a few bad ones.

6. My Findings

I'd lean toward Random Forest it's an ensemble, so it should stay more stable as the dataset grows, even though it ties Logistic Regression right now. If catching every good applicant matters most, Decision Tree's numbers make a fair case too. Main takeaway: 100 rows is too small to fully trust these results. The real test is running this same pipeline on a much bigger loan dataset.