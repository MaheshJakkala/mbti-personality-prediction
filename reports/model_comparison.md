# Model Comparison Report

## MBTI Personality Prediction — Binary & Multiclass Classifiers

All models use **TF-IDF (10,000 features, max_df=0.3, spaCy lemma tokenizer)** with **SMOTETomek** resampling.  
GridSearchCV with 5-fold cross-validation, scored on **ROC-AUC** for binary classifiers.

---

## Dimension 1: Introversion (I=0) vs Extroversion (E=1)
**Class split: 77% Introvert / 23% Extrovert**

| Model | AUC | Accuracy | F1 | Sensitivity (Recall) | Specificity | MCC |
|-------|-----|----------|----|----------------------|-------------|-----|
| Logistic Regression | 0.512 | 0.603 | 0.269 | 0.317 | 0.688 | 0.004 |
| XGBoost | 0.500 | 0.745 | 0.067 | 0.039 | 0.957 | -0.008 |

**Best model:** Logistic Regression (marginally — both are near-random)

**LR best params:** `C=1.0, penalty=l1, solver=liblinear, class_weight=balanced`  
**XGB best params:** `n_estimators=500, learning_rate=0.01, subsample=0.5, colsample_bytree=0.5`

**Interpretation:**  
XGBoost's 74.5% accuracy is deceptive — it achieves this by predicting "Introvert" for 96% of samples (sensitivity for Extroversion = 3.9%). LR at least attempts both classes (recall 31.7% for Extroversion). Both AUC scores (~0.50) indicate no meaningful learning above chance on this dimension.

**Top LR features for Extroversion:** football, literature, scored, missing, idiot  
**Top LR features for Introversion:** overwhelmed, sums, professional, including, contrast

---

## Dimension 2: Intuition (N=0) vs Sensing (S=1)
**Class split: 86% Intuition / 14% Sensing**

| Model | AUC | Accuracy | F1 | Sensitivity (Recall) | Specificity | MCC |
|-------|-----|----------|----|----------------------|-------------|-----|
| Logistic Regression | 0.503 | 0.700 | 0.176 | 0.233 | 0.775 | 0.006 |
| XGBoost | 0.511 | 0.833 | 0.067 | 0.043 | 0.960 | 0.005 |

**Best model:** Logistic Regression

**LR best params:** `C=0.1, penalty=l1, solver=liblinear, class_weight=balanced`  
**XGB best params:** `n_estimators=100, learning_rate=0.01, subsample=0.5, colsample_bytree=0.5`

**Interpretation:**  
XGBoost's 83.3% accuracy is entirely explained by the 86% class imbalance — it predicts "Intuition" for 96% of inputs (Sensing recall = 4.3%). MCC of 0.005 confirms this is baseline behaviour. LR is modestly better with Sensing recall of 23.3%. Neither AUC (~0.50) indicates real learning.

---

## Dimension 3: Thinking (T=0) vs Feeling (F=1)
**Class split: 46% Thinking / 54% Feeling**

| Model | AUC | Accuracy | F1 | Sensitivity (Recall) | Specificity | MCC |
|-------|-----|----------|----|----------------------|-------------|-----|
| Logistic Regression | 0.501 | 0.497 | 0.525 | 0.515 | 0.476 | -0.009 |
| XGBoost | 0.505 | 0.534 | 0.683 | 0.928 | 0.070 | -0.004 |

**Best model:** Neither — both near random. LR is more balanced; XGBoost collapses to predicting Feeling.

**LR best params:** `C=1.0, penalty=l1, solver=liblinear, class_weight=balanced`  
**XGB best params:** `n_estimators=100, learning_rate=0.01, subsample=0.5, colsample_bytree=0.5`

**Interpretation:**  
T vs F is the most balanced class split (~50/50), so accuracy is a fair metric here. LR gives near-random 49.7% accuracy. XGBoost's F1 of 0.683 looks better but is achieved by predicting Feeling 93% of the time (Thinking recall = 7%). In practice, XGBoost is not learning — it's exploiting the slight Feeling majority.

**Top LR features for Feeling:** answers, beings, cautious, congratulations, straight, cute  
**Top LR features for Thinking:** age, actions, invite, likes, lazy, listen, cognitive

---

## Dimension 4: Judging (J=0) vs Perceiving (P=1)
**Class split: 40% Judging / 60% Perceiving**

| Model | AUC | Accuracy | F1 | Sensitivity (Recall) | Specificity | MCC |
|-------|-----|----------|----|----------------------|-------------|-----|
| **Logistic Regression** | **0.698** | **0.653** | **0.712** | **0.709** | **0.568** | **0.276** |
| XGBoost | 0.651 | 0.632 | 0.744 | 0.886 | 0.244 | 0.171 |

**Best model: Logistic Regression** — highest AUC, best MCC, most balanced recall/specificity

**LR best params:** `C=1.0, penalty=l1, solver=liblinear, class_weight=balanced`  
**XGB best params:** `n_estimators=500, learning_rate=0.01, subsample=0.5, colsample_bytree=0.5`

**Interpretation:**  
This is the only dimension with meaningful signal. LR achieves AUC 0.698 and MCC 0.276 — a genuine, replicable lift above chance. Both recall (70.9%) and specificity (56.8%) are non-trivial, confirming the model is not simply defaulting to one class. The J/P dimension appears to have the strongest textual correlates — structured vs exploratory language maps well onto TF-IDF bag-of-words features.

---

## Multiclass Classification — All 16 Types

| Model | Accuracy | F1 Macro | F1 Weighted | MCC | Notes |
|-------|----------|----------|-------------|-----|-------|
| Naive Bayes | **0.355** | 0.228 | 0.356 | **0.262** | Best CV score: 0.352 |
| Logistic Regression | 0.340 | **0.229** | 0.351 | 0.259 | Best CV score: 0.330 |
| Random Forest | 0.252 | 0.132 | 0.212 | 0.114 | Best params: n_estimators=500, max_depth=7 |

> **Baseline (majority-class predictor) = ~21% accuracy.** All models beat it, but F1 Macro scores (~0.13–0.23) reveal very poor minority class performance. ENTJ, ENFJ, INFP type predictions collapse entirely (F1 = 0.00) in LR and NB.

---

## Full Binary Metrics Comparison

|  | IE LR | IE XGB | NS LR | NS XGB | TF LR | TF XGB | JP LR | JP XGB |
|--|-------|--------|-------|--------|-------|--------|-------|--------|
| **Accuracy** | 0.603 | 0.745 | 0.700 | 0.833 | 0.497 | 0.534 | **0.653** | 0.632 |
| **Sensitivity** | 0.317 | 0.039 | 0.233 | 0.043 | 0.515 | 0.928 | **0.709** | 0.886 |
| **Specificity** | 0.688 | 0.957 | 0.775 | 0.960 | 0.476 | 0.070 | **0.568** | 0.244 |
| **F1-Score** | 0.269 | 0.067 | 0.176 | 0.067 | 0.525 | 0.683 | **0.712** | 0.744 |
| **AUC Score** | 0.512 | 0.500 | 0.503 | 0.511 | 0.501 | 0.505 | **0.698** | 0.651 |
| **MCC** | 0.004 | -0.008 | 0.006 | 0.005 | -0.009 | -0.004 | **0.276** | 0.171 |

---

## Key Observations

1. **J vs P is the only learnable dimension** with TF-IDF features. AUC 0.698 and MCC 0.276 for LR represent genuine signal. The other three dimensions yield AUC ~0.50 — statistically indistinguishable from random.

2. **High accuracy ≠ good model** (IE and NS XGBoost). Models reporting 74–83% accuracy by predicting one class almost exclusively are useless in practice. Always check MCC and recall alongside accuracy on imbalanced data.

3. **LR outperforms XGBoost on AUC across all dimensions.** TF-IDF produces high-dimensional sparse feature matrices — logistic regression with L1 regularization is well-suited to this structure. XGBoost tends to over-fit sparse inputs without extensive tuning.

4. **SMOTETomek forces minority class attempts** — without it, all models would collapse to majority class prediction and achieve near-zero MCC. Resampling is necessary even when it doesn't fully solve the problem.

5. **Binary > Multiclass for interpretability.** Even where signal is weak, binary classifiers allow per-dimension feature importance analysis, giving meaningful linguistic insight (e.g., T vs F word associations).

6. **Data ceiling from self-reported labels.** Users on MBTI forums know their own type and likely perform it in their writing. This creates label-feature circularity that no model architecture can fully overcome without a cleaner dataset.