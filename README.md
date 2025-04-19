# Insurance-quote-conversion-prediction-using-ML-models-SMOTE-feature-selection-parallel-training
To build a classification pipeline that predicts the likelihood of quote conversion using structured customer and quote data. The pipeline handles class imbalance, selects top features, and compares model performances with cross-validation and ROC AUC metrics.
---

## 🛠️ Technologies & Techniques

| Task                         | Tool / Technique            |
|------------------------------|-----------------------------|
| Classification Models        | MLPClassifier, LinearSVC, Decision Tree, Random Forest, KNN |
| Feature Selection            | SelectKBest (Chi-Square)    |
| Class Imbalance Handling     | SMOTE (Synthetic Minority Oversampling) |
| Evaluation Metrics           | Accuracy, ROC AUC, F1 Score |
| Performance Tuning           | Cross-validation, GridSearchCV |
| Acceleration                 | Parallel Processing (n_jobs) |
| Libraries                    | scikit-learn, imbalanced-learn, pandas, NumPy |

---

##  Workflow Summary

1. **Data Preprocessing**
   - Handled missing values and encoded categorical variables
   - Scaled features and performed label encoding

2. **Feature Selection**
   - Selected top features using SelectKBest and chi-squared scores

3. **Class Imbalance Handling**
   - Applied SMOTE to oversample the minority class

4. **Model Training**
   - Trained multiple classifiers: MLP, LinearSVC, Decision Trees, Random Forest, KNN
   - Evaluated using ROC AUC and classification reports

5. **Performance Optimization**
   - Used parallel processing to speed up model evaluation
   - Tuned hyperparameters for best models

---

##  Sample Results


| Model          | ROC AUC | Accuracy | F1 Score |
|----------------|---------|----------|----------|
| MLPClassifier  | 0.91    | 88.4%    | 0.87     |
| Random Forest  | 0.89    | 86.9%    | 0.85     |
| Linear SVC     | 0.87    | 85.3%    | 0.83     |

---

##  Highlights

- Balanced the dataset using SMOTE for fair classification
- Used SelectKBest to improve efficiency and model focus
- Compared model performance using ROC AUC and F1 metrics
- Accelerated training and validation with parallel computing
