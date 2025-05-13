# Credit Card Default Prediction

This project focuses on predicting whether a client will default on their credit card payment next month using machine learning. It involves detailed data preprocessing, multiple model comparisons, and experiment tracking using MLflow.

---

## Project Stages

### 1.  Exploratory Data Analysis (EDA)
- Checked class imbalance and default rate
- Analyzed feature distributions (age, bill amounts, payments)
- Identified skewed features and outliers

### 2.  Preprocessing
- Encoding:
  - One-Hot Encoding for linear models (Logistic Regression, KNN)
  - Label Encoding for tree-based models (XGBoost, Random Forest, LightGBM)
- Feature Scaling: `StandardScaler` applied to numeric features
- Feature Engineering: Created features like `isFullyPaid`, `bill trends`, and `payment differences`

### 3.  Model Building
- **Models Evaluated**:
  - Logistic Regression
  - K-Nearest Neighbors
  - Random Forest
  - LightGBM
  - XGBoost (Best Model)

- **Tuning Applied**:
  - Optuna Hyperparameter Tuning
  - Threshold Tuning to focus on catching defaulters (best threshold: `0.32`)

### 4. Evaluation
- Best Model: **XGBoost**
- Encoding: **Label Encoding**
- Threshold: **0.32**
- **Performance**:
  - Accuracy: `~0.79`
  - F1 Score: `~0.54`
  - AUC: `~0.77`
- **Business Goal**: Focused on recall (catching defaulters), not just accuracy

---

##  Model Calibration Plot

![image](https://github.com/user-attachments/assets/cf40dfac-bdf9-4630-8b43-6793447a7488)



> The calibration curve shows that the predicted probabilities from the XGBoost model are well-aligned with actual default rates (Regression Line ≈ Ideal Line), validating the model's confidence in its predictions.

---

##  MLflow Tracking

All experiments were tracked using MLflow:

- Logged:
  - Parameters: model name, threshold, hyperparameters
  - Metrics: accuracy, f1 score, recall, precision, AUC
  - Artifacts: models, feature lists, tuning plots
- Separate runs for:
  - Pre-tuned models
  - Optuna-tuned models
  - Threshold-tuned models

You can filter runs using `params.model` like:
- `XGBoost Pre-Tuned`
- `XGBoost Tuned`
- `XGBoost Threshold Tuned`

---

##  Conclusion

This project demonstrates an end-to-end ML pipeline that:
- Handles imbalanced classification
- Applies tuning aligned with business goal(focused on catching defaulters than increasing accuracy)
- Leverages MLflow for full transparency
- Selects the best model based on both performance and interpretability

## How to Use MLflow Locally
### Step-by-step:

1. ip install mlflow
2. mlflow ui and copy the link
3.mlflow.set_tracking_uri("http://127.0.0.1:5000") #paste the link copied
4. now log models and metrics

