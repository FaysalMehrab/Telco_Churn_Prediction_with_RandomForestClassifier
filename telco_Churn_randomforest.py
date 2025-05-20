import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading and preprocessing data...")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.drop(columns=['customerID'])
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Feature engineering
print("Applying feature engineering...")
df['MonthlyCostPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1e-6)
df['TotalChargeRatio'] = df['TotalCharges'] / df['MonthlyCharges']
df['ServiceDensity'] = df[[col for col in df.columns if 'Streaming' in col]].sum(axis=1)

# Split data
print("Splitting data into train and test sets...")
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Identify column types
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
num_cols = X_train.select_dtypes(exclude='object').columns.tolist()
print(f"Categorical columns: {cat_cols}")
print(f"Numerical columns: {num_cols}")

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

# Full pipeline with feature selection, SMOTE, and classifier
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(mutual_info_classif)),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Improved param_grid for better precision/recall balance
param_grid = {
    'feature_selection__k': [12, 15, 18],
    'classifier__n_estimators': [200, 300, 400],
    'classifier__max_depth': [3, 5, 7],
    'classifier__min_samples_split': [5, 10, 15],
    'classifier__min_samples_leaf': [1, 2, 3],
    'classifier__class_weight': ['balanced', None]
}

print("Starting grid search with cross-validation...")
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='recall', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("\nBest model and parameters found:")
print(grid_search.best_estimator_)
print(f"Best Parameters: {grid_search.best_params_}")

# Best model evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Metrics calculation
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print("\nInitial test set metrics (default threshold 0.5):")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"AUC: {roc_auc:.4f}")
print("Confusion Matrix:")
print(cm)

# Threshold tuning for optimal F1
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_threshold = thresholds[np.argmax(f1_scores)]
optimal_y_pred = (y_proba >= optimal_threshold).astype(int)

print(f"\nOptimal Threshold for F1: {optimal_threshold:.4f}")
print("Metrics at optimal threshold:")
print(f"Adjusted Accuracy: {accuracy_score(y_test, optimal_y_pred):.4f}")
print(f"Adjusted Precision: {precision_score(y_test, optimal_y_pred):.4f}")
print(f"Adjusted Recall: {recall_score(y_test, optimal_y_pred):.4f}")
print("Adjusted Confusion Matrix:")
print(confusion_matrix(y_test, optimal_y_pred))

# Feature importance extraction
feature_names = (best_model.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .get_feature_names_out(cat_cols).tolist() + num_cols)
selected_mask = best_model.named_steps['feature_selection'].get_support()
selected_features = np.array(feature_names)[selected_mask]
importances = best_model.named_steps['classifier'].feature_importances_

feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 10 Features by Importance:")
print(feature_importance.head(10))

# --- Plots and Export ---

# Fix seaborn warning for feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15),
            hue='Feature', palette='magma', legend=False)
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, optimal_y_pred), annot=True, fmt='d', cmap='Blues', 
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.title("Confusion Matrix (Optimal Threshold)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

# Export data
print("\nExporting predictions and feature importances to CSV...")
pd.DataFrame({'Actual': y_test, 'Predicted': optimal_y_pred, 'Probability': y_proba}) \
  .to_csv("predictions.csv", index=False)
feature_importance.to_csv("feature_importance.csv", index=False)
print("Export complete.")