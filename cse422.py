

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.utils import resample

df = pd.read_csv("/content/sample_data/consumer_classification_dataset.csv")
df.head()

print(f"There are {df.shape[1]} features in the dataset")
print("Binary classification problem. Because we are predicting two labels from the features.")
print(f"There are {df.size} data points.")
print(set(df.dtypes))

print(df.isna().sum())

len(df)

df["Churn"].value_counts()

labels = ["Churn", "Not Churn"]
values = [743, 757]

plt.bar(labels, values)
plt.xlabel("Labels")
plt.ylabel("Counts")
plt.title("Bar Chart of Labels")
plt.show()

numerical_data = df.select_dtypes(include='number')
numerical_features=numerical_data.columns.tolist()
categorical_data=df.select_dtypes(include= 'object')
categorical_features=categorical_data.columns.tolist()

print(numerical_features)
print(categorical_features)

numerical_data.describe().T

categorical_data.describe().T

plt.figure(figsize=(16, 10))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()


outlier_summary = {}

for col in numerical_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_summary[col] = len(outliers)


outlier_df = pd.DataFrame.from_dict(outlier_summary, orient='index', columns=['Outlier_Count'])
print("\nOutlier Summary (IQR Method):\n")
print(outlier_df.sort_values(by='Outlier_Count', ascending=False))

for col in df.columns:
    if df[col].isna().sum() > 0:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

df.head()

df.dtypes

correlation_matrix = df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Features', fontsize=16)
plt.show()

df["Churn"].value_counts()

df_majority = df[df['Churn'] == 0]
df_minority = df[df['Churn'] == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

X = df_upsampled.drop(columns=['Churn'])
y = df_upsampled['Churn']

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric='logloss', random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, early_stopping=True, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'AUC': auc,
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred, output_dict=True)
    }

for name, res in results.items():
    print(f"{name}")
    print(f"  - Accuracy : {res['Accuracy']:.4f}")
    print(f"  - Precision: {res['Precision']:.4f}")
    print(f"  - Recall   : {res['Recall']:.4f}")
    print(f"  - AUC      : {res['AUC']:.4f}\n")

metrics_df = pd.DataFrame({
    'Accuracy': [results[m]['Accuracy'] for m in models],
    'Precision': [results[m]['Precision'] for m in models],
    'Recall': [results[m]['Recall'] for m in models]
}, index=models.keys())

metrics_df.plot(kind='bar', figsize=(10,6))
plt.title('Accuracy, Precision, Recall Comparison')
plt.ylabel('Score')
plt.ylim(0,1)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, res) in zip(axes, results.items()):
    sns.heatmap(res['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix: {name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
for name, model in models.items():
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["AUC"]:.2f})')

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()