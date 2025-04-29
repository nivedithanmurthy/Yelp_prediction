import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # For non-GUI environments
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Create output directory
os.makedirs('hpc_outputs', exist_ok=True)

# Load Data
review_df = pd.read_csv('review_with_aspect_sentiment.csv')
business_df = pd.read_csv('business_prepared.csv')

# Merge
df = pd.merge(review_df, business_df, on='business_id', suffixes=('', '_biz'))

# Create Target
df['performance'] = df['business_stars'].apply(lambda x: 'grow' if x >= 4.0 else 'decline' if x <= 2.5 else 'stable')
le = LabelEncoder()
df['target'] = le.fit_transform(df['performance'])

# Feature Engineering
df['review_length'] = df['text'].apply(lambda x: len(str(x)))

# New Feature List
features = [
    'stars',
    'sentiment_overall',
    'sentiment_food',
    'sentiment_ambiance',
    'sentiment_service',
    'sentiment_location',
    'sentiment_price',
    'reviewer_average_stars',
    'reviewer_useful',
    'review_length',
    'business_review_count'
]

df = df.dropna(subset=features + ['target'])

X = df[features]
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Models
models = {
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
    "SVM": SVC(class_weight='balanced', probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "LightGBM": LGBMClassifier(class_weight='balanced', random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

best_model = None
best_score = 0
best_model_name = None

# Open log file
log_file = open("hpc_outputs/training_log.txt", "w")

# Train and Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, target_names=le.classes_)
    weighted_f1 = classification_report(y_test, preds, output_dict=True)['weighted avg']['f1-score']

    # Log
    log_file.write(f"\n{name} Performance:\n")
    log_file.write(report)
    log_file.write("\n" + "-"*60 + "\n")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"hpc_outputs/{name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()

    if weighted_f1 > best_score:
        best_model = model
        best_score = weighted_f1
        best_model_name = name

# Save Best Model
model_filename = f"hpc_outputs/best_model_{best_model_name.replace(' ','_')}.pkl"
joblib.dump(best_model, model_filename)
log_file.write(f"\nBest Model: {best_model_name} with Weighted F1-score: {best_score:.4f}\n")
log_file.write(f"Model saved as: {model_filename}\n")

# Feature Importance
importances = None
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_[0])

if importances is not None:
    feature_importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    plt.figure(figsize=(8,6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title(f"Feature Importance - {best_model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"hpc_outputs/{best_model_name.replace(' ','_')}_feature_importance.png")
    plt.close()
else:
    log_file.write(f"Feature importance not available for {best_model_name}\n")

log_file.close()