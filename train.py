import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# === Load normalized features and labels ===
X = np.load("D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\features_balanced_normalized.npy")
y = np.load("D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\labels_balanced.npy")

# === Check label distribution ===
label_counts = pd.Series(y).value_counts()
plt.figure(figsize=(8, 6))
label_counts.plot(kind="bar", color="skyblue")
plt.title("Accent Class Distribution")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === Define model with best GridSearchCV parameters ===
rf = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=6,
    min_samples_leaf=1,
    max_features='log2',
    max_depth=None,
    class_weight='balanced',
    random_state=42
)

# === Cross-validation setup ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === Cross-validation accuracy scores ===
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
print(f"✅ Cross-validation accuracies for each fold: {cv_scores}")
print(f"✅ Mean accuracy: {cv_scores.mean():.4f}")
print(f"✅ Std deviation: {cv_scores.std():.4f}")

# === Get cross-validated predictions for detailed metrics ===
y_pred = cross_val_predict(rf, X, y, cv=cv)

# === Overall accuracy on CV predictions ===
overall_accuracy = accuracy_score(y, y_pred)
print(f"\n📊 Overall accuracy (CV predictions): {overall_accuracy:.4f}\n")

# === Classification report ===
print("📊 Classification Report (CV predictions):")
print(classification_report(y, y_pred))

# === Confusion matrix ===
cm = confusion_matrix(y, y_pred, labels=np.unique(y))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix (Cross-Validated Predictions)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# === Train final model on full data ===
rf.fit(X, y)

# === Save final trained model ===
joblib.dump(rf, "D:\\Charbel_LEBBOUS\\AI Projects\\accent-detector\\model\\random_forest_accent_classifier_tuned.pkl")
print("💾 Final tuned model saved to random_forest_accent_classifier_tuned.pkl")

# === Feature importance plot ===
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Top 20 Feature Importances")
plt.bar(range(20), importances[indices[:20]])
plt.xticks(range(20), indices[:20])
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# === Learning curve ===
train_sizes, train_scores, test_scores = learning_curve(
    rf, X, y, cv=cv, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Training Score", marker='o')
plt.plot(train_sizes, test_mean, label="Validation Score", marker='s')
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Save predictions to CSV for future tracking ===
results_df = pd.DataFrame({
    'y_true': y,
    'y_pred': y_pred
})
results_df.to_csv("cv_predictions.csv", index=False)
print("📁 Predictions saved to cv_predictions.csv")
