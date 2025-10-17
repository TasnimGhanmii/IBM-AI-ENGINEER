
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

# -----------------------------------------------------------
# 1.  load data
# -----------------------------------------------------------
data = load_iris()
X, y = data.data, data.target
labels = data.target_names

# -----------------------------------------------------------
# 2.  baseline pipeline (fixed params)
# -----------------------------------------------------------
base_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])

# -----------------------------------------------------------
# 3.  train / test split (stratified)
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------
# 4.  fit baseline & evaluate
# -----------------------------------------------------------
base_pipe.fit(X_train, y_train)
base_score = base_pipe.score(X_test, y_test)
print(f"Baseline test accuracy: {base_score:.3f}")

y_pred_base = base_pipe.predict(X_test)

# -----------------------------------------------------------
# 5.  confusion matrix – baseline
# -----------------------------------------------------------
cm_base = confusion_matrix(y_test, y_pred_base)
plt.figure()
sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Baseline Pipeline – Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig("cm_baseline.png"); plt.close()

# -----------------------------------------------------------
# 6.  hyper-parameter tuning via GridSearchCV
# -----------------------------------------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()),
    ("knn", KNeighborsClassifier())
])

param_grid = {
    "pca__n_components": [2, 3],
    "knn__n_neighbors": [3, 5, 7]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    verbose=2
)

grid.fit(X_train, y_train)

print("Best CV params:", grid.best_params_)
print(f"Best CV accuracy: {grid.best_score_:.3f}")

# -----------------------------------------------------------
# 7.  tuned model on hold-out test set
# -----------------------------------------------------------
tuned_score = grid.score(X_test, y_test)
print(f"Tuned test accuracy: {tuned_score:.3f}")

y_pred_tuned = grid.predict(X_test)

# -----------------------------------------------------------
# 8.  confusion matrix – tuned
# -----------------------------------------------------------
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
plt.figure()
sns.heatmap(cm_tuned, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Tuned Pipeline – Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig("cm_tuned.png"); plt.close()

print("Images saved: cm_baseline.png, cm_tuned.png")