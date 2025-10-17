
# ------------------------------------------------------------
# 1.  Imports
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------------
# 2.  Load data
# ------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
labels = data.target_names
feature_names = data.feature_names
print("Classes:", labels)

# ------------------------------------------------------------
# 3.  Standardise
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# 4.  Add Gaussian noise to simulate measurement error
# ------------------------------------------------------------
RNG = np.random.RandomState(42)
NOISE_FACTOR = 0.5
X_noisy = X_scaled + NOISE_FACTOR * RNG.normal(size=X.shape)

# ------------------------------------------------------------
# 5.  Visualise noise impact
# ------------------------------------------------------------
df_clean = pd.DataFrame(X_scaled, columns=feature_names)
df_noisy = pd.DataFrame(X_noisy, columns=feature_names)

feat = feature_names[5]  # mean texture
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df_clean[feat], bins=20, alpha=0.7, color="blue")
plt.title("Original (noise-free)")

plt.subplot(1, 3, 2)
plt.hist(df_noisy[feat], bins=20, alpha=0.7, color="red")
plt.title("With Gaussian noise")

plt.subplot(1, 3, 3)
plt.scatter(df_clean[feat], df_noisy[feat], alpha=0.5)
plt.xlabel("Original")
plt.ylabel("Noisy")
plt.title("Scatter comparison")

plt.tight_layout()
plt.savefig("noise_visualisation.png")
plt.close()

# ------------------------------------------------------------
# 6.  Train / test split (noisy data)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_noisy, y, test_size=0.30, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 7.  Models
# ------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel="linear", C=1, random_state=42)

knn.fit(X_train, y_train)
svm.fit(X_train, y_train)

# ------------------------------------------------------------
# 8.  Evaluation helpers
# ------------------------------------------------------------
def evaluate(model, X, y, phase="Testing"):
    """Return dict of scores and pretty-print them."""
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    print(f"\n{model.__class__.__name__} {phase} Accuracy: {acc:.3f}")
    print(classification_report(y, pred, target_names=labels))
    return pred, acc

# ------------------------------------------------------------
# 9.  Test-set performance
# ------------------------------------------------------------
y_pred_knn_test, _ = evaluate(knn, X_test, y_test, "Testing")
y_pred_svm_test, _ = evaluate(svm, X_test, y_test, "Testing")

# ------------------------------------------------------------
# 10.  Confusion matrices – test
# ------------------------------------------------------------
def plot_cm(y_true, y_pred, title, save_as):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()

plot_cm(y_test, y_pred_knn_test, "KNN – Test Confusion Matrix", "knn_test_cm.png")
plot_cm(y_test, y_pred_svm_test, "SVM – Test Confusion Matrix", "svm_test_cm.png")

# ------------------------------------------------------------
# 11.  Training-set performance (over-fit check)
# ------------------------------------------------------------
y_pred_knn_train, acc_knn_train = evaluate(knn, X_train, y_train, "Training")
y_pred_svm_train, acc_svm_train = evaluate(svm, X_train, y_train, "Training")

plot_cm(y_train, y_pred_knn_train, "KNN – Training CM", "knn_train_cm.png")
plot_cm(y_train, y_pred_svm_train, "SVM – Training CM", "svm_train_cm.png")

# ------------------------------------------------------------
# 12.  Quick over-fit summary
# ------------------------------------------------------------
print("\nOver-fit check (accuracy):")
print(f"KNN  – Train: {acc_knn_train:.3f}  Test: {accuracy_score(y_test, y_pred_knn_test):.3f}")
print(f"SVM  – Train: {acc_svm_train:.3f}  Test: {accuracy_score(y_test, y_pred_svm_test):.3f}")

print("\nDone – all images saved as *.png")