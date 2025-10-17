
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# -----------------------------------------------------------
# 1.  helper: pretty metrics
# -----------------------------------------------------------
def regression_results(y_true, y_pred, label: str):
    ev = explained_variance_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {label} ---")
    print(f"explained_variance: {ev:7.4f}")
    print(f"r2:                 {r2:7.4f}")
    print(f"MAE:                {mae:7.4f}")
    print(f"MSE:                {mse:7.4f}")
    print(f"RMSE:               {rmse:7.4f}\n")

# -----------------------------------------------------------
# 2.  simple 1-D data with optional outliers
# -----------------------------------------------------------
np.random.seed(42)
X = 2 * np.random.rand(1000, 1)
y_ideal = 4 + 3 * X
y = y_ideal + 1.0 * np.random.randn(1000, 1)  # noisy but NO outliers yet

# add a few outliers
y_outlier = pd.Series(y.ravel())
outlier_idx = np.where(X.ravel() > 1.5)[0]
selected = np.random.choice(outlier_idx, size=5, replace=False)
y_outlier.iloc[selected] += np.random.uniform(50, 100, size=5)
y_outlier = y_outlier.values.reshape(-1, 1)

# -----------------------------------------------------------
# 3.  fit 3 models on OUTLIER data
# -----------------------------------------------------------
lin = LinearRegression().fit(X, y_outlier)
rid = Ridge(alpha=1.0).fit(X, y_outlier)
las = Lasso(alpha=0.2).fit(X, y_outlier)

pred_lin = lin.predict(X)
pred_rid = rid.predict(X)
pred_las = las.predict(X)

print("=== Results on data WITH outliers ===")
regression_results(y, pred_lin, "Ordinary")
regression_results(y, pred_rid, "Ridge")
regression_results(y, pred_las, "Lasso")

# -----------------------------------------------------------
# 4.  plot fits (with outliers)
# -----------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.scatter(X, y, alpha=0.4, ec="k", label="Original (no-outlier) data")
plt.plot(X, y_ideal, lw=3, color="g", label="Ideal (noise-free)")
plt.plot(X, pred_lin, lw=3, label="Linear")
plt.plot(X, pred_rid, ls="--", lw=2, label="Ridge")
plt.plot(X, pred_las, lw=2, label="Lasso")
plt.xlabel("Feature (X)"); plt.ylabel("Target (y)")
plt.title("Fits on data WITH outliers")
plt.legend(); plt.tight_layout(); plt.savefig("fits_with_outliers.png"); plt.close()

# -----------------------------------------------------------
# 5.  repeat on CLEAN data (no outliers)
# -----------------------------------------------------------
lin_clean = LinearRegression().fit(X, y)
rid_clean = Ridge(alpha=1.0).fit(X, y)
las_clean = Lasso(alpha=0.2).fit(X, y)

pred_lin_clean = lin_clean.predict(X)
pred_rid_clean = rid_clean.predict(X)
pred_las_clean = las_clean.predict(X)

print("=== Results on CLEAN data ===")
regression_results(y, pred_lin_clean, "Ordinary")
regression_results(y, pred_rid_clean, "Ridge")
regression_results(y, pred_las_clean, "Lasso")

# -----------------------------------------------------------
# 6.  high-dim regression via make_regression
# -----------------------------------------------------------
X_big, y_big, true_coef = make_regression(
    n_samples=100, n_features=100, n_informative=10,
    noise=10, random_state=42, coef=True
)
ideal_big = X_big @ true_coef

X_train, X_test, y_train, y_test, ideal_train, ideal_test = train_test_split(
    X_big, y_big, ideal_big, test_size=0.3, random_state=42
)

# full 100-feature models
lin_big = LinearRegression().fit(X_train, y_train)
rid_big = Ridge(alpha=1.0).fit(X_train, y_train)
las_big = Lasso(alpha=0.1).fit(X_train, y_train)

pred_lin_big = lin_big.predict(X_test)
pred_rid_big = rid_big.predict(X_test)
pred_las_big = las_big.predict(X_test)

print("=== High-dim FULL feature set ===")
regression_results(y_test, pred_lin_big, "Ordinary")
regression_results(y_test, pred_rid_big, "Ridge")
regression_results(y_test, pred_las_big, "Lasso")

# -----------------------------------------------------------
# 7.  Lasso-based feature selection
# -----------------------------------------------------------
thresh = 5.0  # abs coefficient threshold
selected_idx = np.where(np.abs(las_big.coef_) > thresh)[0]
print(f"Lasso selected {len(selected_idx)} features out of {X_big.shape[1]}")

X_train_sel = X_train[:, selected_idx]
X_test_sel  = X_test[:, selected_idx]

# retrain on selected features
lin_sel = LinearRegression().fit(X_train_sel, y_train)
rid_sel = Ridge(alpha=1.0).fit(X_train_sel, y_train)
las_sel = Lasso(alpha=0.1).fit(X_train_sel, y_train)

pred_lin_sel = lin_sel.predict(X_test_sel)
pred_rid_sel = rid_sel.predict(X_test_sel)
pred_las_sel = las_sel.predict(X_test_sel)

print("=== After Lasso feature selection ===")
regression_results(y_test, pred_lin_sel, "Ordinary")
regression_results(y_test, pred_rid_sel, "Ridge")
regression_results(y_test, pred_las_sel, "Lasso")

# -----------------------------------------------------------
# 8.  coefficient plot (full vs selected)
# -----------------------------------------------------------
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
x_axis = np.arange(len(true_coef))
plt.scatter(x_axis, true_coef, label="Ideal", color="blue", alpha=0.6)
plt.bar(x_axis - 0.2, lin_big.coef_, width=0.2, label="Linear", alpha=0.7)
plt.bar(x_axis, rid_big.coef_, width=0.2, label="Ridge", alpha=0.7)
plt.bar(x_axis + 0.2, las_big.coef_, width=0.2, label="Lasso", alpha=0.7)
plt.xlabel("Feature index"); plt.ylabel("Coefficient")
plt.title("Coefficients – FULL feature set"); plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(selected_idx, true_coef[selected_idx], label="Ideal (selected)", color="blue")
plt.bar(selected_idx - 0.2, lin_sel.coef_, width=0.2, label="Linear", alpha=0.7)
plt.bar(selected_idx, rid_sel.coef_, width=0.2, label="Ridge", alpha=0.7)
plt.bar(selected_idx + 0.2, las_sel.coef_, width=0.2, label="Lasso", alpha=0.7)
plt.xlabel("Feature index"); plt.ylabel("Coefficient")
plt.title("Coefficients – SELECTED features"); plt.legend()

plt.tight_layout()
plt.savefig("coefficient_comparison.png")
plt.close()

print("Done – images saved: fits_with_outliers.png, coefficient_comparison.png")