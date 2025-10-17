
# ------------------------------------------------------------------
# 1.  Imports
# ------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from scipy.stats import skew

# ------------------------------------------------------------------
# 2.  Load data
# ------------------------------------------------------------------
data = fetch_california_housing()
X, y = data.data, data.target
print(data.DESCR)

# ------------------------------------------------------------------
# 3.  Train / test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 4.  Quick EDA
# ------------------------------------------------------------------
eda = pd.DataFrame(X_train, columns=data.feature_names)
eda["MedHouseVal"] = y_train
print(eda.describe())

# ------------------------------------------------------------------
# 5.  Distribution of target
# ------------------------------------------------------------------
plt.figure()
plt.hist(1e5 * y_train, bins=30, color="lightblue", edgecolor="black")
plt.title(f"Median House Value Distribution\nSkewness: {skew(y_train):.2f}")
plt.xlabel("Median House Value ($)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("target_distribution.png")
plt.close()

# ------------------------------------------------------------------
# 6.  Model fitting
# ------------------------------------------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# ------------------------------------------------------------------
# 7.  Metrics
# ------------------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation on held-out 20 %:")
print(f"MAE : ${mae*1e5:,.0f}")
print(f"MSE : ${mse*1e10:,.0f}")
print(f"RMSE: ${rmse*1e5:,.0f}")
print(f"R²  : {r2:.4f}")

# ------------------------------------------------------------------
# 8.  Actual vs Predicted
# ------------------------------------------------------------------
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Random Forest – Actual vs Predicted")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.close()

# ------------------------------------------------------------------
# 9.  Residuals
# ------------------------------------------------------------------
residuals = 1e5 * (y_test - y_pred)
plt.figure()
plt.hist(residuals, bins=30, color="lightblue", edgecolor="black")
plt.title("Residuals Distribution")
plt.xlabel("Residuals ($)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("residuals_histogram.png")
plt.close()

print("\nResidual statistics:")
print(f"Mean : ${np.mean(residuals):,.0f}")
print(f"Std  : ${np.std(residuals):,.0f}")

# ------------------------------------------------------------------
# 10.  Residuals vs Actual (sorted)
# ------------------------------------------------------------------
res_df = pd.DataFrame({"Actual": 1e5 * y_test, "Residuals": residuals})
res_df = res_df.sort_values("Actual")

plt.figure()
plt.scatter(res_df["Actual"], res_df["Residuals"], marker="o", alpha=0.4, ec="k")
plt.title("Residuals vs Actual (sorted)")
plt.xlabel("Actual Median House Value ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.tight_layout()
plt.savefig("residuals_vs_actual.png")
plt.close()

# ------------------------------------------------------------------
# 11.  Feature importances
# ------------------------------------------------------------------
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = data.feature_names

plt.figure()
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.close()

# ------------------------------------------------------------------
# 12.  Correlation matrix (extra)
# ------------------------------------------------------------------
corr = eda.corr(numeric_only=True)
print("\nTop 5 correlations with MedHouseVal:")
print(corr["MedHouseVal"].abs().sort_values(ascending=False).head())

print("\nAll images saved as *.png in the current directory.")