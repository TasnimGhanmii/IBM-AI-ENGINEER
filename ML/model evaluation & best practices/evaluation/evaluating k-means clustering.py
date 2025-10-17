
# -----------------------------------------------------------
# 0.  Install missing packages (idempotent)
# -----------------------------------------------------------
import sys, subprocess, importlib

PKGS = {
    "numpy": "2.2.0",
    "pandas": "2.2.3",
    "scikit-learn": "1.6.0",
    "matplotlib": "3.9.3",
    "scipy": "1.14.1"
}

for pkg, ver in PKGS.items():
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", f"{pkg}=={ver}"]
        )

# -----------------------------------------------------------
# 1.  Imports
# -----------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import cm

# -----------------------------------------------------------
# 2.  Evaluation helper (silhouette + Davies-Bouldin)
# -----------------------------------------------------------
def evaluate_clustering(X, labels, n_clusters, ax=None, title_suffix=""):
    if ax is None:
        ax = plt.gca()

    sil_avg = silhouette_score(X, labels)
    sample_vals = silhouette_samples(X, labels)

    unique_labs = np.unique(labels)
    cmap = cm.tab10
    colors = {lab: cmap(float(lab) / n_clusters) for lab in unique_labs}

    y_lower = 10
    for lab in unique_labs:
        vals = sample_vals[labels == lab]
        vals.sort()
        size = vals.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, vals,
            facecolor=colors[lab], edgecolor=colors[lab], alpha=0.7
        )
        ax.text(-0.05, y_lower + 0.5 * size, str(lab))
        y_lower = y_upper + 10

    ax.set_title(f"Silhouette {title_suffix}\nAvg={sil_avg:.2f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster")
    ax.axvline(x=sil_avg, color="red", linestyle="--")
    ax.set_xlim([-0.25, 1])
    ax.set_yticks([])

# -----------------------------------------------------------
# 3.  Synthetic 4-blob data
# -----------------------------------------------------------
X, y = make_blobs(
    n_samples=500, n_features=2, centers=4,
    cluster_std=[1.0, 3, 5, 2], random_state=42
)

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_km = kmeans.fit_predict(X)

# -----------------------------------------------------------
# 4.  Visualise blobs + clustering + silhouette
# -----------------------------------------------------------
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], s=50, alpha=0.6, edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', s=200, marker='X', label='Centroids')
plt.title("4 synthetic blobs")
plt.legend()

plt.subplot(1, 3, 2)
colors = cm.tab10(y_km.astype(float) / n_clusters)
plt.scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.6, edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', s=200, marker='X', label='Centroids')
plt.title("k-means k=4")
plt.legend()

ax3 = plt.subplot(1, 3, 3)
evaluate_clustering(X, y_km, n_clusters, ax=ax3, title_suffix="k-means")

plt.tight_layout()
plt.savefig("blobs_k4.png")
plt.close()

# -----------------------------------------------------------
# 5.  Stability across random seeds
# -----------------------------------------------------------
n_runs = 8
inertia_vals = []

plt.figure(figsize=(16, 16))
cols = 2
rows = (n_runs + cols - 1) // cols

for run in range(n_runs):
    km = KMeans(n_clusters=4, random_state=None)
    km.fit(X)
    inertia_vals.append(km.inertia_)

    plt.subplot(rows, cols, run + 1)
    plt.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='tab10', alpha=0.6, edgecolor='k')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                c='red', marker='x', s=200)
    plt.title(f"Run {run+1}")
    plt.xlabel("Feature 1"); plt.ylabel("Feature 2")

plt.tight_layout()
plt.savefig("stability_runs.png")
plt.close()

print("Inertia across random seeds:", [round(v, 2) for v in inertia_vals])

# -----------------------------------------------------------
# 6.  Metrics vs k (elbow, silhouette, Davies-Bouldin)
# -----------------------------------------------------------
k_range = range(2, 11)
inertias, sils, dbs = [], [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    labs = km.fit_predict(X)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X, labs))
    dbs.append(davies_bouldin_score(X, labs))

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(k_range, inertias, marker='o')
plt.title("Elbow (inertia)"); plt.xlabel("k"); plt.ylabel("inertia")

plt.subplot(1, 3, 2)
plt.plot(k_range, sils, marker='o', color='green')
plt.title("Silhouette"); plt.xlabel("k"); plt.ylabel("silhouette")

plt.subplot(1, 3, 3)
plt.plot(k_range, dbs, marker='o', color='red')
plt.title("Davies-Bouldin"); plt.xlabel("k"); plt.ylabel("DB index")

plt.tight_layout()
plt.savefig("metrics_vs_k.png")
plt.close()

# -----------------------------------------------------------
# 7.  Visual comparison k=2,3,4
# -----------------------------------------------------------
plt.figure(figsize=(18, 12))
for idx, k in enumerate([2, 3, 4]):
    km = KMeans(n_clusters=k, random_state=42)
    labs = km.fit_predict(X)

    # scatter row
    ax1 = plt.subplot(2, 3, idx + 1)
    colors = cm.tab10(labs.astype(float) / k)
    ax1.scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.6, edgecolor='k')
    ax1.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                c='red', marker='X', s=200)
    ax1.set_title(f"k={k}")

    # silhouette row
    ax2 = plt.subplot(2, 3, idx + 4)
    evaluate_clustering(X, labs, k, ax=ax2, title_suffix=f"k={k}")

plt.tight_layout()
plt.savefig("k_comparison.png")
plt.close()

# -----------------------------------------------------------
# 8.  Shape sensitivity demo (make_classification)
# -----------------------------------------------------------
X_clf, y_clf = make_classification(
    n_samples=300, n_features=2, n_informative=2, n_redundant=0,
    n_classes=3, n_clusters_per_class=1, random_state=42
)

km_clf = KMeans(n_clusters=3, random_state=42)
y_km_clf = km_clf.fit_predict(X_clf)

vor = Voronoi(km_clf.cluster_centers_)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
x_min, x_max = X_clf[:, 0].min() - 1, X_clf[:, 0].max() + 1
y_min, y_max = X_clf[:, 1].min() - 1, X_clf[:, 1].max() + 1

# true labels
ax = axes[0, 0]
ax.scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, cmap='tab10', alpha=0.5, ec='k')
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='red', line_alpha=0.6)
ax.set_title("True labels + Voronoi")
ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

# k-means labels
ax = axes[0, 1]
ax.scatter(X_clf[:, 0], X_clf[:, 1], c=y_km_clf, cmap='tab10', alpha=0.5, ec='k')
ax.scatter(km_clf.cluster_centers_[:, 0], km_clf.cluster_centers_[:, 1],
           c='red', marker='x', s=200)
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='red', line_alpha=0.6)
ax.set_title("k-means + Voronoi")
ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

# silhouette true
evaluate_clustering(X_clf, y_clf, 3, ax=axes[1, 0], title_suffix="true labels")

# silhouette k-means
evaluate_clustering(X_clf, y_km_clf, 3, ax=axes[1, 1], title_suffix="k-means")

plt.tight_layout()
plt.savefig("shape_sensitivity.png")
plt.close()

print("All figures saved: blobs_k4.png, stability_runs.png, metrics_vs_k.png, k_comparison.png, shape_sensitivity.png")