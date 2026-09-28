"""PCA: понижение размерности с анализом."""
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

# === 1. Данные ===
digits = load_digits()
X, y = digits.data, digits.target
print(f"Исходная размерность: {X.shape}")

# === 2. PCA ===
pca = PCA(n_components=0.95)  # сохранить 95% дисперсии
X_pca = pca.fit_transform(X)
print(f"После PCA: {X_pca.shape}")
print(f"Компонент: {pca.n_components_}")
print(f"Суммарная дисперсия: {pca.explained_variance_ratio_.sum():.4f}")

# === 3. График explained variance ===
pca_full = PCA().fit(X)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(10, 5))
plt.plot(cumsum, marker='o')
plt.axhline(0.95, color='red', linestyle='--', label='95%')
plt.xlabel("Число компонент"); plt.ylabel("Накопленная дисперсия")
plt.title("PCA: Explained Variance")
plt.legend()
plt.grid(True)
plt.savefig("pca_variance.png")
plt.show()

# === 4. Сравнение моделей ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
score_original = cross_val_score(clf, X, y, cv=3).mean()
score_pca = cross_val_score(clf, X_pca, y, cv=3).mean()

print(f"Точность без PCA: {score_original:.4f}")
print(f"Точность с PCA:  {score_pca:.4f}")

# === 5. Визуализация 2D ===
X_2d = PCA(n_components=2).fit_transform(X)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar(scatter)
plt.title("PCA: 2D проекция")
plt.savefig("pca_2d.png")
plt.show()
