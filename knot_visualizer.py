import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
from sklearn.decomposition import PCA
import umap
from matplotlib.lines import Line2D

SIGNATURE_COLORS = {
    0: 'green',
    2: 'cyan',
    4: 'orange',
    6: 'pink',
    8: 'purple',
    10: 'blue',
    12: 'sienna'
}

class PCAVisualizer:
    def __init__(self, input_vectors, signatures, color_map):
        self.input_vectors = input_vectors
        self.signatures = signatures.squeeze().ravel()
        self.color_map = color_map

    def plot(self, title_prefix="Latent"):
        pca_result = PCA(n_components=2).fit_transform(self.input_vectors)

        plt.figure(figsize=(8, 6))
        for sig, color in self.color_map.items():
            idx = self.signatures == sig
            plt.scatter(pca_result[idx, 0], pca_result[idx, 1],
                        color=color, label=f"Signature {sig}", s=10, alpha=0.7)

        plt.title(f"PCA of {title_prefix} Space\nColored by Signature")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(title="Signature", loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class UMAPVisualizer:
    def __init__(self, input_vectors, signatures, mse_per_sample, color_map):
        self.input_vectors = input_vectors
        self.signatures = signatures.squeeze().ravel()
        self.mse = mse_per_sample
        self.color_map = color_map
        self.highlight_signatures = [8, 10, 12]

    def plot(self, title_prefix="Latent"):
        reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, random_state=42)
        umap_result = reducer.fit_transform(self.input_vectors)

        sizes = np.log10(self.mse + 1) * 1000
        colors = [self.color_map.get(sig, 'gray') for sig in self.signatures]

        plt.figure(figsize=(10, 6))
        plt.scatter(umap_result[:, 0], umap_result[:, 1], c=colors, s=sizes, alpha=0.7)
        plt.title(f"UMAP of {title_prefix} Space\nColor: Signature | Size: Reconstruction Error")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=f'Signature {sig}',
                   markerfacecolor=color, markersize=8, alpha=0.9)
            for sig, color in self.color_map.items()
        ]
        plt.legend(handles=legend_elements, title="Signature", loc="best")

        highlight_idx = [i for i, sig in enumerate(self.signatures) if sig in self.highlight_signatures]
        plt.scatter(umap_result[highlight_idx, 0],
                    umap_result[highlight_idx, 1],
                    c=[self.color_map[self.signatures[i]] for i in highlight_idx],
                    s=np.log10(self.mse[highlight_idx] + 1) * 1000,
                    linewidth=0.8, alpha=1.0)

        plt.tight_layout()
        plt.show()

class ErrorVisualizer:
    def __init__(self, errors, signatures, color_map):
        self.errors = errors
        self.signatures = signatures.squeeze().ravel()
        self.color_map = color_map

    def plot_loglog_error(self, rank_min=50, rank_max=5000):
        errors_sorted = np.sort(self.errors)[::-1]
        ranks = np.arange(1, len(errors_sorted) + 1)
        log_rank = np.log10(ranks)
        log_error = np.log10(errors_sorted)

        mask = (ranks > rank_min) & (ranks < rank_max)
        slope, intercept, r_value, _, _ = linregress(log_rank[mask], log_error[mask])
        alpha = -slope
        r2 = r_value**2
        fit_line = 10 ** (intercept + slope * log_rank)

        plt.figure(figsize=(7, 5))
        plt.scatter(ranks, errors_sorted, s=6, color="#3182bd", alpha=0.7, label="samples")
        plt.plot(ranks, fit_line, "r--", lw=2,
                 label=fr"fit (rank {rank_min}–{rank_max}): $\alpha={alpha:.2f}$, $R^2={r2:.2f}$")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Error rank (log)")
        plt.ylabel("MSE error (log)")
        plt.title("Log–Log Distribution of Prediction Error per Sample", weight="bold")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

    def plot_violin(self):
        df_plot = pd.DataFrame({
            'Signature': self.signatures,
            'Error MSE': self.errors
        })

        unique_sigs = sorted(df_plot['Signature'].unique())
        color_palette = [self.color_map.get(sig, 'gray') for sig in unique_sigs]

        plt.figure(figsize=(12, 7))
        sns.violinplot(x='Signature', y='Error MSE', data=df_plot,
                       inner="quartile", scale='width', linewidth=1.2,
                       palette=color_palette)
        sns.pointplot(x='Signature', y='Error MSE', data=df_plot,
                      estimator='mean', color='darkred',
                      linestyles='--', markers='o', scale=0.6, errwidth=1.5)
        plt.title('Prediction Error Distribution by Knot Signature', fontsize=16, weight='bold')
        plt.xlabel('Knot Signature')
        plt.ylabel('HOMFLY Prediction MSE')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()