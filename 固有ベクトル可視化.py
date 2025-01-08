import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import eigh

def create_adjacency_matrix(n):
    """グラフの隣接行列を作成します。"""
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    return A

def compute_laplacian(A):
    """隣接行列からラプラシアン行列を計算します。"""
    row_sums = np.sum(A, axis=1)
    D = np.diag(row_sums)
    L = D - A
    return L, D

def compute_normalized_laplacian(L, D):
    """ラプラシアン行列の正規化を行います。"""
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    L_normalized = np.dot(D_inv_sqrt, np.dot(L, D_inv_sqrt))
    return L_normalized

# ノード数
N = 50

# 隣接行列を作成し、ラプラシアン行列と正規化ラプラシアンを計算
A = create_adjacency_matrix(N)
L, D = compute_laplacian(A)
L_normalized = compute_normalized_laplacian(L, D)

# ラプラシアンの固有値と固有ベクトルを計算
eigenvalues, eigenvectors = eigh(L_normalized)

# グラフのレイアウトを直線上に設定（ノード番号を0からN-1まで振る）
pos = {i: (i + 1, 0) for i in range(N)}

# サブプロットの行と列を計算（5列で調整）
cols = 5
rows = (N + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
axes = axes.flatten()

for idx in range(N):
    ax = axes[idx]

    # 現在の固有ベクトル
    eigenvector = eigenvectors[:, idx]

    # ノードの描画
    nx.draw(
        nx.path_graph(N), 
        pos, 
        with_labels=True, 
        labels={i: i + 1 for i in range(N)},  # ノードラベルを表示
        node_color='lightblue', 
        node_size=500, 
        font_size=10, 
        ax=ax
    )

    # 固有ベクトルの各成分をノードに対応させて可視化
    for i in range(N):
        # ノード位置
        x, y = pos[i]
        # 矢印の描画（上：正、下：負）
        ax.arrow(x, y, 0, eigenvector[i], head_width=0.2, head_length=0.1, fc='r', ec='r')

    # ラベルをつける
    ax.set_title(f'Eigenvector {idx + 1}')

# 不使用のサブプロットを非表示にする
for idx in range(N, len(axes)):
    axes[idx].axis('off')

# 全体のラベルを設定
plt.suptitle('Laplacian Eigenvectors Visualization', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # タイトルが重ならないように調整
plt.show()
