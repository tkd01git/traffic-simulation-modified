import pandas as pd
import numpy as np
from scipy.linalg import eigh

def calculate_F_lambda_global_rank(df, L_normalized):
    """
    各行のF_lambda列（F_lambda1からF_lambda40）の絶対値を計算し、
    全てのF_lambdaに対して順位を付けて出力。
    """
    global_ranks = []

    for _, row in df.iterrows():
        # 行データを取得
        df = np.array(row, dtype=float)

        # データ次元をラプラシアン行列の次元に合わせる
        if len(df) != L_normalized.shape[0]:
            raise ValueError(f"データ次元({len(df)})がラプラシアン行列の次元({L_normalized.shape[0]})と一致しません。")

        # ラプラシアン行列の固有値と固有ベクトルを計算
        eigenvalues, eigenvectors = eigh(L_normalized, eigvals_only=False)

        # 固有値を昇順にソート
        sorted_indices = np.argsort(eigenvalues)
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # 各固有値に対応するフーリエ係数 F_lambda を計算
        F_lambda = np.zeros(len(sorted_eigenvectors[0]), dtype=complex)
        for i in range(len(sorted_eigenvectors[0])):
            u_lambda_i = sorted_eigenvectors[:, i]
            F_lambda[i] = np.dot(df, u_lambda_i.conjugate())

        # F_lambda の絶対値を計算し、ランキングを決定
        abs_F_lambda = np.abs(F_lambda)
        rank = abs_F_lambda.argsort()[::-1].argsort() + 1  # 降順で1位を最大値に
        global_ranks.append(rank)

    return global_ranks

def write_sheet_with_ranks(writer, sheet_name, time_data, speed_data, global_ranks):
    """
    シートにデータを書き込む関数。
    """
    max_len = len(time_data)
    output_data = {
        'Time': list(time_data),
        'speed': list(speed_data)
    }

    # ランキングを各列に書き込む
    for lambda_index in range(40):
        output_data[f'Rank of F_lambda{lambda_index+1}'] = [
            ranks[lambda_index] if len(ranks) > lambda_index else '' for ranks in global_ranks
        ]

    # DataFrameに変換して書き込み
    results_df = pd.DataFrame(output_data)
    results_df.to_excel(writer, index=False, sheet_name=sheet_name)

# 入力ファイルと出力ファイルの設定
file_path = "prodata91.xlsx"
output_file = "λrankresult91.xlsx"

# データの読み込み
sheet_name1 = 'Sheet1'
df = pd.read_excel(file_path, sheet_name=sheet_name1, header=None, skiprows=1, nrows=40).values.flatten()
time_data = pd.read_excel(file_path, sheet_name=sheet_name1, usecols=[0], nrows=1).values.flatten()
speed_data = pd.read_excel(file_path, sheet_name=sheet_name1, usecols=[99], nrows=1).values.flatten()

# グラフのラプラシアン行列を作成
n = 40
A = np.zeros((n, n))
for i in range(n - 1):
    A[i, i + 1] = 1
    A[i + 1, i] = 1
L = np.diag(np.sum(A, axis=1)) - A
L_normalized = np.dot(np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1)))), 
                      np.dot(L, np.linalg.inv(np.sqrt(np.diag(np.sum(A, axis=1))))))

# F_lambdaランキングを計算
global_ranks = calculate_F_lambda_global_rank(df, L_normalized)

# シートに分けて保存
with pd.ExcelWriter(output_file, mode='w') as writer:
    write_sheet_with_ranks(writer, sheet_name1, time_data, speed_data, global_ranks)

print(f"{output_file} に固有値のグローバルランキングを含むデータを保存しました。")
