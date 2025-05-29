import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

def prepare_data(path, features, target):
    """
    讀取 Excel 並標準化特徵欄位。
    """
    data = pd.read_excel(path)
    data = data.dropna(subset=features + [target])
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    return data


def compute_loocv_metrics(X, y):
    """
    使用影響矩陣進行 LOOCV，計算 Q²、R²、RMSE。
    """
    n = X.shape[0]
    X_design = np.hstack([np.ones((n, 1)), X])  # 加入截距
    XtX_inv = np.linalg.inv(X_design.T @ X_design)
    beta = XtX_inv @ X_design.T @ y
    H = X_design @ XtX_inv @ X_design.T
    h = np.diag(H)
    y_pred = X_design @ beta
    y_loo = (y_pred - h * y) / (1 - h)

    ss_total = np.sum((y - np.mean(y))**2)
    ss_res_loocv = np.sum((y - y_loo)**2)
    ss_res_full = np.sum((y - y_pred)**2)

    return {
        "r2_full": 1 - ss_res_full / ss_total,
        "q2_loocv": 1 - ss_res_loocv / ss_total,
        "rmse": np.sqrt(np.mean((y - y_loo)**2)),
        "coefficients": beta[1:].tolist(),
        "intercept": beta[0]
    }


def evaluate_combinations(data, target, feature_set):
    X = data[feature_set].values
    y = data[target].values
    try:
        result = compute_loocv_metrics(X, y)
        result["features"] = feature_set
        return result if result["r2_full"] > 0.7 else None
    except np.linalg.LinAlgError:
        return None


def search_best_models(data, features, target, max_features=5, r2_threshold=0.8, n_jobs=-1):
    """
    測試所有特徵組合，篩選出表現優異的模型。
    """
    all_results = []
    for k in range(1, max_features + 1):
        combs = list(combinations(features, k))
        print(f" 測試 {k} 個特徵組合：共 {len(combs)} 種")
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_combinations)(data, target, list(c)) for c in combs
        )
        all_results.extend([res for res in results if res is not None])
    return sorted(all_results, key=lambda x: x["q2_loocv"], reverse=True)