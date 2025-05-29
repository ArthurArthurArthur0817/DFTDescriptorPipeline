import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_data(df, feature_list, target):
    """
    清理與標準化 df 中的特徵欄位，回傳建模用 DataFrame

    Args:
        df: 原始 DataFrame（由 extractor 回傳）
        feature_list: 要使用的特徵欄位名稱 list
        target: 要預測的標的變數欄位名稱

    Returns:
        DataFrame: 包含標準化特徵與目標變數的乾淨資料
    """
    df_clean = df[feature_list + [target]].dropna()
    if df_clean.empty:
        raise ValueError("⚠️ 沒有足夠的資料可用於建模（可能全是 NaN）")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[feature_list])
    df_scaled = pd.DataFrame(X_scaled, columns=feature_list)
    df_scaled[target] = df_clean[target].values

    return df_scaled