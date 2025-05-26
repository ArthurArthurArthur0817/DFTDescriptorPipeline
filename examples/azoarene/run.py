from descriptors.aggregate import generate_feature_table
from descriptors.regression import search_best_models, prepare_data

import os

folder = os.path.dirname(__file__)
log_folder = os.path.join(folder, "logfiles")
excel_path = os.path.join(folder, "data.xlsx")

# 產出特徵表格
df = generate_feature_table(log_folder, excel_path)

# 選定目標欄位
target = "ddG"
features = [
    'Ar_NBO_C2', 'Ar_NBO_=O', 'Ar_NBO_-O',
    'Ar_v_C=O', 'Ar_I_C=O', 'Ar_dp',
    'Ar_polar', 'Ar_LUMO', 'Ar_HOMO',
    'Ar_Ster_L', 'Ar_Ster_B1', 'Ar_Ster_B5',
    'L_C1_C2'
]

df = prepare_data(df, features, target)
results = search_best_models(df, features, target, max_features=5)

print("Best model found:")
print(results[0])
