import os
import pandas as pd
from extractor import (
    extract_homo_lumo,
    extract_dipole_moment,
    extract_polarizability,
    extract_nbo_section,
    find_oh_bonds,
    find_c1_c2,
    extract_nbo_values,
    extract_coordinates,
    extract_nbo_charges,
    extract_frequencies
)

def generate_feature_table(log_folder, excel_path):
    """
    從指定的 Excel 與 log 資料夾中提取分子參數並整合至新 Excel。
    
    Args:
        excel_path (str): 原始 Excel 路徑，應包含 Ar 欄位（分子名稱）
        log_folder (str): .log 檔案所在資料夾路徑

    Returns:
        output_path (str): 更新後 Excel 檔案的輸出路徑
    """
    df = pd.read_excel(excel_path)

    for index, row in df.iterrows():
        ar = str(row["Ar"])
        log_file = os.path.join(log_folder, f"{ar}.log")

        if not os.path.exists(log_file):
            print(f"❌ 跳過 {ar}.log（找不到檔案）")
            continue

        print(f"🔍 處理中：{ar}.log")
        
        # 初始化欄位
        Ar_c = Ar_e = Ar_a = Ar_b = Ar_d = Ar_f = Ar_g = None
        Ar_NBO_C2 = Ar_NBO_O1 = Ar_NBO_O2 = Ar_I_C_O = Ar_v_C_O = None
        L_C1_C2 = homo = lumo = dipole_moment = avg_polar = None

        try:
            avg_polar = extract_polarizability(log_file)
            homo, lumo = extract_homo_lumo(log_file)
            dipole_moment = extract_dipole_moment(log_file)
            nbo_content = extract_nbo_section(log_file)

            if nbo_content:
                oh_atoms = find_oh_bonds(nbo_content)
                c1, c2, a, b, d, f, g = find_c1_c2(nbo_content, oh_atoms)
                Ar_c, Ar_e, Ar_a, Ar_b, Ar_d, Ar_f, Ar_g = c1, c2, a, b, d, f, g

                if c1 and c2 and a:
                    _, _, _, _ = extract_nbo_values(log_file, c1, c2, a)
                    Ar_NBO_C1, Ar_NBO_C2, Ar_NBO_O1, Ar_NBO_O2 = extract_nbo_charges(log_file, c1, c2, a)
                    Ar_I_C_O, Ar_v_C_O = extract_frequencies(log_file, c1, d)
                    _, _, L_C1_C2 = extract_coordinates(log_file, c1, c2)

        except Exception as e:
            print(f"⚠️ 解析錯誤於 {ar}.log: {e}")
            continue

        # 更新 DataFrame 欄位
        df.at[index, "Ar_NBO_C2"] = Ar_NBO_C2
        df.at[index, "Ar_NBO_=O"] = Ar_NBO_O1
        df.at[index, "Ar_NBO_-O"] = Ar_NBO_O2
        df.at[index, "Ar_v_C=O"] = Ar_v_C_O
        df.at[index, "Ar_I_C=O"] = Ar_I_C_O
        df.at[index, "Ar_dp"] = dipole_moment
        df.at[index, "Ar_polar"] = avg_polar
        df.at[index, "Ar_LUMO"] = lumo
        df.at[index, "Ar_HOMO"] = homo
        df.at[index, "L_C1_C2"] = L_C1_C2

        df.at[index, "Ar_c"] = Ar_c
        df.at[index, "Ar_e"] = Ar_e
        df.at[index, "Ar_a"] = Ar_a
        df.at[index, "Ar_b"] = Ar_b
        df.at[index, "Ar_d"] = Ar_d
        df.at[index, "Ar_f"] = Ar_f
        df.at[index, "Ar_g"] = Ar_g

    # 儲存結果
    
    return df