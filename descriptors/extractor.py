import os
import re
import glob
import pandas as pd
from morfeus import read_xyz, Sterimol
from morfeus.utils import get_radii

# 常見元素對應表
atomic_symbols = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}


def extract_last_standard_orientation(log_path):
    """
    解析 log 檔，提取最後一組 Standard orientation 幾何資訊
    """
    with open(log_path, "r") as f:
        lines = f.readlines()

    geometries = []
    block = []
    reading = False

    for line in lines:
        if "Standard orientation" in line:
            block = []
            reading = True
            continue
        if reading:
            if "-----" in line:
                continue
            if any(x in line for x in ["Center", "Atomic", "Number"]):
                continue
            if line.strip() == "":
                if block:
                    geometries.append(block)
                    block = []
                reading = False
            else:
                if re.match(r"^\s*\d+\s+\d+\s+\d+\s+[-+]?\d*\.\d+", line):
                    block.append(line)
                else:
                    if block:
                        geometries.append(block)
                        block = []
                    reading = False

    if not geometries:
        return None

    atoms = []
    for line in geometries[-1]:
        parts = line.split()
        try:
            atomic_num = int(parts[1])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            symbol = atomic_symbols.get(atomic_num, None)
            if symbol is None:
                print(f"Unknown atomic number {atomic_num} in {log_path}")
                return None
            atoms.append((symbol, x, y, z))
        except Exception as e:
            print(f"Parse error in {log_path}: {e}")
            return None

    return atoms


def write_xyz(atom_list, filename):
    """
    將原子座標寫入 XYZ 檔案
    """
    with open(filename, "w") as f:
        f.write(f"{len(atom_list)}\n")
        f.write("Extracted from Gaussian log\n")
        for atom in atom_list:
            f.write(f"{atom[0]}  {atom[1]:.8f}  {atom[2]:.8f}  {atom[3]:.8f}\n")


def compute_sterimol_parameters(excel_path, log_folder):
    """
    對每個分子計算 Sterimol 參數並回寫至 Excel。
    
    Args:
        excel_path (str): 包含 Ar, Ar_c, Ar_e 等欄位的 Excel 路徑
        log_folder (str): .log 檔所在資料夾

    Returns:
        str: 儲存的結果 Excel 檔案路徑
    """
    df = pd.read_excel(excel_path)

    # 建立欄位
    df["Ar_Ster_L"] = None
    df["Ar_Ster_B1"] = None
    df["Ar_Ster_B5"] = None

    log_files = glob.glob(os.path.join(log_folder, "*.log"))
    log_map = {os.path.basename(f).replace(".log", ""): f for f in log_files}

    for idx, row in df.iterrows():
        mol_name = str(row["Ar"])
        log_path = log_map.get(mol_name)

        if not log_path:
            print(f"❌ {mol_name}.log not found, skipping.")
            continue

        atoms = extract_last_standard_orientation(log_path)
        if not atoms:
            print(f"⚠️ {mol_name}: No geometry found.")
            continue

        try:
            exclude_atoms = [int(row["Ar_a"]), int(row["Ar_b"]), int(row["Ar_d"])]
            atoms_to_keep = [a for i, a in enumerate(atoms) if (i + 1) not in exclude_atoms]

            if len(atoms_to_keep) < 2:
                print(f"⚠️ {mol_name}: Too few atoms after exclusion.")
                continue

            xyz_path = f"{mol_name}_filtered.xyz"
            write_xyz(atoms_to_keep, xyz_path)

            atom1 = int(row["Ar_c"])
            atom2 = int(row["Ar_e"])

            elements, coords = read_xyz(xyz_path)
            radii = get_radii(elements, radii_type="bondi")
            radii = [1.09 if r == 1.20 else r for r in radii]  # 修正半徑過大問題

            sterimol = Sterimol(elements, coords, atom1, atom2, radii=radii)
            df.at[idx, "Ar_Ster_L"] = sterimol.L_value
            df.at[idx, "Ar_Ster_B1"] = sterimol.B_1_value
            df.at[idx, "Ar_Ster_B5"] = sterimol.B_5_value
            print(f"✅ {mol_name}: L={sterimol.L_value:.3f}, B1={sterimol.B_1_value:.3f}, B5={sterimol.B_5_value:.3f}")

        except Exception as e:
            print(f"❌ Error computing sterimol for {mol_name}: {e}")

    output_path = "updated_data.xlsx"
    df.to_excel(output_path, index=False)
    print(f"✅ Sterimol 計算完成，已儲存至：{output_path}")
    return output_path
