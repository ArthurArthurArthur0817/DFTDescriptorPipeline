import os
import pandas as pd
from morfeus import read_xyz, sterimol

# Atomic number to symbol mapping
atomic_symbols = {
    1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

def extract_last_standard_orientation(log_path):
    import re
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
                if re.match(r"^\s*\d+\s+\d+\s+\d+\s+[-+]?\d*\.\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", line):
                    block.append(line)
                else:
                    if block:
                        geometries.append(block)
                        block = []
                    reading = False

    if not geometries:
        return None

    last_geom = geometries[-1]
    atoms = []
    for line in last_geom:
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
    with open(filename, "w") as f:
        f.write(f"{len(atom_list)}\n")
        f.write("Extracted from Gaussian log\n")
        for atom in atom_list:
            f.write(f"{atom[0]}  {atom[1]:.8f}  {atom[2]:.8f}  {atom[3]:.8f}\n")

def compute_sterimol_parameters(excel_path, log_folder):
    df = pd.read_excel(excel_path)
    L_vals, B1_vals, B5_vals = [], [], []

    for i, row in df.iterrows():
        compound = row['Ar']
        log_path = os.path.join(log_folder, f"{compound}.log")
        atoms = extract_last_standard_orientation(log_path)

        if atoms is None:
            print(f"Sterimol error on {compound}: invalid geometry")
            L_vals.append(None)
            B1_vals.append(None)
            B5_vals.append(None)
            continue

        xyz_path = os.path.join(log_folder, f"{compound}.xyz")
        write_xyz(atoms, xyz_path)

        try:
            mol = read_xyz(xyz_path)
            L, B1, B5 = sterimol(mol, 1, 2)  # Assuming atom 1 and 2 as bond axis
            L_vals.append(L)
            B1_vals.append(B1)
            B5_vals.append(B5)
        except Exception as e:
            print(f"Sterimol error on {compound}: {e}")
            L_vals.append(None)
            B1_vals.append(None)
            B5_vals.append(None)

    df["Ar_Ster_L"] = L_vals
    df["Ar_Ster_B1"] = B1_vals
    df["Ar_Ster_B5"] = B5_vals

    output_path = os.path.join(log_folder, "updated_data.xlsx")
    df.to_excel(output_path, index=False)
    print(f"✅ Sterimol 計算完成，已儲存至：{output_path}")
    return output_path