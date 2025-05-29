import os
import re
import numpy as np
import pandas as pd
from sterimol import calculate_sterimol_descriptors

def extract_coordinates(log_file, c1_index, c2_index):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    std_ori_idx = [i for i, line in enumerate(lines) if "Standard orientation:" in line]
    if not std_ori_idx:
        raise ValueError("Standard orientation section not found.")

    idx = std_ori_idx[-1]
    coord_lines = lines[idx + 5:]
    coords = {}
    for line in coord_lines:
        if "-----" in line:
            break
        parts = line.split()
        atom_idx = int(parts[0])
        x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
        coords[atom_idx] = np.array([x, y, z])

    coord1 = coords[int(c1_index)]
    coord2 = coords[int(c2_index)]
    dist = np.linalg.norm(coord1 - coord2)
    return coord1, coord2, dist

def extract_nbo_charges(log_file, c1_index, c2_index, a_index):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    nbo_idx = [i for i, line in enumerate(lines) if "Summary of Natural Population Analysis:" in line]
    if not nbo_idx:
        raise ValueError("NBO section not found.")

    idx = nbo_idx[-1]
    charge_lines = lines[idx + 4:]
    charges = {}
    for line in charge_lines:
        if not line.strip() or not re.match(r'\s*\d+\s+\w+\s+[-\d.]+', line):
            continue
        parts = line.strip().split()
        atom_idx = int(parts[0])
        charge = float(parts[2])
        charges[atom_idx] = charge

    return (
        charges.get(int(c1_index), None),
        charges.get(int(c2_index), None),
        charges.get(int(a_index), None),
        charges.get(int(a_index)+1, None)
    )

def extract_frequencies(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    freq_values = []
    intensity_values = []

    for line in lines:
        if "Frequencies --" in line:
            freqs = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            freq_values.extend([float(f) for f in freqs])
        if "IR Inten    --" in line:
            intensities = re.findall(r"[-+]?\d*\.\d+|\d+", line)
            intensity_values.extend([float(i) for i in intensities])

    if not freq_values or not intensity_values:
        raise ValueError("No frequencies or intensities found.")

    max_idx = np.argmax(intensity_values)
    return intensity_values[max_idx], freq_values[max_idx]

def extract_homo_lumo(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    homo = None
    lumo = None
    energies = []

    for line in lines:
        if "Alpha  occ. eigenvalues" in line or "Beta  occ. eigenvalues" in line:
            occ_vals = [float(x) for x in line.strip().split()[-5:]]
            energies.extend(occ_vals)
        if "Alpha virt. eigenvalues" in line or "Beta virt. eigenvalues" in line:
            virt_vals = [float(x) for x in line.strip().split()[-5:]]
            energies.extend(virt_vals)

    if not energies:
        raise ValueError("No orbital eigenvalues found.")

    sorted_energies = sorted(energies)
    midpoint = len(sorted_energies) // 2
    homo = sorted_energies[midpoint - 1]
    lumo = sorted_energies[midpoint]

    return homo, lumo

def process_log_files(log_folder, excel_path):
    df = pd.read_excel(excel_path)
    df["Ar_NBO_C1"] = None
    df["Ar_NBO_C2"] = None
    df["Ar_NBO_=O"] = None
    df["Ar_NBO_-O"] = None
    df["Ar_I_C=O"] = None
    df["Ar_v_C=O"] = None
    df["L_C1_C2"] = None
    df["Ar_HOMO"] = None
    df["Ar_LUMO"] = None
    df["Ar_Ster_L"] = None
    df["Ar_Ster_B1"] = None
    df["Ar_Ster_B5"] = None

    for i, row in df.iterrows():
        log_file = os.path.join(log_folder, str(row["Ar"]) + ".log")
        if not os.path.exists(log_file):
            print(f"⚠️ {log_file} not found.")
            continue

        print(f"🔍 處理中：{row['Ar']}.log")

        try:
            coord1, coord2, L = extract_coordinates(log_file, row["C1"], row["C2"])
            df.at[i, "L_C1_C2"] = L
        except Exception as e:
            print(f"❌ Coordinate error on {row['Ar']}: {e}")

        try:
            nbo_c1, nbo_c2, nbo_o, nbo_om = extract_nbo_charges(log_file, row["C1"], row["C2"], row["A"])
            df.at[i, "Ar_NBO_C1"] = nbo_c1
            df.at[i, "Ar_NBO_C2"] = nbo_c2
            df.at[i, "Ar_NBO_=O"] = nbo_o
            df.at[i, "Ar_NBO_-O"] = nbo_om
        except Exception as e:
            print(f"❌ NBO error on {row['Ar']}: {e}")

        try:
            ir_intensity, frequency = extract_frequencies(log_file)
            df.at[i, "Ar_I_C=O"] = ir_intensity
            df.at[i, "Ar_v_C=O"] = frequency
        except Exception as e:
            print(f"❌ Frequency error on {row['Ar']}: {e}")

        try:
            homo, lumo = extract_homo_lumo(log_file)
            df.at[i, "Ar_HOMO"] = homo
            df.at[i, "Ar_LUMO"] = lumo
        except Exception as e:
            print(f"❌ HOMO/LUMO error on {row['Ar']}: {e}")

        try:
            L, B1, B5 = calculate_sterimol_descriptors(log_file, row["C1"], row["C2"])
            df.at[i, "Ar_Ster_L"] = L
            df.at[i, "Ar_Ster_B1"] = B1
            df.at[i, "Ar_Ster_B5"] = B5
        except Exception as e:
            print(f"Sterimol error on {row['Ar']}: {e}")

    print("✅ 所有 log 處理完成，已產生 dataframe")
    return df