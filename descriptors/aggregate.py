# -*- coding: utf-8 -*-
import os
import re
import math
import pandas as pd
from morfeus import read_xyz, Sterimol
from morfeus.utils import get_radii

# ------------------------
# 解析用工具
# ------------------------
def extract_homo_lumo(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    matches = re.findall(r"Population.*?SCF [Dd]ensity.*?(\s+Alpha.*?)\n\s*Condensed", content, re.DOTALL)
    if not matches:
        return None, None
    scf_section = matches[-1]
    energies_alpha = [re.findall(r"([-+]?\d*\.\d+|\d+)", s) for s in scf_section.split("Alpha virt.", 1)]
    if len(energies_alpha) == 2:
        occ, virt = [list(map(float, e)) for e in energies_alpha]
        return max(occ) if occ else None, min(virt) if virt else None
    return None, None

def extract_dipole_moment(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    matches = re.findall(r"Dipole moment \(field-independent basis, Debye\):.*?(X=.*?Tot=.*?)\n", content, re.DOTALL)
    if not matches:
        return None
    match = re.search(r"Tot=\s*([-+]?\d*\.\d+|\d+)", matches[-1])
    return float(match.group(1)) if match else None

def extract_polarizability(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    matches = re.findall(r"Exact polarizability:\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)", content)
    if not matches:
        return None
    values = list(map(float, matches[-1]))
    return sum(values) / len(values)

def extract_nbo_section(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r"Natural Bond Orbitals \(Summary\):(.*?)(-+\n)", content, re.DOTALL)
    return match.group(1) if match else None

def find_oh_bonds(nbo_section):
    matches = re.findall(r"BD \(\s*1\s*\)\s*O\s*(\d+)\s*-\s*H\s*(\d+)", nbo_section)
    return [(int(a), int(b)) for a, b in matches]

def find_c1_c2(nbo_section, oh_atoms):
    for a, b in oh_atoms:
        c_candidates = re.findall(rf"BD \(\s*1\s*\)\s*C\s*(\d+)\s*-\s*O\s*{a}", nbo_section)
        for c in map(int, c_candidates):
            e_candidates = re.findall(rf"BD \(\s*1\s*\)\s*C\s*(\d+)\s*-\s*C\s*{c}", nbo_section)
            for e in map(int, e_candidates):
                d_match = re.search(rf"BD \(\s*1\s*\)\s*C\s*{c}\s*-\s*O\s*(\d+)", nbo_section)
                d = int(d_match.group(1)) if d_match else None
                f = g = None
                return c, e, a, b, d, f, g
    return None, None, None, None, None, None, None

def extract_nbo_values(log_file, c1, c2, a):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    section = re.search(r"Natural Bond Orbitals \(Summary\):(.*?)-{30,}", content, re.DOTALL)
    if not section:
        return None, None, None, None
    nbo_section = section.group(1)
    occ1 = re.search(rf"BD \(   1\) C\s+{c1}\s+-\s+O\s+{a}\s+([\d\.]+)\s+([-\d\.]+)", nbo_section)
    occ2 = re.search(rf"BD \(   1\) C\s+{c2}\s+-\s+C\s+{c1}\s+([\d\.]+)\s+([-\d\.]+)", nbo_section)
    return (
        float(occ1.group(1)) if occ1 else None,
        float(occ1.group(2)) if occ1 else None,
        float(occ2.group(1)) if occ2 else None,
        float(occ2.group(2)) if occ2 else None
    )

def extract_nbo_charges(log_file, c1, c2, a):
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    idx = next((i for i in range(len(lines)-1, -1, -1) if "Summary of Natural Population Analysis" in lines[i]), None)
    if idx is None:
        raise ValueError("NPA section not found")
    charges = {}
    for line in lines[idx:]:
        match = re.match(r'\s*(\w+)\s+(\d+)\s+([-\d\.]+)', line)
        if match:
            k = f"{match.group(1)}{match.group(2)}"
            charges[k] = float(match.group(3))
    return charges.get(f"C{c1}"), charges.get(f"C{c2}"), charges.get(f"O{a-1}"), charges.get(f"O{a}")

def extract_coordinates(log_file, c1, c2):
    coordinates = {}
    with open(log_file, 'r') as f:
        inside = False
        for line in f:
            if "Standard orientation" in line:
                inside = True
                continue
            if inside:
                if "--" in line or "Center" in line:
                    continue
                parts = line.split()
                if len(parts) >= 6 and parts[0].isdigit():
                    center = int(parts[0])
                    atomic = int(parts[1])
                    if atomic == 6:
                        coordinates[center] = tuple(map(float, parts[3:6]))
    if c1 in coordinates and c2 in coordinates:
        x1, y1, z1 = coordinates[c1]
        x2, y2, z2 = coordinates[c2]
        return coordinates[c1], coordinates[c2], math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    return None, None, None

def extract_frequencies(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "Frequencies --" in line and re.search(r"A\s+A\s+A", lines[i-1]):
            freqs = list(map(float, re.findall(r'-?\d+\.\d+', line)))
            mass = list(map(float, re.findall(r'-?\d+\.\d+', lines[i+1])))
            inten = list(map(float, re.findall(r'-?\d+\.\d+', lines[i+3])))
            for f, m, ir in zip(freqs, mass, inten):
                if 1800 <= f <= 1900 and 10 <= m <= 11:
                    return ir, f
    raise ValueError("No suitable frequency found")

def extract_last_standard_orientation(log_path):
    atomic_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
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
            if "-----" in line or "Center" in line:
                continue
            if line.strip() == "":
                if block:
                    geometries.append(block)
                    block = []
                reading = False
            else:
                parts = line.split()
                if len(parts) >= 6:
                    block.append(parts)
    if not geometries:
        return None
    atoms = []
    for parts in geometries[-1]:
        symbol = atomic_symbols.get(int(parts[1]), None)
        if symbol:
            atoms.append((symbol, float(parts[3]), float(parts[4]), float(parts[5])))
    return atoms

def write_xyz(atom_list, filename):
    with open(filename, "w") as f:
        f.write(f"{len(atom_list)}\nExtracted from Gaussian log\n")
        for atom in atom_list:
            f.write(f"{atom[0]}  {atom[1]:.8f}  {atom[2]:.8f}  {atom[3]:.8f}\n")

# ------------------------
# 特徵主函數
# ------------------------
def generate_feature_table(log_folder, excel_path):
    df = pd.read_excel(excel_path)
    if "Ar" not in df.columns:
        if "Ar1" in df.columns:
            df["Ar"] = df["Ar1"].astype(str)
        else:
            raise ValueError("Need Ar or Ar1 column")

    for index, row in df.iterrows():
        name = str(row["Ar"])
        log_file = os.path.join(log_folder, f"{name}.log")
        if not os.path.exists(log_file):
            continue

        try:
            homo, lumo = extract_homo_lumo(log_file)
            dp = extract_dipole_moment(log_file)
            polar = extract_polarizability(log_file)
            nbo_section = extract_nbo_section(log_file)
            oh = find_oh_bonds(nbo_section)
            c1, c2, a, b, d, f, g = find_c1_c2(nbo_section, oh)
            charge_c1, charge_c2, charge_o1, charge_o2 = extract_nbo_charges(log_file, c1, c2, a)
            occ1, e1, occ2, e2 = extract_nbo_values(log_file, c1, c2, a)
            _, _, dist = extract_coordinates(log_file, c1, c2)
            ir, freq = extract_frequencies(log_file)
            atoms = extract_last_standard_orientation(log_file)
            atoms_filtered = [atom for i, atom in enumerate(atoms) if (i+1) not in [a, b, d]]
            xyz_path = f"/tmp/{name}_filtered.xyz"
            write_xyz(atoms_filtered, xyz_path)
            elements, coords = read_xyz(xyz_path)
            radii = get_radii(elements, radii_type="bondi")
            radii = [1.09 if r == 1.20 else r for r in radii]
            sterimol = Sterimol(elements, coords, c1, c2, radii=radii)
            df.at[index, "Ar_HOMO"] = homo
            df.at[index, "Ar_LUMO"] = lumo
            df.at[index, "Ar_dp"] = dp
            df.at[index, "Ar_polar"] = polar
            df.at[index, "Ar_NBO_C1"] = charge_c1
            df.at[index, "Ar_NBO_C2"] = charge_c2
            df.at[index, "Ar_NBO_=O"] = charge_o1
            df.at[index, "Ar_NBO_-O"] = charge_o2
            df.at[index, "Ar_I_C=O"] = ir
            df.at[index, "Ar_v_C=O"] = freq
            df.at[index, "L_C1_C2"] = dist
            df.at[index, "Ar_Ster_L"] = sterimol.L_value
            df.at[index, "Ar_Ster_B1"] = sterimol.B_1_value
            df.at[index, "Ar_Ster_B5"] = sterimol.B_5_value
        except Exception as e:
            print(f"Error in {name}: {e}")

    return df

if __name__ == "__main__":
    log_dir = "/content/logs"
    excel_path = "/content/Heck_boronic_acid.xlsx"
    df = generate_feature_table(log_dir, excel_path)
    df.to_excel("/content/output_updated.xlsx", index=False)
    print("✅ Done. Saved to output_updated.xlsx")