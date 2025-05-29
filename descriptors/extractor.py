import os
import re
import glob
import math
import pandas as pd
from morfeus import read_xyz, Sterimol
from morfeus.utils import get_radii

# ===== 共用常數 =====
atomic_symbols = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}


# ===== HOMO / LUMO =====
def extract_homo_lumo(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    matches = re.findall(r"Population.*?SCF [Dd]ensity.*?(\s+Alpha.*?)\n\s*Condensed", content, re.DOTALL)
    if not matches:
        print(f"❌ SCF Density section not found in {log_file}")
        return None, None

    scf_section = matches[-1]
    split_parts = scf_section.split("Alpha virt.", 1)

    if len(split_parts) != 2:
        return None, None

    occ_energies = list(map(float, re.findall(r"-?\d+\.\d+", split_parts[0])))
    virt_energies = list(map(float, re.findall(r"-?\d+\.\d+", split_parts[1])))

    if not occ_energies or not virt_energies:
        return None, None

    return max(occ_energies), min(virt_energies)


# ===== Dipole / Polarizability =====
def extract_dipole_moment(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    match = re.search(r"Tot=\s*([-+]?\d*\.\d+)", content)
    return float(match.group(1)) if match else None


def extract_polarizability(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    matches = re.findall(r"Exact polarizability:\s+([-+]?\d*\.\d+)", content)
    values = list(map(float, matches))
    return sum(values[i] for i in [0, 2, 5]) / 3 if len(values) >= 6 else None


# ===== NBO 分析 =====
def extract_nbo_section(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r"Natural Bond Orbitals \(Summary\):(.*?)-+\n", content, re.DOTALL)
    return match.group(1) if match else None


def find_oh_bonds(nbo_section):
    return [(int(a), int(b)) for a, b in re.findall(r"BD \( *1\) O\s*(\d+) - H\s*(\d+)", nbo_section)]


def find_c1_c2(nbo_section, oh_bond_atoms):
    for a, b in oh_bond_atoms:
        c_matches = re.findall(rf"BD \( *1\) C\s*(\d+) - O {a}", nbo_section)
        for c in c_matches:
            c = int(c)
            d_matches = re.findall(rf"BD \( *[12]\) C {c} - O (\d+)", nbo_section)
            for d in d_matches:
                d = int(d)
                e_matches = re.findall(rf"BD \( *1\) C (\d+) - C {c}", nbo_section)
                for e in e_matches:
                    e = int(e)
                    return c, e, a, b, d, None, None
    return None, None, None, None, None, None, None


def extract_nbo_values(log_file, c1, c2, a):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r"Natural Bond Orbitals \(Summary\):(.*?)-{30,}", content, re.DOTALL)
    if not match:
        return None, None, None, None
    section = match.group(1)

    def find_value(pattern):
        m = re.search(pattern, section)
        return (float(m.group(1)), float(m.group(2))) if m else (None, None)

    occ1, en1 = find_value(rf"BD \( *1\) C {c1} - O {a} +?([\d\.]+)\s+([-\d\.]+)")
    occ2, en2 = find_value(rf"BD \( *1\) C {c2} - C {c1} +?([\d\.]+)\s+([-\d\.]+)")
    return occ1, en1, occ2, en2


def extract_nbo_charges(log_file, c1, c2, a):
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start = next((i for i, l in enumerate(lines) if "Summary of Natural Population Analysis" in l), None)
    if start is None:
        raise ValueError("NBO Summary not found.")

    charges = {}
    for line in lines[start:]:
        m = re.match(r"\s*(\w+)\s+(\d+)\s+([-\d\.]+)", line)
        if m:
            atom, num, charge = m.groups()
            charges[f"{atom}{num}"] = float(charge)

    return (
        charges.get(f"C{c1}"),
        charges.get(f"C{c2}"),
        charges.get(f"O{a-1}"),
        charges.get(f"O{a}")
    )


# ===== Coordinates 與距離 =====
def extract_coordinates(log_file, c1, c2):
    coordinates = {}
    with open(log_file, 'r') as f:
        inside = False
        for line in f:
            if "Standard orientation" in line:
                coordinates.clear()
                inside = True
                continue
            if inside:
                if "----" in line: continue
                parts = line.split()
                if len(parts) == 6 and parts[0].isdigit():
                    idx = int(parts[0])
                    atomic_number = int(parts[1])
                    x, y, z = map(float, parts[3:])
                    if atomic_number == 6:
                        coordinates[idx] = (x, y, z)

    if c1 in coordinates and c2 in coordinates:
        x1, y1, z1 = coordinates[c1]
        x2, y2, z2 = coordinates[c2]
        return (x1, y1, z1), (x2, y2, z2), math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    return None, None, None


# ===== Frequencies 與 IR =====
def extract_frequencies(log_file, atom_c, atom_d):
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    def parse_floats(s): return list(map(float, re.findall(r"-?\d+\.\d+", s)))

    matched = []
    i = next((i for i, line in enumerate(lines) if "Frequencies --" in line), None)
    if i is None: raise ValueError("Frequencies block not found")

    while i < len(lines):
        if "Frequencies --" in lines[i]:
            freq_line = parse_floats(lines[i])
            ir_line = parse_floats(lines[i+3])
            displacements = []
            j = i + 5
            while j < len(lines) and "Frequencies --" not in lines[j] and lines[j].strip():
                parts = lines[j].split()
                if len(parts) >= 11:
                    disp1 = list(map(float, parts[2:5]))
                    disp2 = list(map(float, parts[5:8]))
                    disp3 = list(map(float, parts[8:11]))
                    displacements.append([disp1, disp2, disp3])
                j += 1

            for idx, freq in enumerate(freq_line):
                if 1800 <= freq <= 1900:
                    try:
                        v1 = displacements[atom_c-1][idx]
                        v2 = displacements[atom_d-1][idx]
                        disp = [(a-b)**2 for a, b in zip(v1, v2)]
                        matched.append((freq, ir_line[idx], math.sqrt(sum(disp))))
                    except:
                        continue
            i = j
        else:
            i += 1

    if not matched:
        raise ValueError("No valid frequencies found")
    matched.sort(key=lambda x: x[2], reverse=True)
    return matched[0][1], matched[0][0]


# ===== Sterimol 工具 =====
def extract_last_standard_orientation(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()

    geometries, block = [], []
    reading = False

    for line in lines:
        if "Standard orientation" in line:
            block = []
            reading = True
            continue
        if reading:
            if "-----" in line or any(k in line for k in ["Center", "Atomic", "Number"]):
                continue
            if not line.strip():
                if block:
                    geometries.append(block)
                    block = []
                reading = False
            else:
                if re.match(r"^\s*\d+\s+\d+\s+\d+\s+[-+]?\d*\.\d+", line):
                    block.append(line)

    if not geometries:
        return None

    atoms = []
    for line in geometries[-1]:
        parts = line.split()
        try:
            atomic_num = int(parts[1])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            symbol = atomic_symbols.get(atomic_num)
            if symbol is None:
                return None
            atoms.append((symbol, x, y, z))
        except:
            return None

    return atoms


def write_xyz(atom_list, filename):
    with open(filename, "w") as f:
        f.write(f"{len(atom_list)}\nExtracted from log\n")
        for atom in atom_list:
            f.write(f"{atom[0]}  {atom[1]:.6f}  {atom[2]:.6f}  {atom[3]:.6f}\n")


def compute_sterimol_parameters(excel_path, log_folder):
    df = pd.read_excel(excel_path)
    df["Ar_Ster_L"], df["Ar_Ster_B1"], df["Ar_Ster_B5"] = None, None, None

    log_map = {os.path.basename(f).replace(".log", ""): f for f in glob.glob(os.path.join(log_folder, "*.log"))}

    for idx, row in df.iterrows():
        name = str(row["Ar"])
        path = log_map.get(name)
        if not path:
            print(f"{name}.log not found")
            continue

        atoms = extract_last_standard_orientation(path)
        if not atoms:
            continue

        try:
            exclude = [int(row["Ar_a"]), int(row["Ar_b"]), int(row["Ar_d"])]
            filtered = [a for i, a in enumerate(atoms) if (i+1) not in exclude]
            xyz_path = f"{name}_filtered.xyz"
            write_xyz(filtered, xyz_path)

            elements, coords = read_xyz(xyz_path)
            radii = get_radii(elements, radii_type="bondi")
            radii = [1.09 if r == 1.20 else r for r in radii]

            atom1 = int(row["Ar_c"])
            atom2 = int(row["Ar_e"])
            sterimol = Sterimol(elements, coords, atom1, atom2, radii)

            df.at[idx, "Ar_Ster_L"] = sterimol.L_value
            df.at[idx, "Ar_Ster_B1"] = sterimol.B_1_value
            df.at[idx, "Ar_Ster_B5"] = sterimol.B_5_value
        except Exception as e:
            print(f"Sterimol error on {name}: {e}")

    output_path = "updated_data.xlsx"
    df.to_excel(output_path, index=False)
    print(f"✅ Sterimol 計算完成，已儲存至：{output_path}")
    return output_path