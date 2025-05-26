# descriptors/extractor.py
import re
import math         # ✅ 加這行
import numpy as np
from utils import parse_floats

def extract_homo_lumo(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 找到所有 SCF Density Population analysis 的部分
    matches = re.findall(r"Population.*?SCF [Dd]ensity.*?(\s+Alpha.*?)\n\s*Condensed", content, re.DOTALL)

    if not matches:
        print("SCF Density Population analysis not found.")
        return None, None

    # 取最後一次出現的 SCF Density 來確保是最新的
    scf_section = matches[-1]

    # 解析 Alpha occ. 和 Alpha virt.
    energies_alpha = [re.findall(r"([-+]?\d*\.\d+|\d+)", s_part) for s_part in scf_section.split("Alpha virt.", 1)]
    if len(energies_alpha) == 2:
        occupied_energies_alpha, unoccupied_energies_alpha = [list(map(float, e)) for e in energies_alpha]

        # 提取 HOMO 和 LUMO
        homo_alpha = max(occupied_energies_alpha) if occupied_energies_alpha else None
        lumo_alpha = min(unoccupied_energies_alpha) if unoccupied_energies_alpha else None

        return homo_alpha, lumo_alpha

    print("HOMO/LUMO energies could not be extracted.")
    return None, None

def extract_dipole_moment(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 找到所有 Dipole moment 的部分
    matches = re.findall(r"Dipole moment \(field-independent basis, Debye\):.*?(X=.*?Tot=.*?)\n", content, re.DOTALL)

    if not matches:
        print("Dipole moment not found.")
        return None

    # 取最後一次出現的 Dipole moment 數據
    last_dipole_section = matches[-1]

    # 提取 Tot 的數值
    tot_match = re.search(r"Tot=\s*([-+]?\d*\.\d+|\d+)", last_dipole_section)
    if tot_match:
        return float(tot_match.group(1))

    print("Total Dipole moment could not be extracted.")
    return None

def extract_polarizability(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 查找最後一次出現的 "Exact polarizability:" 及其後的數字
    matches = re.findall(r"Exact polarizability:\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)", content)

    if not matches:
        print("Exact polarizability not found.")
        return None

    # 取最後一次出現的數據
    last_polarizability = matches[-1]

    # 提取第一個、第三個和第六個數值並計算平均值
    values = [float(last_polarizability[i]) for i in [0, 2, 5]]
    avg_polarizability = sum(values) / len(values)

    return avg_polarizability

def extract_nbo_section(log_file):
    """ 讀取 log 檔並提取 NBO Summary 區塊 """
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    match = re.search(r"Natural Bond Orbitals \(Summary\):(.*?)(-+\n)", content, re.DOTALL)
    if not match:
        print("NBO summary section not found.")
        return None

    return match.group(1)


#=============================================================================================================================


import re

def extract_nbo_values(log_file, c1, c2, a):
    """ 提取 Occupancy 與 Energy 值 """
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    match = re.search(r"Natural Bond Orbitals \(Summary\):(.*?)-{30,}", content, re.DOTALL)
    if not match:
        print("NBO Summary section not found.")
        return None

    nbo_section = match.group(1)

    bond_patterns = {
        "C1-O": rf"BD \(   1\) C\s+{c1}\s+-\s+O\s+{a}\s+([\d\.]+)\s+([-\d\.]+)",
        "C1-C2": rf"BD \(   1\) C\s+{c2}\s+-\s+C\s+{c1}\s+([\d\.]+)\s+([-\d\.]+)"
    }

    # 初始化變數
    occupancy_C1_O = occupancy_C1_C2 = None
    energy_C1_O = energy_C1_C2 = None

    for key, pattern in bond_patterns.items():
        match = re.search(pattern, nbo_section)
        if match:
            if key == "C1-O":
                occupancy_C1_O = float(match.group(1))
                energy_C1_O = float(match.group(2))
            elif key == "C1-C2":
                occupancy_C1_C2 = float(match.group(1))
                energy_C1_C2 = float(match.group(2))

    return occupancy_C1_O, energy_C1_O, occupancy_C1_C2, energy_C1_C2

def extract_coordinates(log_file, c1, c2):
    """ 提取 C1, C2 的座標並計算歐幾里得距離 """
    coordinates = {}
    inside_standard_orientation = False

    with open(log_file, 'r') as file:
        for line in file:
            if "Standard orientation" in line:
                inside_standard_orientation = True
                continue
            if inside_standard_orientation:
                if "----" in line:
                    continue

                parts = line.split()

                if len(parts) == 6 and parts[0].isdigit() and parts[1].isdigit():
                    center_number = int(parts[0])
                    atomic_number = int(parts[1])
                    x, y, z = map(float, parts[3:])

                    if atomic_number == 6:
                        coordinates[center_number] = (x, y, z)

    if c1 in coordinates and c2 in coordinates:
        x1, y1, z1 = coordinates[c1]
        x2, y2, z2 = coordinates[c2]
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

        # 座標與距離結果存儲為變數
        coord_C1 = (x1, y1, z1)
        coord_C2 = (x2, y2, z2)
        euclidean_distance = distance

        return coord_C1, coord_C2, euclidean_distance
    else:
        print("Error: Could not find coordinates for both C1 and C2.")
        return None, None, None

def extract_nbo_charges(log_file, c1, c2, a):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.readlines()

    summary_index = None
    for i in range(len(content) - 1, -1, -1):
        if "Summary of Natural Population Analysis" in content[i]:
            summary_index = i
            break

    if summary_index is None:
        raise ValueError(f"❌ Cannot find NBO summary in {log_file}")

    charges = {}
    for line in content[summary_index:]:
        match = re.match(r'\s*(\w+)\s+(\d+)\s+([-\d\.]+)', line)
        if match:
            atom, num, charge = match.groups()
            charges[f"{atom}{num}"] = float(charge)

    Ar_NBO_C1 = charges.get(f"C{c1}", None)
    Ar_NBO_C2 = charges.get(f"C{c2}", None)
    Ar_NBO_O1 = charges.get(f"O{a-1}", None)
    Ar_NBO_O2 = charges.get(f"O{a}", None)

    return Ar_NBO_C1, Ar_NBO_C2, Ar_NBO_O1, Ar_NBO_O2


def extract_frequencies(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.readlines()

    freq_block_start = None
    for i in range(len(content)):
        if re.search(r'\s+A\s+A\s+A', content[i]):
            if "Frequencies --" in content[i+1]:
                freq_block_start = i + 1
                break

    if freq_block_start is None:
        raise ValueError(f"未找到 {log_file} 的 Frequencies 區塊")

    matched_frequencies = []

    i = freq_block_start
    while i < len(content):
        if "Frequencies --" in content[i]:
            freq_line = content[i]
            red_mass_line = content[i + 1]
            ir_inten_line = content[i + 3]

            freqs = parse_floats(freq_line)
            red_masses = parse_floats(red_mass_line)
            ir_intensities = parse_floats(ir_inten_line)

            for f, m, ir in zip(freqs, red_masses, ir_intensities):
                if 1800 <= f <= 1900 and 10 <= m <= 11:
                    matched_frequencies.append((f, ir))

            i += 4
        else:
            i += 1

    if not matched_frequencies:
        print(f"⚠️ No frequencies found in {log_file}")
        return None
    # 根據 IR 強度由大到小排序，選最強的
    matched_frequencies.sort(key=lambda x: x[1], reverse=True)
    Ar_v_C_O, Ar_I_C_O = matched_frequencies[0]

    return Ar_I_C_O, Ar_v_C_O

def find_oh_bonds(nbo_section):
    """ 找出 BD (1) O a - H b 的鍵結，取得 O 和 H 的原子編號 """
    oh_bonds = re.findall(r"BD \(\s*1\s*\)\s*O\s*(\d+)\s*-\s*H\s*(\d+)", nbo_section)

    if not oh_bonds:
        print("No O-H bonds found.")
    return [(int(a), int(b)) for a, b in oh_bonds]

def find_c1_c2(nbo_section, oh_bond_atoms):
    for a,b in oh_bond_atoms:
        c_candidates = re.findall(rf"BD \(\s*1\s*\)\s*C\s*(\d+)\s*-\s*O\s*{a}", nbo_section)

        for c in c_candidates:
            c = int(c)
            o_d_candidates = re.findall(rf"BD \(\s*[12]\s*\)\s*C\s*{c}\s*-\s*O\s*(\d+)", nbo_section)

            for d in o_d_candidates:
                d = int(d)
                e_candidates = re.findall(rf"BD \(\s*1\s*\)\s*C\s*(\d+)\s*-\s*C\s*{c}", nbo_section)

                for e in e_candidates:
                    e = int(e)

                    # 搜尋與 e 相連的鍵結
                    bond_types = re.findall(rf"BD \(\s*(1|2)\s*\)\s*(\w+)\s*(\d+)\s*-\s*(\w+)\s*(\d+)", nbo_section)

                    single_bonds = []
                    double_bonds = []
                    bond_pairs = {}
                    e_neighbors = []

                    for bond_type, atom1, num1, atom2, num2 in bond_types:
                        num1, num2 = int(num1), int(num2)

                        # 只記錄與 e 有關的鍵
                        if num1 == e or num2 == e:
                            other = num2 if num1 == e else num1
                            e_neighbors.append((bond_type, other))

                            bond_pair = frozenset([num1, num2])
                            if bond_pair not in bond_pairs:
                                bond_pairs[bond_pair] = set()
                            bond_pairs[bond_pair].add(bond_type)

                    # 統計單鍵雙鍵數量
                    single_count = sum("1" in types for types in bond_pairs.values())
                    double_count = sum("2" in types for types in bond_pairs.values())

                    if single_count >= 2 and double_count >= 1:
                        # 進一步找出 f 與 g
                        f, g = None, None
                        single_neighbors = [n for t, n in e_neighbors if t == "1"]
                        double_neighbors = [n for t, n in e_neighbors if t == "2"]

                        for neighbor in single_neighbors:
                            if f is None:
                                f = neighbor
                            elif g is None and neighbor != f:
                                g = neighbor

                        for neighbor in double_neighbors:
                            if g is None or neighbor == f:
                                g = neighbor

                        print(f"Found C1: {c}, C2: {e}, A: {a},B: {b}, D: {d}, F: {f}, G: {g}")
                        return c, e, a, b, d, f, g

    return None, None, None, None, None, None, None


#=============================================================================================================================