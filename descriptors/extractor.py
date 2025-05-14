# extractor.py - Extract quantum descriptors from Gaussian log files
import re

def extract_homo_lumo(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    matches = re.findall(r"Population.*?SCF [Dd]ensity.*?(\s+Alpha.*?)\n\s*Condensed", content, re.DOTALL)
    if not matches:
        return None, None
    scf_section = matches[-1]

    energies_alpha = [re.findall(r"([-+]?\d*\.\d+|\d+)", s) for s in scf_section.split("Alpha virt.", 1)]
    if len(energies_alpha) == 2:
        occupied, unoccupied = [list(map(float, e)) for e in energies_alpha]
        return max(occupied) if occupied else None, min(unoccupied) if unoccupied else None
    return None, None

def extract_dipole_moment(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    matches = re.findall(r"Dipole moment \(field-independent basis, Debye\):.*?(X=.*?Tot=.*?)\n", content, re.DOTALL)
    if not matches:
        return None
    tot_match = re.search(r"Tot=\s*([-+]?\d*\.\d+|\d+)", matches[-1])
    return float(tot_match.group(1)) if tot_match else None

def extract_polarizability(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    matches = re.findall(r"Exact polarizability:\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)", content)
    if not matches:
        return None
    values = [float(matches[-1][i]) for i in [0, 2, 5]]
    return sum(values) / len(values)

def extract_nbo_section(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r"Natural Bond Orbitals \(Summary\):(.*?)(-+\n)", content, re.DOTALL)
    return match.group(1) if match else None
