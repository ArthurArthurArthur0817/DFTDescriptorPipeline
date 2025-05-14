
import os
import pandas as pd
from descriptors.extractor import extract_homo_lumo, extract_dipole_moment, extract_polarizability
from descriptors.sterimol import extract_nbo_section, find_oh_bonds, find_c1_c2

def generate_feature_table(log_dir="logfiles"):
    rows = []
    for file in os.listdir(log_dir):
        if file.endswith(".log"):
            path = os.path.join(log_dir, file)
            name = file.replace(".log", "")
            homo, lumo = extract_homo_lumo(path)
            dipole = extract_dipole_moment(path)
            polar = extract_polarizability(path)
            nbo = extract_nbo_section(path)
            oh_atoms = find_oh_bonds(nbo) if nbo else []
            c1, c2, anchor = find_c1_c2(nbo, oh_atoms) if nbo else (None, None, None)
            rows.append({
                "name": name,
                "Ar_HOMO": homo,
                "Ar_LUMO": lumo,
                "Ar_dp": dipole,
                "Ar_polar": polar,
                "Ar_C1": c1,
                "Ar_C2": c2,
                "Ar_anchor": anchor
            })
    return pd.DataFrame(rows).dropna()
