import os
import pandas as pd
from extractor import (
    extract_polarizability, extract_homo_lumo, extract_dipole_moment,
    extract_nbo_section, extract_nbo_values, extract_coordinates,
    extract_nbo_charges, extract_frequencies, find_oh_bonds, find_c1_c2
)
from sterimol import extract_last_standard_orientation, write_xyz
from morfeus import read_xyz, Sterimol
from morfeus.utils import get_radii

from extractor import extract_nbo_charges, extract_coordinates, extract_nbo_values, extract_frequencies, extract_polarizability, extract_homo_lumo, extract_dipole_moment, extract_nbo_section, find_oh_bonds, find_c1_c2

def write_xyz(filepath, elements, coordinates, exclude_indices):
    with open(filepath, 'w') as f:
        f.write(f"{len(elements) - len(exclude_indices)}\n\n")
        for i, (el, coord) in enumerate(zip(elements, coordinates)):
            if i not in exclude_indices:
                f.write(f"{el} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

def generate_feature_table(log_folder, excel_path):
    df = pd.read_excel(excel_path)

    for index, row in df.iterrows():
        ar = row["Ar"]
        log_file = os.path.join(log_folder, f"{ar}.log")
        print(f"➡️ 處理: {ar}.log")

        Ar_NBO_C1 = Ar_NBO_C2 = Ar_NBO_O1 = Ar_NBO_O2 = None
        Ar_v_C_O = Ar_I_C_O = None
        dipole_moment = avg_polar = homo = lumo = None
        occupancy_C1_O = energy_C1_O = occupancy_C1_C2 = energy_C1_C2 = None
        coord_C1 = coord_C2 = L_C1_C2 = None
        Ar_c = Ar_e = Ar_a = Ar_b = Ar_d = Ar_f = Ar_g = None
        Ar_Ster_L = Ar_Ster_B1 = Ar_Ster_B5 = None

        if not os.path.exists(log_file):
            print(f"⛔ 跳過，找不到 {log_file}")
            continue

        try:
            avg_polar = extract_polarizability(log_file)
            homo, lumo = extract_homo_lumo(log_file)
            dipole_moment = extract_dipole_moment(log_file)
            nbo_content = extract_nbo_section(log_file)

            if nbo_content:
                oh_atoms = find_oh_bonds(nbo_content)
                c1, c2, a, b, d, f, g = find_c1_c2(nbo_content, oh_atoms)
                print(f"✅ Found C1: {c1}, C2: {c2}, A: {a},B: {b}, D: {d}, F: {f}, G: {g}")
                Ar_c, Ar_e, Ar_a, Ar_b, Ar_d, Ar_f, Ar_g = c1, c2, a, b, d, f, g

                occupancy_C1_O, energy_C1_O, occupancy_C1_C2, energy_C1_C2 = extract_nbo_values(log_file, c1, c2, a)
                coord_C1, coord_C2, L_C1_C2 = extract_coordinates(log_file, c1, c2)
                Ar_NBO_C1, Ar_NBO_C2, Ar_NBO_O1, Ar_NBO_O2 = extract_nbo_charges(log_file, c1, c2, a)
                Ar_I_C_O, Ar_v_C_O = extract_frequencies(log_file)

                elements, coords = read_log(log_file)
                exclude_atoms = [Ar_a, Ar_b, Ar_d]
                write_xyz(log_file.replace(".log", ".xyz"), elements, coords, exclude_atoms)

                radii = Radius().get(elements)
                sterimol = Sterimol(elements, coords, Ar_c, Ar_e, radii=radii)
                Ar_Ster_L = sterimol.L_value
                Ar_Ster_B1 = sterimol.B_1_value
                Ar_Ster_B5 = sterimol.B_5_value

        except Exception as e:
            print(f"❌ 錯誤於 {log_file}: {e}")

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

        df.at[index, "Ar_Ster_L"] = Ar_Ster_L
        df.at[index, "Ar_Ster_B1"] = Ar_Ster_B1
        df.at[index, "Ar_Ster_B5"] = Ar_Ster_B5

    return df