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


def generate_feature_table(log_folder, excel_path):
    df = pd.read_excel(excel_path)
    df_result = df.copy()

    for index, row in df.iterrows():
        mol_name = str(row["Ar"])
        log_file = os.path.join(log_folder, f"{mol_name}.log")

        if not os.path.exists(log_file):
            print(f"Skipping {mol_name}, log file not found.")
            continue

        # Descriptor Extraction
        avg_polar = extract_polarizability(log_file)
        homo, lumo = extract_homo_lumo(log_file)
        dipole = extract_dipole_moment(log_file)
        nbo_section = extract_nbo_section(log_file)

        Ar_c, Ar_e, Ar_a = None, None, None
        Ar_b, Ar_d, Ar_f, Ar_g = None, None, None, None
        Ar_NBO_C2 = Ar_NBO_O1 = Ar_NBO_O2 = Ar_v_C_O = Ar_I_C_O = L_C1_C2 = None

        if nbo_section:
            oh_atoms = find_oh_bonds(nbo_section)
            result = find_c1_c2(nbo_section, oh_atoms)
            if result:
                c1, c2, a, b, d, f, g = result
                Ar_c, Ar_e, Ar_a, Ar_b, Ar_d, Ar_f, Ar_g = c1, c2, a, b, d, f, g

                occ_C1_O, ene_C1_O, occ_C1_C2, ene_C1_C2 = extract_nbo_values(log_file, c1, c2, a)
                coord1, coord2, L_C1_C2 = extract_coordinates(log_file, c1, c2)
                Ar_NBO_C1, Ar_NBO_C2, Ar_NBO_O1, Ar_NBO_O2 = extract_nbo_charges(log_file, c1, c2, a)
                freq_data = extract_frequencies(log_file)
                if freq_data is not None:
                    Ar_I_C_O, Ar_v_C_O = freq_data
                else:
                    Ar_I_C_O = None
                    Ar_v_C_O = None

        # Sterimol
        Ar_Ster_L = Ar_Ster_B1 = Ar_Ster_B5 = None
        atoms = extract_last_standard_orientation(log_file)
        if atoms and Ar_a and Ar_c and Ar_e:
            try:
                exclude_atoms = [Ar_a, Ar_b, Ar_d]
                atoms_filtered = [a for i, a in enumerate(atoms) if (i + 1) not in exclude_atoms]
                if len(atoms_filtered) >= 2:
                    xyz_path = f"/tmp/{mol_name}_filtered.xyz"
                    write_xyz(atoms_filtered, xyz_path)
                    elements, coords = read_xyz(xyz_path)
                    radii = get_radii(elements, radii_type="bondi")
                    radii = [1.09 if r == 1.20 else r for r in radii]
                    sterimol = Sterimol(elements, coords, Ar_c, Ar_e, radii=radii)
                    Ar_Ster_L = sterimol.L_value
                    Ar_Ster_B1 = sterimol.B_1_value
                    Ar_Ster_B5 = sterimol.B_5_value
            except Exception as e:
                print(f"Sterimol error for {mol_name}: {e}")

        # 填入 DataFrame
        df_result.at[index, "Ar_dp"] = dipole
        df_result.at[index, "Ar_polar"] = avg_polar
        df_result.at[index, "Ar_LUMO"] = lumo
        df_result.at[index, "Ar_HOMO"] = homo
        df_result.at[index, "Ar_NBO_C2"] = Ar_NBO_C2
        df_result.at[index, "Ar_NBO_=O"] = Ar_NBO_O1
        df_result.at[index, "Ar_NBO_-O"] = Ar_NBO_O2
        df_result.at[index, "Ar_v_C=O"] = Ar_v_C_O
        df_result.at[index, "Ar_I_C=O"] = Ar_I_C_O
        df_result.at[index, "L_C1_C2"] = L_C1_C2
        df_result.at[index, "Ar_Ster_L"] = Ar_Ster_L
        df_result.at[index, "Ar_Ster_B1"] = Ar_Ster_B1
        df_result.at[index, "Ar_Ster_B5"] = Ar_Ster_B5

    return df_result
