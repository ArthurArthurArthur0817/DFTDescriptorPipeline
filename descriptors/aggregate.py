
import os
import pandas as pd
from descriptors.extractor import (
    extract_homo_lumo, extract_dipole_moment, extract_polarizability,
    extract_nbo_section, extract_nbo_values, extract_coordinates,
    extract_frequencies, find_oh_bonds, extract_last_standard_orientation
)
from descriptors.sterimol import write_xyz
from morfeus import read_xyz, Sterimol
from morfeus.utils import get_radii

def generate_feature_table(log_folder, excel_path):
    df = pd.read_excel(excel_path)

    if "Ar" not in df.columns:
        if "Ar1" in df.columns:
            df["Ar"] = df["Ar1"].astype(str)
        else:
            raise ValueError("Excel file must contain 'Ar' or 'Ar1' column.")

    for index, row in df.iterrows():
        mol_name = str(row["Ar"])
        log_file = os.path.join(log_folder, f"{mol_name}.log")
        if not os.path.exists(log_file):
            print(f"❌ Missing log file: {log_file}")
            continue

        print(f"Processing {mol_name}...")

        # Default all to None
        Ar_HOMO = Ar_LUMO = Ar_dp = Ar_polar = None
        Ar_NBO_C1 = Ar_NBO_C2 = Ar_NBO_O1 = Ar_NBO_O2 = None
        Ar_I_C_O = Ar_v_C_O = L_C1_C2 = None
        Ar_Ster_L = Ar_Ster_B1 = Ar_Ster_B5 = None

        # 基本特徵提取
        try:
            Ar_polar = extract_polarizability(log_file)
        except:
            pass
        try:
            Ar_HOMO, Ar_LUMO = extract_homo_lumo(log_file)
        except:
            pass
        try:
            Ar_dp = extract_dipole_moment(log_file)
        except:
            pass

        # NBO & Frequency & Geometry
        result = find_oh_bonds(log_file)
        if result:
            c1, c2, a, b, d, f, g = result
            print(f"Found C1: {c1}, C2: {c2}, A: {a},B: {b}, D: {d}, F: {f}, G: {g}")
            try:
                coord1, coord2, L_C1_C2 = extract_coordinates(log_file, c1, c2)
            except:
                print(f"Error: Could not find coordinates for both C1 and C2.")
                coord1 = coord2 = L_C1_C2 = None

            try:
                if a is not None:
                    Ar_NBO_C1, Ar_NBO_C2, Ar_NBO_O1, Ar_NBO_O2 = extract_nbo_charges(log_file, c1, c2, a)
                else:
                    print(f"⚠️ Skipping NBO O-related charges for {log_file} (a=None)")
            except Exception as e:
                print(f"❌ Error extracting NBO charges from {log_file}: {e}")

            try:
                freq_data = extract_frequencies(log_file)
                if freq_data:
                    Ar_I_C_O, Ar_v_C_O = freq_data
                else:
                    print(f"⚠️ No frequencies found in {log_file}")
            except:
                pass

            # Sterimol 部分
            Ar_c, Ar_e, Ar_a, Ar_b, Ar_d, Ar_f, Ar_g = c1, c2, a, b, d, f, g
            atoms = extract_last_standard_orientation(log_file)
            if atoms and Ar_a is not None and Ar_c is not None and Ar_e is not None:
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

        # 寫入 DataFrame
        df.at[index, "Ar_HOMO"] = Ar_HOMO
        df.at[index, "Ar_LUMO"] = Ar_LUMO
        df.at[index, "Ar_dp"] = Ar_dp
        df.at[index, "Ar_polar"] = Ar_polar
        df.at[index, "Ar_NBO_C1"] = Ar_NBO_C1
        df.at[index, "Ar_NBO_C2"] = Ar_NBO_C2
        df.at[index, "Ar_NBO_=O"] = Ar_NBO_O1
        df.at[index, "Ar_NBO_-O"] = Ar_NBO_O2
        df.at[index, "Ar_I_C=O"] = Ar_I_C_O
        df.at[index, "Ar_v_C=O"] = Ar_v_C_O
        df.at[index, "L_C1_C2"] = L_C1_C2
        df.at[index, "Ar_Ster_L"] = Ar_Ster_L
        df.at[index, "Ar_Ster_B1"] = Ar_Ster_B1
        df.at[index, "Ar_Ster_B5"] = Ar_Ster_B5

    return df

def extract_nbo_charges(log_file, c1, c2, a):
    if a is None:
        return None, None, None, None
    section = extract_nbo_section(log_file)
    charges = extract_nbo_values(section)
    Ar_NBO_C1 = charges.get(f"C{c1}", None)
    Ar_NBO_C2 = charges.get(f"C{c2}", None)
    Ar_NBO_O1 = charges.get(f"O{a-1}", None)
    Ar_NBO_O2 = charges.get(f"O{a}", None)
    return Ar_NBO_C1, Ar_NBO_C2, Ar_NBO_O1, Ar_NBO_O2
