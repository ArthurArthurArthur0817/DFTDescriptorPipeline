# sterimol.py - Locate atoms and support Sterimol calculations
import re

def find_oh_bonds(nbo_section):
    return [int(a) for a, _ in re.findall(r"BD \(\s*1\s*\)\s*O\s*(\d+)\s*-\s*H\s*(\d+)", nbo_section)]

def find_c1_c2(nbo_section, oh_atoms):
    for a in oh_atoms:
        c_candidates = re.findall(rf"BD \(\s*1\s*\)\s*C\s*(\d+)\s*-\s*O\s*{a}", nbo_section)
        for c in map(int, c_candidates):
            o_d_candidates = re.findall(rf"BD \(\s*[12]\s*\)\s*C\s*{c}\s*-\s*O\s*(\d+)", nbo_section)
            for d in map(int, o_d_candidates):
                e_candidates = re.findall(rf"BD \(\s*1\s*\)\s*C\s*(\d+)\s*-\s*C\s*{c}", nbo_section)
                for e in map(int, e_candidates):
                    bond_types = re.findall(r"BD \(\s*(1|2)\s*\)\s*(\w+)\s*(\d+)\s*-\s*(\w+)\s*(\d+)", nbo_section)
                    bond_pairs = {}
                    for bond_type, _, n1, _, n2 in bond_types:
                        nums = frozenset([int(n1), int(n2)])
                        bond_pairs.setdefault(nums, set()).add(bond_type)
                    single_count = sum('1' in t for t in bond_pairs.values())
                    double_count = sum('2' in t for t in bond_pairs.values())
                    if single_count >= 2 and double_count >= 1:
                        return c, e, a
    return None, None, None
