from descriptors.utils import atomic_symbols

def extract_last_standard_orientation(log_path):
    import re
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
            if "-----" in line:
                continue
            if any(x in line for x in ["Center", "Atomic", "Number"]):
                continue
            if line.strip() == "":
                if block:
                    geometries.append(block)
                    block = []
                reading = False
            else:
                if re.match(r"^\s*\d+\s+\d+\s+\d+\s+[-+]?\d*\.\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", line):
                    block.append(line)
                else:
                    if block:
                        geometries.append(block)
                        block = []
                    reading = False

    if not geometries:
        return None

    last_geom = geometries[-1]
    atoms = []
    for line in last_geom:
        parts = line.split()
        try:
            atomic_num = int(parts[1])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            symbol = atomic_symbols.get(atomic_num, None)
            if symbol is None:
                print(f"Unknown atomic number {atomic_num} in {log_path}")
                return None
            atoms.append((symbol, x, y, z))
        except Exception as e:
            print(f"Parse error in {log_path}: {e}")
            return None

    return atoms

def write_xyz(atom_list, filename):
    with open(filename, "w") as f:
        f.write(f"{len(atom_list)}\n")
        f.write("Extracted from Gaussian log\n")
        for atom in atom_list:
            f.write(f"{atom[0]}  {atom[1]:.8f}  {atom[2]:.8f}  {atom[3]:.8f}\n")