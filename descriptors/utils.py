import re
import os
import re

# ✅ 加在這裡
atomic_symbols = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P',
    16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

def parse_floats(line):
    # 僅匹配合法的浮點數（避免單獨的 "." 或 "--"）
    return [float(x) for x in re.findall(r'-?\d+\.\d+', line)]