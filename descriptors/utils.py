def parse_floats(line):
    # 僅匹配合法的浮點數（避免單獨的 "." 或 "--"）
    return [float(x) for x in re.findall(r'-?\d+\.\d+', line)]