


def read_normal_txt(file, m=0):
    data = []
    n = 0
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            data.append(line)
            n += 1
            if n > m and m != 0:
                break
    return data