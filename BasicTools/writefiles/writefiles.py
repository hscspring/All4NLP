import jieba


def write2txt(file, datalist, MIN, MAX, listlen=True):
    with open(file, 'w') as f:
        for line in datalist:
            if listlen:
                lenline = len(jieba.lcut(line))
            else:
                lenline = len(line)
            if lenline >= MIN and lenline <= MAX:
                f.write(line+"\n")