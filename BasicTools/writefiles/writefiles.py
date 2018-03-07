import jieba


def write2txt(file, datalist, lenneed=True, MIN=8, MAX=32, listlen=True):
    with open(file, 'w') as f:
        for line in datalist:
            if lenneed:
                if listlen:
                    lenline = len(jieba.lcut(line))
                else:
                    lenline = len(line)
                if lenline >= MIN and lenline <= MAX:
                    f.write(line+"\n")
            else:
                f.write(line+"\n")