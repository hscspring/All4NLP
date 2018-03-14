import re


DIGIT = 'D'
ENGLISH = 'E'
pwhi = re.compile(rf'[\t\u3000 ]+')
pnum = re.compile(rf'[１２３４５６７８９０\d]+[.%:：]*[１２３４５６７８９０\d]*')
peng = re.compile(rf'[a-zA-Z]+')
pzh = re.compile(rf'[\u2E80-\uFE4Fa-zA-Z\d、，。：；！？…《》（）\(\)『』「」]+')
# 替换掉类似（1）【1】这样的序号
p_pairpuc = re.compile(r'''
    (?P<p_pairpuc>
    \W【[\s\d]+】\W
    |\W〖[\s\d]+〗\W
    |\W〈[\s\d]+〉\W
    |\W＜[\s\d]+＞\W
    |\W<[\s\d]+>\W
    |\W〔[\s\d]+〕\W
    |\W﹝[\s\d]+﹞\W
    |\W［[\s\d]+］\W
    |\W\[[\s\d]+\]\W
    |\W（[\s\d]+）\W
    |\W\([\s\d]+\)\W
    |\W\{[\s\d]+\}\W
    )''', re.UNICODE | re.VERBOSE)  # 标准对
# 替换掉类似 1、1. 这样的序号
p_order = re.compile(rf'[ ]*[\d]+[、，. ]+')
# 替换掉多余的标点符号
p_punc = re.compile(rf'[、，。：；！？…]+')

def readtext_raw(file, m=0, encoding='utf-8'):
    data = []
    n = 0
    with open(file, 'r', encoding=encoding) as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            data.append(line)
            n += 1
            if n > m and m != 0:
                break
    return data


def readtext_clean(file, reporder=True, repeng=False, repnum=True, onlyzh=True):
    data = []
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.lower()
            if reporder:
                line = p_pairpuc.sub(" ", line)
                line = p_order.sub(" ", line)
            if repeng:
                line = peng.sub(ENGLISH, line)
            if repnum:
                line = pnum.sub(DIGIT, line)
            line = pwhi.sub(" ", line)
            line = line.replace(",", "，")
            line = line.replace(".", "。")
            line = line.replace("?", "？")
            line = line.replace("!", "！")
            line = line.replace(":", "：")
            line = line.replace(";", "；")
            line = line.replace("\"", "“")
            line = line.replace("”", "“")
            line = line.replace("……", "…")
            if onlyzh:
                line = "".join(pzh.findall(line))
            if len(line) == 0:
                continue
            data.append(line)
    return data
