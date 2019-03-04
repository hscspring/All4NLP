import re


DIGIT = 'D'
ENGLISH = 'E'
# 空白
pwhi = re.compile(rf'[\t\u3000 ]+')
# 数字
pnum = re.compile(rf'-*[１２３４５６７８９０\d]+[.%:：]*[１２３４５６７８９０\d%]*')
# 英文
peng = re.compile(rf'[a-zA-Z]+')
# 图片
ppic = re.compile(r'!\[.*\]\(.+\)')
# 链接
plink = re.compile(r'\[\w+\]\(.+\)')
# 井号星号
pjxhao = re.compile(r'[>*#\xa0]+')
# 表情
pemj = re.compile(
    u'['u'\U0001F300-\U0001F64F'u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+',
    re.UNICODE)
# 纯汉字
pzh = re.compile(rf'[\u4e00-\u9fa5]+')
# 汉字、英文、数字、常规标点
pnormal = re.compile(rf'[\u4e00-\u9fa5a-zA-Z\d、，。：；！“”‘’？…《》（）\(\)『』「」\n]+')
# 中文大字符集
pzh_big = re.compile(rf'[\u2E80-\uFE4F]+')
# 替换掉类似（1）【1】这样的序号
p_pairpuc = re.compile(r'''
    (?P<p_pairpuc>
    【[\s\d]+】
    |〖[\s\d]+〗
    |〈[\s\d]+〉
    |＜[\s\d]+＞
    |<[\s\d]+>
    |〔[\s\d]+〕
    |﹝[\s\d]+﹞
    |［[\s\d]+］
    |\[[\s\d]+\]
    |（[\s\d]+）
    |\([\s\d]+\)
    |\{[\s\d]+\}
    )''', re.UNICODE | re.VERBOSE)  # 标准对
# 替换掉类似 1、1. 这样的序号
p_order = re.compile(rf'[ ]*[\d]+[、，. ]+')
# 替换掉多余的标点符号
p_punc = re.compile(rf'[、，。：；！？…]+')


def readtext_raw(file, m=0, encoding='utf-8'):
    data = []
    n = 0
    with open(file, 'r', encoding=encoding, errors='ignore') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0:
                continue
            data.append(line)
            n += 1
            if n > m and m != 0:
                break
    return data


def readtext_clean(file,
                   reporder=False,
                   repeng=False,
                   repnum=False,
                   normaltext=False):
    data = []
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = re.compile(rf'[\n]+').sub("\n", str(line))
            line = pwhi.sub(" ", line)
            line = ppic.sub("", line)
            line = plink.sub("", line)
            line = pjxhao.sub("", line)
            line = pemj.sub("", line)
            line = line.replace(",", "，")
            line = line.replace(":", "：")
            line = line.replace(";", "；")
            # 点替换为句号，但点后面有数字不替换
            line = re.compile(r'(?<!\d)[.]').sub("。", line)
            line = re.compile(rf'[?？]+').sub("？", line)
            line = re.compile(rf'[!！]+').sub("！", line)
            line = re.compile(rf'[。]+').sub("。", line)
            if reporder:
                line = p_pairpuc.sub(" ", line)
            if repeng:
                line = peng.sub(ENGLISH, line)
            if repnum:
                line = pnum.sub(DIGIT, line)
            if normaltext:
                line = " ".join(pnormal.findall(line))
            if len(line) == 0:
                continue
            data.append(line)
    return data
