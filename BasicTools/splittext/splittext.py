import re
import jieba
import jieba.posseg as pseg


def split2sens(text):
    pstop = re.compile(rf'[。！？?!…]”*')
    sens = []
    stoplist = pstop.findall(text)
    senlist = []
    for sen in pstop.split(text):
        if len(sen) == 0:
            continue
        senlist.append(sen)
    for i, sen in enumerate(senlist):
        try:
            sen = sen + stoplist[i]
            sens.append(sen)
        except IndexError:
            continue
    return sens


def cut2words(text):
    return jieba.lcut(text)

def cut2wpos(text, pos=None):
    data = []
    for w,p in pseg.cut(text):
        if p == pos:
            continue
        data.append((w,p))
    return data
