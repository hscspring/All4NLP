import re

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

