# coding:utf-8

"""
在目录下运行下面代码：
python get_w2r.py --filedir='/Users/HaoShaochun/Desktop/testwuxia' --select="风格建议"  
filedir 是要处理的文章目录
select  是文件名筛选标记（包含的字段）
"""
from collections import Counter, OrderedDict
import json
import os
import click


@click.command()
@click.option('--filedir', default="./", help="input file dir")
@click.option('--select', default="风格建议", help="select you need")
def get_w2r(filedir, select):

    #     for file in pathlib.Path(filedir).glob(select):
    for file in os.listdir(filedir):
        if select not in file:
            continue
        words = []
        w2r = {}
        with open(file, 'r') as f:
            for line in f.readlines():
                if len(line.split()) != 2:
                    continue
                line = line.strip()
                w = line.split()
                words.append(w[0])
                w2r[w[0]] = w[1]
        count = OrderedDict(sorted(Counter(words).items(),
                                   key=lambda t: t[1], reverse=True))
        outdir = os.path.join(filedir, 'w2r', file, '.txt')
        with open(filedir+'w2r_'+file+'.txt', 'w') as f:
            for w, freq in count.items():
                outline = w + ", " + w2r[w] + ", " + str(freq)

                f.write(outline + "\n")

if __name__ == '__main__':
    get_w2r()
