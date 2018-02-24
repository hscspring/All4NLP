# coding:utf-8

"""
转码工具
python transcoding.py --out_encoding="gbk"
"""

import codecs
import chardet
import os

import click
import docx



@click.command()
@click.option('--filepath', default="./", help="input file path")
@click.option('--select', default="md|txt|markdown|docx", help="select you need")
@click.option('--outpath', default="./", help="output file path")
@click.option('--out_encoding', default="utf-8", help="encoding for you want, GBK is not wider than gb18030")
def transcoding(filepath, select, outpath, out_encoding):

    for file in os.listdir(filepath):

        suffix = file.split('.')[-1]
        if suffix not in select.split('|'):
            continue

        tmp_text = []
        if suffix == 'docx':
            doc = docx.Document(filepath+"/"+file)
            for line in doc.paragraphs:
                tmp_text.append(line.text)
        else:
            with codecs.open(filepath+"/"+file, 'rb') as f:
                data = f.read()
                input_encoding = chardet.detect(data)['encoding']
                print("File %s is encoding by %s" % (file, input_encoding))
                data = data.decode(input_encoding, errors="replace")
                for line in data.split("\n"):
                    line = line.strip()
                    tmp_text.append(line)

        with codecs.open(outpath+"/out"+file+'.txt', 'w', encoding=out_encoding, errors="replace") as f:
            for line in tmp_text:
                f.write(line + "\n")


if __name__ == '__main__':
    transcoding()
