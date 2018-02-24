# coding:utf-8


def list2ngrams(lst, n, exact=True):
    """ Convert list into character ngrams. """
    return ["".join(lst[i:i+n]) for i in range(len(lst)-(n-1))]
