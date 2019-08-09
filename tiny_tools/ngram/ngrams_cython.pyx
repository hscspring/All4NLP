from collections import defaultdict


def ngrams(tokens, int MIN_N, int MAX_N):
    cdef Py_ssize_t i, j, n_tokens

    count = defaultdict(int)

    join_spaces = "".join

    n_tokens = len(tokens)
    for i in range(n_tokens):
        for j in range(i+MIN_N, min(n_tokens, i+MAX_N)+1):
            count[join_spaces(tokens[i:j])] += 1

    return count
