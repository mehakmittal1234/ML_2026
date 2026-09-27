#!/usr/bin/env python3
"""Remove the pairs of a drop list from any matching_results.tsv (stdlib only).
usage: python3 apply_drop_list.py <matching_results.tsv> <drop_pairs.tsv> <out matching_results.tsv>"""
import sys
src, drop, out = sys.argv[1], sys.argv[2], sys.argv[3]
bad = {}
with open(drop, encoding='utf-8') as f:
    next(f)
    for line in f:
        a, b = line.rstrip('\n').split('\t')[:2]
        bad.setdefault(a, set()).add(b)
n_in = n_out = 0
with open(src, encoding='utf-8') as f, open(out, 'w', encoding='utf-8') as g:
    g.write(next(f))
    for line in f:
        s1, _, ids = line.rstrip('\n').partition('\t')
        lst = [x for x in ids.split(',') if x]
        keep = [x for x in lst if x not in bad.get(s1, ())]
        n_in += len(lst); n_out += len(keep)
        g.write(f"{s1}\t{','.join(keep)}\n")
print(f'pairs {n_in} -> {n_out} ({n_in - n_out} removed)')
