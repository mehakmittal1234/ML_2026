#!/usr/bin/env python3
"""Apply a pair diff (action = remove | add) to a matching_results.tsv (stdlib only).
usage: python3 apply_diff.py <base matching_results.tsv> <diff.tsv> <out matching_results.tsv>"""
import sys
base, diff, out = sys.argv[1], sys.argv[2], sys.argv[3]
rm, add = {}, {}
with open(diff, encoding='utf-8') as f:
    next(f)
    for line in f:
        act, a, b = line.rstrip('\n').split('\t')[:3]
        (rm if act == 'remove' else add).setdefault(a, set()).add(b)
n_in = n_out = 0
with open(base, encoding='utf-8') as f, open(out, 'w', encoding='utf-8') as g:
    g.write(next(f))
    for line in f:
        s1, _, ids = line.rstrip('\n').partition('\t')
        lst = [x for x in ids.split(',') if x]
        n_in += len(lst)
        keep = [x for x in lst if x not in rm.get(s1, ())]
        keep += sorted(add.get(s1, set()) - set(keep))
        n_out += len(keep)
        g.write(f"{s1}\t{','.join(keep)}\n")
print(f'pairs {n_in} -> {n_out}')
