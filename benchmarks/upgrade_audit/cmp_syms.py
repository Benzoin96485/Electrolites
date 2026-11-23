"""Compare two ``dump_syms.py`` outputs and list, per Electrolites module,
every referenced GPU4PySCF symbol that is GONE, changed signature (SIG),
changed source (BODY, with the number of changed lines) or value (VAL).

    python benchmarks/upgrade_audit/cmp_syms.py old.json new.json status.json > report.txt

Needs ``refs.json`` (from ``extract_refs.py``) in the working directory.
A SIG line can be a false alarm when only a default value's repr differs
(e.g. a function object's address); BODY lists every textual change, most of
which are harmless -- the report is a reading list, not a verdict.  It cannot
see C ABI changes: a ctypes call into a kernel whose argument list moved
shows up nowhere here.
"""
import difflib
import json
import sys

refs = json.load(open('refs.json'))
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
print('compare', a['version'], '->', b['version'])
status = {}
for s in a:
    if s == 'version':
        continue
    x, y = a[s], b[s]
    if 'err' in x and 'err' in y:
        continue                    # not a real reference (unresolved in both)
    if 'err' in y:
        status[s] = ('GONE', y['err'])
        continue
    if 'err' in x:
        status[s] = ('NEW', '')
        continue
    if x.get('sig') != y.get('sig'):
        status[s] = ('SIG', f"{x.get('sig')}  ->  {y.get('sig')}")
        continue
    if x.get('src') != y.get('src'):
        d = list(difflib.unified_diff(x.get('src', '').splitlines(),
                                      y.get('src', '').splitlines(),
                                      lineterm='', n=0))
        nch = sum(1 for line in d if line[:1] in '+-'
                  and not line.startswith(('+++', '---')))
        status[s] = ('BODY', f'{nch} lines')
        continue
    if x.get('val') != y.get('val'):
        status[s] = ('VAL', f"{x.get('val')} -> {y.get('val')}")
json.dump(status, open(sys.argv[3], 'w'), indent=0)
for mod, syms in refs.items():
    hits = [(s, status[s]) for s in syms if s in status and status[s][0] != 'NEW']
    if not hits:
        continue
    print(f'== {mod}')
    for s, (k, m) in hits:
        print(f'   {k:4s} {s}   {m[:230]}')
