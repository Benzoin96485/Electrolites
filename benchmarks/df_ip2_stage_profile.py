"""The ip2 call's stages, kernel by kernel, from an Nsight Systems report.

`docs/DF_WORK_LIST.md` **W37**.  ``FASTDFHESSIP_TIME=1 FASTDFHESSIP_NVTX=1``
makes ``fastdfhessip._Stages`` put an NVTX mark at the end of every stage,
after its synchronise, so every kernel that starts between two marks belongs
to the second one.  This script reads the report's SQLite export and prints,
for each stage of one call, its wall time, the time the GPU was busy in it,
and the kernels that took that time -- by name, count and total -- with,
where a stage's flop count is known (``--sizes``), the rate it ran at.

    FASTDFHESSIP_TIME=1 FASTDFHESSIP_NVTX=1 nsys profile -t cuda,nvtx \\
        -c cudaProfilerApi -o $SCRATCH/p python benchmarks/df_hess_terms_probe.py \\
        --xyz ... --patch all --terms ip2 --variants ours --reps 1 --nvtx
    nsys export --type sqlite -o $SCRATCH/p.sqlite $SCRATCH/p.nsys-rep
    python benchmarks/df_ip2_stage_profile.py $SCRATCH/p.sqlite \\
        [--range ip2:ours:0] [--top 6]

No GPU is needed to read a report.
"""
import argparse
import re
import sqlite3
from collections import defaultdict

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument('db')
ap.add_argument('--range', default='',
                help='only marks inside the NVTX range of this name '
                     "(df_hess_terms_probe.py --nvtx's 'term:label:rep'); "
                     'default: the last call in the report')
ap.add_argument('--top', type=int, default=6)
ap.add_argument('--width', type=int, default=70)
a = ap.parse_args()

con = sqlite3.connect(a.db)
cur = con.cursor()
tables = {r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table'")}


def strings():
    return dict(cur.execute('SELECT id, value FROM StringIds'))


S = strings()


def text(row_text, row_id):
    return row_text if row_text is not None else S.get(row_id, '')


ev = list(cur.execute(
    'SELECT start, end, text, textId, eventType FROM NVTX_EVENTS '
    'ORDER BY start'))
marks = [(s, text(t, tid)) for s, e, t, tid, et in ev
         if (e is None or e == s) and '|' in (text(t, tid) or '')]
ranges = [(s, e, text(t, tid)) for s, e, t, tid, et in ev
          if e is not None and e > s]
if a.range:
    rs = [r for r in ranges if r[2] == a.range]
    assert rs, f'no NVTX range {a.range!r}; have ' + \
        ', '.join(sorted({r[2] for r in ranges}))
    lo, hi = rs[0][0], rs[0][1]
    marks = [m for m in marks if lo <= m[0] <= hi]
else:
    # the last call: back from the last mark to the first stage's mark
    # after the previous call's final one
    tags = [m[1].split('|')[0] for m in marks]
    last = len(marks) - 1
    first = last
    while first > 0 and marks[first - 1][1].split('|')[1] != 'E aggregate' \
            and tags[first - 1] == tags[last]:
        first -= 1
    marks = marks[first:last + 1]
    lo = marks[0][0] - 1

kern = []
if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
    kern += [(s, e, S.get(n, str(n))) for s, e, n in cur.execute(
        'SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL '
        'WHERE end >= ? AND start <= ?', (lo, marks[-1][0]))]
for tab, name in (('CUPTI_ACTIVITY_KIND_MEMCPY', '[memcpy]'),
                  ('CUPTI_ACTIVITY_KIND_MEMSET', '[memset]')):
    if tab in tables:
        kern += [(s, e, name) for s, e in cur.execute(
            f'SELECT start, end FROM {tab} WHERE end >= ? AND start <= ?',
            (lo, marks[-1][0]))]
kern.sort()


def short(name):
    name = re.sub(r'\s+', ' ', name)
    return name if len(name) <= a.width else name[:a.width - 3] + '...'


prev = lo
tot_wall = tot_busy = 0.
rows = []
for t, label in marks:
    ks = [k for k in kern if prev <= k[0] < t]
    busy = sum(min(e, t) - s for s, e, _ in ks)
    by = defaultdict(lambda: [0, 0.])
    for s, e, n in ks:
        by[n][0] += 1
        by[n][1] += e - s
    rows.append((label, (t - prev) * 1e-6, busy * 1e-6, by))
    tot_wall += t - prev
    tot_busy += busy
    prev = t

agg = defaultdict(lambda: [0., 0., {}])
order = []
for label, wall, busy, by in rows:
    stage = label.split('|', 1)[1]
    if stage not in agg:
        order.append(stage)
    g = agg[stage]
    g[0] += wall
    g[1] += busy
    for n, (c, d) in by.items():
        x = g[2].setdefault(n, [0, 0.])
        x[0] += c
        x[1] += d

print(f'# {a.db}: {len(marks)} marks, {tot_wall * 1e-6:.1f} ms wall, '
      f'{tot_busy * 1e-6:.1f} ms GPU busy')
for stage in order:
    wall, busy, by = agg[stage]
    print(f'\n{stage:24s} wall {wall:9.1f} ms   busy {busy:9.1f} ms '
          f'({100 * busy / max(wall, 1e-9):4.1f} %)')
    for n, (c, d) in sorted(by.items(), key=lambda x: -x[1][1])[:a.top]:
        print(f'    {d * 1e-6:9.1f} ms  {c:6d}x  {short(n)}')
