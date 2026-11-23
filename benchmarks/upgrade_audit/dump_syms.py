"""Resolve every GPU4PySCF symbol Electrolites references, in the GPU4PySCF
that is first on the path, and record its signature and source.

    python benchmarks/upgrade_audit/extract_refs.py electrolites refs.json
    python benchmarks/upgrade_audit/dump_syms.py refs.json d181.json
    PYTHONPATH=/path/to/g4p-1.9.2 python benchmarks/upgrade_audit/dump_syms.py refs.json d192.json
    python benchmarks/upgrade_audit/cmp_syms.py d181.json d192.json st.json > cmp.txt

Does not import electrolites, so nothing is patched.  Needs a visible GPU
only because ``import gpu4pyscf`` queries the device count.
"""
import importlib
import inspect
import json
import sys
import types

import gpu4pyscf

refs = json.load(open(sys.argv[1]))
syms = sorted(set(s for v in refs.values() for s in v))


def resolve(path):
    parts = path.split('.')
    obj = None
    i = len(parts)
    while i > 0:
        try:
            obj = importlib.import_module('.'.join(parts[:i]))
            break
        except ImportError:
            i -= 1
        except Exception as e:                          # noqa: BLE001
            return None, f'import error {type(e).__name__}: {e}'
    if obj is None:
        return None, 'no module'
    for p in parts[i:]:
        try:
            obj = getattr(obj, p)
        except AttributeError:
            return None, (f'missing: {".".join(parts[:i])} has no '
                          f'{".".join(parts[i:])}')
        except Exception as e:                          # noqa: BLE001
            return None, f'getattr error {type(e).__name__}: {e}'
    return obj, None


res = {'version': gpu4pyscf.__version__}
for s in syms:
    obj, err = resolve(s)
    d = {}
    if err:
        d['err'] = err
    else:
        d['kind'] = type(obj).__name__
        if isinstance(obj, property):
            obj = obj.fget
        if (isinstance(obj, (types.FunctionType, types.MethodType))
                or inspect.isclass(obj)):
            try:
                d['sig'] = str(inspect.signature(obj))
            except Exception:                           # noqa: BLE001
                pass
            try:
                d['src'] = inspect.getsource(obj)
                d['file'] = inspect.getsourcefile(obj).split('gpu4pyscf/', 1)[-1]
            except Exception:                           # noqa: BLE001
                pass
        elif isinstance(obj, (int, float, str, bool, tuple)):
            d['val'] = repr(obj)[:200]
    res[s] = d
json.dump(res, open(sys.argv[2], 'w'))
print(res['version'], sum('err' in v for k, v in res.items() if k != 'version'),
      'unresolved of', len(syms))
