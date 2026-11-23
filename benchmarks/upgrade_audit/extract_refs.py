"""Collect every gpu4pyscf symbol each electrolites module touches (static AST)."""
import ast, json, os, sys
root = sys.argv[1]
out = {}
for fn in sorted(os.listdir(root)):
    if not fn.endswith('.py'):
        continue
    src = open(os.path.join(root, fn)).read()
    tree = ast.parse(src)
    alias = {}   # local name -> dotted gpu4pyscf path
    syms = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('gpu4pyscf'):
            for a in node.names:
                full = f'{node.module}.{a.name}'
                alias[a.asname or a.name] = full
                syms.add(full)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith('gpu4pyscf'):
                    if a.asname:
                        alias[a.asname] = a.name
                    else:
                        alias['gpu4pyscf'] = 'gpu4pyscf'
                    syms.add(a.name)
    def chain(n):
        parts = []
        while isinstance(n, ast.Attribute):
            parts.append(n.attr); n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id); return parts[::-1]
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            c = chain(node)
            if c and c[0] in alias:
                base = alias[c[0]]
                # keep up to 3 trailing attributes
                for k in range(1, min(len(c), 4)):
                    syms.add('.'.join([base] + c[1:k+1]))
        # string targets like ('gpu4pyscf.scf.jk', '_VHFOpt.get_k')
        if isinstance(node, ast.Tuple) and len(node.elts) == 2 and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
            m, a = node.elts[0].value, node.elts[1].value
            if m.startswith('gpu4pyscf') and not a.startswith('gpu4pyscf'):
                syms.add(f'{m}.{a}')
    out[fn[:-3]] = sorted(syms)
json.dump(out, open(sys.argv[2], 'w'), indent=0)
print(sum(len(v) for v in out.values()), 'refs;', len(set(s for v in out.values() for s in v)), 'unique')
