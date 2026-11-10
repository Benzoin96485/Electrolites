"""Analytic transposes of variable-size GPU matrix-chain plans.

For Y[o] += alpha*w*A0*...*Ak, each operand bar is another batched
matrix chain, with scatter-add for repeated input blocks. Scalar weights
receive the complete product contraction. Plans are reused across VJPs;
no dense Jacobian or per-pair Python differentiation is constructed.
"""
import numpy as np
import cupy as cp

from . import reverse as ad
from .ragged import Ragged, op
from .vbatch import VChainPlan, VDotPlan, _Op


def _flip(operand):
    return op(operand.layout, operand.idx, not operand.trans, operand.row)


def _numeric(layout, flat):
    result = Ragged(layout, [])
    offset = 0
    for members, (rows, cols) in zip(layout.members, layout.shapes):
        size = len(members)*rows*cols
        result.data.append(flat[offset:offset+size].reshape(len(members), rows, cols))
        offset += size
    if offset != flat.size:
        raise ValueError('Ragged storage size differs from its layout')
    return result


class AdjointRagged:
    """Ragged values whose arithmetic participates in the tensor VJP."""
    def __init__(self, layout, data=None):
        self.layout = layout
        self.data = ([ad.Variable(cp.zeros((len(m), r, c)), tracked=False)
                      for m, (r, c) in zip(layout.members, layout.shapes)]
                     if data is None else list(data))

    def copy(self):
        return AdjointRagged(self.layout, [d.copy() for d in self.data])

    def flat(self):
        return ad.concatenate([d.ravel() for d in self.data])

    def set_flat(self, value):
        offset = 0
        for group, members in enumerate(self.layout.members):
            r, c = self.layout.shapes[group]
            size = len(members)*r*c
            self.data[group] = value[offset:offset+size].reshape(len(members), r, c)
            offset += size
        return self

    def get(self, item):
        layout = self.layout
        return self.data[layout.group[item]][layout.slot[item], :layout.rows[item], :layout.cols[item]]

    def _binary(self, other, function):
        rhs = other.data if isinstance(other, (Ragged, AdjointRagged)) else [other]*len(self.data)
        return AdjointRagged(self.layout, [function(a, b) for a, b in zip(self.data, rhs)])

    def __add__(self, other):
        return self._binary(other, lambda a, b: a+b)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a-b)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a*b)

    __rmul__ = __mul__

    def T(self):
        return AdjointRagged(self.layout, [d.transpose(0, 2, 1).copy() for d in self.data])

    def vdot_items(self, other):
        result = ad.Variable(cp.zeros(self.layout.n), tracked=False)
        for group, members in enumerate(self.layout.members):
            if len(members):
                result[cp.asarray(members)] = np.sum(self.data[group]*other.data[group], axis=(1, 2))
        return result


def _operand_snapshots(rags, factories):
    """Keep dynamic input records immutable and large constant blocks shared.

    Regenerated overlaps are supplied by a factory and do not survive the
    forward action in an otherwise uncached chain checkpoint.
    """
    parents, retained = [], []
    for i, rag in enumerate(rags):
        if i in factories:
            parents.append(None)
            retained.append(None)
        elif isinstance(rag, AdjointRagged):
            flat = rag.flat()
            parents.append(flat)
            retained.append(_numeric(rag.layout, ad.primal(flat)))
        else:
            parents.append(None)
            retained.append(rag)

    def numeric():
        return [factories[i]() if i in factories else value for i, value in enumerate(retained)]
    return parents, numeric


def _chain_transpose(operands, output_bar, index):
    """Operand order and source positions for one exact product VJP."""
    before, after = list(range(index)), list(range(index+1, len(operands)))
    if operands[index].trans:
        sources = after+[-1]+before
        chain = [operands[i] for i in after]+[_flip(output_bar)]+[operands[i] for i in before]
    else:
        sources = before[::-1]+[-1]+after[::-1]
        chain = ([_flip(operands[i]) for i in before[::-1]]+[output_bar]
                 +[_flip(operands[i]) for i in after[::-1]])
    target = operands[index]
    plan = VChainPlan(target.layout, target.idx, chain, out_row=target.row)
    return plan, sources


class AdjointChainPlan(VChainPlan):
    """Primal GPU chain with reusable exact reverse-chain plans."""
    def __init__(self, out_layout, out_idx, operands, out_row=None, order=None):
        if len(operands) < 2:
            raise ValueError('The batched adjoint currently requires at least two matrix operands')
        self.operands = list(operands)
        self.output_indices = np.asarray(out_idx, dtype=np.int64)
        self.output_rows = None if out_row is None else np.asarray(out_row, dtype=np.int64)
        self.backward_plans = {}
        self.weight_plan = None
        super().__init__(out_layout, out_idx, operands, out_row=out_row, order=order)

    def _output_bar(self):
        column = self.output_rows is not None and np.all(_Op(self.operands[-1]).n == 1)
        return op(self.out_layout, self.output_indices, bool(column), row=self.output_rows)

    def run(self, out, rags, w=None, alpha=1., *, constant_factories=None):
        if not isinstance(out, AdjointRagged):
            return super().run(out, rags, w=w, alpha=alpha)
        if self.nterms == 0:
            return out
        factories = {} if constant_factories is None else constant_factories
        incoming = out.flat()
        parents, numeric = _operand_snapshots(rags, factories)
        operands = [incoming, *parents, w]
        value = ad.primal(incoming).copy()
        super().run(_numeric(out.layout, value), numeric(), w=ad.primal(w), alpha=alpha)
        output_bar = self._output_bar()

        def pullback(seed, needed):
            bars = [seed if needed[0] else None]+[None]*len(rags)+[None]
            inputs = numeric()
            gradient = _numeric(self.out_layout, seed)
            for index, need in enumerate(needed[1:-1]):
                if not need:
                    continue
                if index not in self.backward_plans:
                    self.backward_plans[index] = _chain_transpose(self.operands, output_bar, index)
                plan, sources = self.backward_plans[index]
                result = Ragged(self.operands[index].layout)
                plan.run(result, [gradient if i == -1 else inputs[i] for i in sources],
                         w=ad.primal(w), alpha=alpha)
                bars[index+1] = result.flat()
            if needed[-1]:
                if self.weight_plan is None:
                    if output_bar.trans:
                        # Write a column into an output row: transpose the
                        # entire product so the Frobenius dot uses that row.
                        left = _flip(output_bar)
                        right = [_flip(o) for o in self.operands[::-1]]
                    else:
                        left, right = output_bar, self.operands
                    self.weight_plan = VDotPlan([left], right)
                factors = inputs[::-1] if output_bar.trans else inputs
                bars[-1] = alpha*self.weight_plan.run([gradient], factors)
            return tuple(bars)

        result = ad.operation(value, operands, lambda seed: pullback(seed, [True]*len(operands)), pullback)
        return out.set_flat(result)


class AdjointDotPlan(VDotPlan):
    """A scalar product of one block and a matrix chain, including both sides."""
    def __init__(self, left, right):
        if len(left) != 1 or len(right) < 2 or left[0].trans:
            raise ValueError('The GPU dot adjoint needs one untransposed left block and a matrix chain')
        self.left, self.right = list(left), list(right)
        self.backward_plans = {}
        super().__init__(left, right)

    def run(self, lrags, rrags, *, constant_factories=None):
        factories = {} if constant_factories is None else constant_factories
        rags = lrags+rrags
        if not any(isinstance(r, AdjointRagged) for r in rags):
            return super().run(lrags, rrags)
        parents, numeric = _operand_snapshots(rags, factories)
        inputs = numeric()
        value = super().run(inputs[:1], inputs[1:])
        lo = self.left[0]

        def pullback(seed, needed):
            inputs = numeric()
            bars = [None]*len(parents)
            if needed[0]:
                if 'left' not in self.backward_plans:
                    self.backward_plans['left'] = VChainPlan(lo.layout, lo.idx, self.right, out_row=lo.row)
                result = Ragged(lo.layout)
                self.backward_plans['left'].run(result, inputs[1:], w=seed)
                bars[0] = result.flat()
            for index, need in enumerate(needed[1:]):
                if not need:
                    continue
                if index not in self.backward_plans:
                    self.backward_plans[index] = _chain_transpose(self.right, lo, index)
                plan, sources = self.backward_plans[index]
                result = Ragged(self.right[index].layout)
                plan.run(result, [inputs[0] if i == -1 else inputs[i+1] for i in sources], w=seed)
                bars[index+1] = result.flat()
            return tuple(bars)

        return ad.operation(value, parents, lambda seed: pullback(seed, [True]*len(parents)), pullback)
