"""Exact deferred RI/orbital gathers at resident-region boundaries.

Only immutable metadata and canonical input indices cross a worker
boundary. No values, cutoffs, occupations or derivatives are approximated.
The context applies only while preparing a resident task; normal readers
retain their original behavior.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import numpy as np
from . import reverse as ad

_capture=ContextVar('dlpno_resident_read_capture',default=None)


@contextmanager
def capture_reads(enabled=True):
    statistics=dict(reads=0,canonical_references=0,avoided_parent_gather_bytes=0)
    token=_capture.set(statistics if enabled else None)
    try:yield statistics
    finally:_capture.reset(token)


def active():return _capture.get() is not None


def _token(key):
    if key is None:return None
    if not isinstance(key,tuple):key=(key,)
    result=[]
    for x in key:
        if isinstance(x,slice):result.append(('slice',x.start,x.stop,x.step))
        elif x is Ellipsis:result.append(('ellipsis',))
        else:
            a=np.asarray(x);result.append((a.dtype.str,a.shape,a.tobytes()))
    return tuple(result)


def _index(value):return value.index if hasattr(value,'index') else int(value)


@dataclass(frozen=True)
class EncodedRead:
    sources:tuple
    source_shapes:tuple
    parts:tuple
    cartesian_shape:tuple
    output_shape:tuple
    signature:tuple

    def evaluate(self,sources):
        xp=ad._xp(sources)
        value=xp.empty(self.cartesian_shape,dtype=np.float64)
        for source,(local,output) in zip(sources,self.parts):
            raw=ad.primal(source)
            value[output]=raw if local is None else raw[local]
        def reverse(bar,needed):
            bar=bar.reshape(self.cartesian_shape)
            return tuple((bar[output] if local is None else ad.IndexedCotangent(shape,local,bar[output])) if need else None
                for shape,(local,output),need in zip(self.source_shapes,self.parts,needed))
        return ad.operation(value.reshape(self.output_shape),sources,
            lambda bar:reverse(bar,[True]*len(sources)),reverse)


class DeferredRead:
    def __init__(self,sources,parts,cartesian_shape,output_shape):
        self.sources=tuple(sources);self.parts=tuple(parts)
        self.cartesian_shape=tuple(cartesian_shape);self.shape=tuple(output_shape)
        self.ndim=len(self.shape);self.dtype=np.dtype(np.float64);self.size=int(np.prod(self.shape));self.nbytes=self.size*8
        if not self.sources or len(self.sources)!=len(self.parts):raise ValueError('Deferred gather source layout differs')
        if any(np.dtype(ad.primal(x).dtype)!=self.dtype for x in self.sources):raise TypeError('Deferred gradient reads require float64')
        current=_capture.get()
        if current is not None:
            current['reads']+=1;current['canonical_references']+=len(self.sources)
            current['avoided_parent_gather_bytes']+=self.nbytes

    def encode(self,register):
        refs=tuple(register(x) for x in self.sources)
        shapes=tuple(tuple(x.shape) for x in self.sources)
        signature=(self.cartesian_shape,self.shape,tuple((_index(ref),shape,_token(local),_token(output))
            for ref,shape,(local,output) in zip(refs,shapes,self.parts)))
        return EncodedRead(refs,shapes,self.parts,self.cartesian_shape,self.shape,signature)


class OperandBank:
    def __init__(self,values):
        self.values=values;self.cache={}
        self.statistics=dict(materialized=0,cache_hits=0,output_bytes=0)

    def __getitem__(self,index):
        if isinstance(index,EncodedRead):return self.read(index)
        return self.values[_index(index)]

    def read(self,encoded):
        key=encoded.signature
        if key in self.cache:
            self.statistics['cache_hits']+=1
            return self.cache[key]
        result=encoded.evaluate(tuple(self[x] for x in encoded.sources))
        self.cache[key]=result;self.statistics['materialized']+=1
        self.statistics['output_bytes']+=result.size*8
        return result
