"""Join background worker shutdown inside the complete gradient call.

This scope owns only pools registered by this gradient. Success, response
errors, and errors during energy-graph construction all join those pools.
Shutdown overlaps subsequent RI/orbital work; no time leaves the kernel.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import time

_current=ContextVar('dlpno_resident_gradient_scope',default=None)


def register(pool):
    scope=_current.get()
    if scope is not None and not any(x is pool for x in scope):scope.append(pool)


def owns(pool):
    scope=_current.get()
    return scope is not None and any(x is pool for x in scope)


@contextmanager
def gradient_scope(gradient):
    pools=[];token=_current.set(pools);failure=None
    try:
        yield
    except BaseException as error:
        failure=error
        raise
    finally:
        _current.reset(token)
        begin=time.perf_counter();errors=[]
        for pool in pools:
            try:pool.close()
            except BaseException as error:errors.append(error)
        if pools:gradient._record('resident_shutdown_wait',begin)
        if errors:
            if failure is not None:
                for error in errors:failure.add_note('Resident cleanup: '+repr(error))
            else:raise errors[0]
