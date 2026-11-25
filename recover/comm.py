"""Communicator layer: mpi4py, a serial stand-in, and a thread-based stand-in for tests.

Only these collectives are used by the driver:
  bcast, barrier, allreduce_max, alltoallv (complex128), scatterv / gatherv (float64),
  gather (Python objects), split (level groups).
"""
import os
import queue
import threading

import numpy as np


def _displs(counts):
    return np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)


# ----------------------------------------------------------------------------- mpi4py
class MPIComm:
    def __init__(self, c):
        from mpi4py import MPI
        self.MPI, self.c = MPI, c
        self.rank, self.size = c.Get_rank(), c.Get_size()

    def bcast(self, obj, root=0):
        return self.c.bcast(obj, root=root)

    def barrier(self):
        self.c.Barrier()

    def allreduce_max(self, x):
        return self.c.allreduce(x, op=self.MPI.MAX)

    def gather(self, obj, root=0):
        return self.c.gather(obj, root=root)

    def split(self, color):
        return MPIComm(self.c.Split(int(color), self.rank))

    @staticmethod
    def _check(counts):
        if int(np.sum(counts)) > 2 ** 31 - 1:
            raise RuntimeError("message larger than 2^31 elements; use more ranks per group")

    def alltoallv(self, sendbuf, scounts, recvbuf, rcounts):
        self._check(scounts); self._check(rcounts)
        T = self.MPI.DOUBLE_COMPLEX
        self.c.Alltoallv([sendbuf, ([int(x) for x in scounts], [int(x) for x in _displs(scounts)]), T],
                         [recvbuf, ([int(x) for x in rcounts], [int(x) for x in _displs(rcounts)]), T])

    def scatterv(self, sendbuf, counts, recvbuf, root=0):
        self._check(counts)
        T = self.MPI.DOUBLE
        spec = [sendbuf, ([int(x) for x in counts], [int(x) for x in _displs(counts)]), T] if self.rank == root else None
        self.c.Scatterv(spec, [recvbuf, T], root=root)

    def gatherv(self, sendbuf, counts, recvbuf, root=0):
        self._check(counts)
        T = self.MPI.DOUBLE
        spec = [recvbuf, ([int(x) for x in counts], [int(x) for x in _displs(counts)]), T] if self.rank == root else None
        self.c.Gatherv([sendbuf, T], spec, root=root)

    def abort(self, code=1):
        self.c.Abort(code)


# ----------------------------------------------------------------------------- serial
class SerialComm:
    rank, size = 0, 1

    def bcast(self, obj, root=0): return obj
    def barrier(self): pass
    def allreduce_max(self, x): return x
    def gather(self, obj, root=0): return [obj]
    def split(self, color): return SerialComm()
    def alltoallv(self, sendbuf, scounts, recvbuf, rcounts): recvbuf[:] = sendbuf
    def scatterv(self, sendbuf, counts, recvbuf, root=0): recvbuf[:] = sendbuf
    def gatherv(self, sendbuf, counts, recvbuf, root=0): recvbuf[:] = sendbuf
    def abort(self, code=1): raise SystemExit(code)


def launcher_ntasks():
    for k in ("SLURM_STEP_NUM_TASKS", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE"):
        v = os.environ.get(k)
        if v and v.isdigit():
            return int(v)
    return 1


def get_world():
    try:
        from mpi4py import MPI
        comm = MPIComm(MPI.COMM_WORLD)
    except ImportError:
        comm = SerialComm()
    if comm.size == 1 and launcher_ntasks() > 1:
        raise SystemExit("ERROR: launcher started %d tasks but MPI size is 1 (mpi4py missing or built "
                         "against another MPI); aborting to avoid independent duplicate runs" % launcher_ntasks())
    return comm


# ----------------------------------------------------------------------------- threads (tests only)
class _World:
    def __init__(self, size):
        self.size = size
        self.bar = threading.Barrier(size)
        self.slots = {}
        self.q = {}
        self.sub = {}
        self.lock = threading.Lock()


class ThreadComm:
    def __init__(self, world, rank):
        self.w, self.rank, self.size = world, rank, world.size

    def barrier(self):
        self.w.bar.wait()

    def _exchange(self, key, obj):
        self.w.slots[(key, self.rank)] = obj
        self.barrier()
        out = [self.w.slots[(key, r)] for r in range(self.size)]
        self.barrier()
        return out

    def bcast(self, obj, root=0):
        return self._exchange("bcast", obj if self.rank == root else None)[root]

    def allreduce_max(self, x):
        return max(self._exchange("max", x))

    def gather(self, obj, root=0):
        out = self._exchange("gather", obj)
        return out if self.rank == root else None

    def split(self, color):
        colors = self._exchange("split", color)
        members = [r for r in range(self.size) if colors[r] == color]
        with self.w.lock:
            key = (id(self.w), color, tuple(members))
            if key not in self.w.sub:
                self.w.sub[key] = _World(len(members))
            sub = self.w.sub[key]
        self.barrier()
        return ThreadComm(sub, members.index(self.rank))

    def alltoallv(self, sendbuf, scounts, recvbuf, rcounts):
        sd, rd = _displs(scounts), _displs(rcounts)
        parts = self._exchange("a2a", [sendbuf[sd[q]:sd[q] + scounts[q]].copy() for q in range(self.size)])
        for r in range(self.size):
            blk = parts[r][self.rank]
            assert blk.size == rcounts[r]
            recvbuf[rd[r]:rd[r] + rcounts[r]] = blk

    def scatterv(self, sendbuf, counts, recvbuf, root=0):
        d = _displs(counts)
        src = self._exchange("scat", sendbuf if self.rank == root else None)[root]
        recvbuf[:] = src[d[self.rank]:d[self.rank] + counts[self.rank]]

    def gatherv(self, sendbuf, counts, recvbuf, root=0):
        parts = self._exchange("gath", sendbuf.copy())
        if self.rank == root:
            d = _displs(counts)
            for r in range(self.size):
                recvbuf[d[r]:d[r] + counts[r]] = parts[r]

    def abort(self, code=1):
        raise SystemExit(code)


def run_threads(fn, size):
    w = _World(size)
    errs = []

    def tgt(r):
        try:
            fn(ThreadComm(w, r))
        except BaseException as e:   # noqa: BLE001
            import traceback
            traceback.print_exc()
            errs.append(e)
            w.bar.abort()

    th = [threading.Thread(target=tgt, args=(r,)) for r in range(size)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errs:
        raise errs[0]
