"""In-process, thread-based stand-in for an MPI communicator (testing only)."""
import threading, queue, numpy as np
class World:
    def __init__(self, size):
        self.size = size; self.bar = threading.Barrier(size)
        self.slots = {}; self.q = {}; self.lock = threading.Lock()
    def comm(self, rank): return ThreadComm(self, rank)
class ThreadComm:
    def __init__(self, world, rank): self.w = world; self.rank = rank; self.size = world.size
    def barrier(self): self.w.bar.wait()
    def bcast(self, obj, root=0):
        if self.rank == root: self.w.slots['bcast'] = obj
        self.barrier(); o = self.w.slots['bcast']; self.barrier(); return o
    def allreduce_max(self, x):
        self.w.slots[('red', self.rank)] = x; self.barrier()
        r = max(self.w.slots[('red', i)] for i in range(self.size)); self.barrier(); return r
    def alltoallv(self, sendbuf, scounts, recvbuf, rcounts):
        sd = np.concatenate(([0], np.cumsum(scounts)[:-1])); rd = np.concatenate(([0], np.cumsum(rcounts)[:-1]))
        for q in range(self.size):
            self.w.slots[('a2a', self.rank, q)] = sendbuf[sd[q]:sd[q]+scounts[q]].copy()
        self.barrier()
        for r in range(self.size):
            blk = self.w.slots[('a2a', r, self.rank)]
            assert blk.size == rcounts[r], (blk.size, rcounts[r])
            recvbuf[rd[r]:rd[r]+rcounts[r]] = blk
        self.barrier()
    def _q(self, src, dst, tag):
        with self.w.lock:
            return self.w.q.setdefault((src, dst, tag), queue.Queue())
    def Send(self, arr, dest, tag): self._q(self.rank, dest, tag).put(np.array(arr, copy=True))
    def Recv(self, arr, source, tag):
        x = self._q(source, self.rank, tag).get(); assert x.shape == arr.shape; arr[...] = x
def run_parallel(fn, size):
    w = World(size); errs = []
    def tgt(r):
        try: fn(w.comm(r))
        except BaseException as e:
            import traceback; traceback.print_exc(); errs.append(e); w.bar.abort()
    th = [threading.Thread(target=tgt, args=(r,)) for r in range(size)]
    [t.start() for t in th]; [t.join() for t in th]
    if errs: raise errs[0]
