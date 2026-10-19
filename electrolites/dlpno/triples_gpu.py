"""Closed-shell DLPNO-(T0) and iterative (T) on the device.

Equations as ``ref_rhf`` (Jiang et al. 2024, Eqs. 53, 109-112).  Triplets
are processed in chunks of equal padded shape: every per-triplet loop of the
reference becomes a batched gather or a strided-batched GEMM over the chunk,
and the triplet natural orbitals of a chunk come from two batched
eigensolver calls (``linalg.eigh_batched``).

Shapes in a chunk: B triplets, A auxiliary functions (padded), P PAOs, V
TNOs, L LMOs of the triplet domain; pair quantities are padded to the
largest pair (PP PAOs, VP PNOs).
"""
import numpy as np
import cupy as cp
from electrolites.dlpno.gather import blocks as gather_blocks, xtbx
from electrolites.dlpno.linalg import chol_inv_batched
import cupyx

from electrolites.dlpno.ragged import padded, Layout, Ragged
from electrolites.dlpno.linalg import eigh_batched, orthonormal_domain

# the six orders of (i, j, k) in W (Jiang Eq. 109) and the transpose that
# brings each partial W back to (a, b, c)
_PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
_BACK = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 0, 1), (1, 2, 0), (2, 1, 0)]


def _perm_case(a1, a2, a3):
    """Psi4 triples_permuter: index into _CASES of the transpose used."""
    if a1 <= a2 <= a3:
        return 0
    if a1 <= a3 <= a2:
        return 1
    if a2 <= a1 <= a3:
        return 2
    if a2 <= a3 <= a1:
        return 3
    if a3 <= a1 <= a2:
        return 4
    return 5


_CASES = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 0, 1), (1, 2, 0), (2, 1, 0)]


def _energy_coeffs(i, j, k):
    """E_ijk = pref * sum_c coef[c] * <V^{case c}, T>, as a length-6 vector."""
    pref = 1.
    if i == j == k:
        pref /= 6.
    elif i == j or j == k or i == k:
        pref /= 2.
    c = np.zeros(6)
    c[0] += 8
    for args, w in (((k, j, i), -4), ((i, k, j), -4), ((j, i, k), -4),
                    ((j, k, i), 2), ((k, i, j), 2)):
        c[_perm_case(*args)] += w
    return pref * c


_EKERN = r'''
// out[b] += sum_{ijk} T[b,i,j,k] * sum_c coef[b,c] * V[b, perm_c(i,j,k)]
// perm_c: V transposed by _CASES[c] (axis n of the transpose is axis cs[c][n] of V)
extern "C" __global__ void tri_energy(const double* __restrict__ V, const double* __restrict__ T,
                                      const double* __restrict__ coef, double* out, const int n)
{
    const int cs[6][3] = {{0,1,2},{0,2,1},{1,0,2},{2,0,1},{1,2,0},{2,1,0}};
    const int b = blockIdx.y;
    const long long n3 = (long long)n * n * n;
    const double* Vb = V + b * n3;
    const double* Tb = T + b * n3;
    double cf[6];
    for (int c = 0; c < 6; ++c) cf[c] = coef[6 * b + c];
    double acc = 0.;
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x; e < n3;
         e += (long long)gridDim.x * blockDim.x) {
        const double t = Tb[e];
        if (t == 0.) continue;
        int idx[3] = {(int)(e / ((long long)n * n)), (int)((e / n) % n), (int)(e % n)};
        double v = 0.;
        for (int c = 0; c < 6; ++c) {
            if (cf[c] == 0.) continue;
            int j[3];
            j[cs[c][0]] = idx[0]; j[cs[c][1]] = idx[1]; j[cs[c][2]] = idx[2];
            v += cf[c] * Vb[((long long)j[0] * n + j[1]) * n + j[2]];
        }
        acc += t * v;
    }
    __shared__ double red[256];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int w = 128; w > 0; w >>= 1) {
        if (threadIdx.x < w) red[threadIdx.x] += red[threadIdx.x + w];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out + b, red[0]);
}
'''
_ek = None


def _batched_energy(V, T3, coef):
    """per triplet sum_c coef[b, c] <V_b transposed by case c, T3_b>: one fused
    pass over T3 (V read at the six permuted positions)."""
    global _ek
    if _ek is None:
        _ek = cp.RawKernel(_EKERN, 'tri_energy')
    B, n = V.shape[0], V.shape[1]
    out = cp.zeros(B)
    if B == 0:
        return out
    V = cp.ascontiguousarray(V)
    T3 = cp.ascontiguousarray(T3)
    coef = cp.ascontiguousarray(cp.asarray(coef, dtype=cp.float64))
    nblk = int(min(64, max(1, (n ** 3 + 255) // 256)))
    _ek((nblk, B), (256,), (V, T3, coef, out, np.int32(n)))
    return out


class TriplesGPU:
    """Mixin for rhf.DLPNO replacing the reference's tno_transform /
    lccsd_t0 / lccsd_t_iterations."""

    MAX_CHUNK_BYTES = None

    def _chunk_bytes(self):
        """batch budget of the triplet chunks: an eighth of the free device memory (>= 1 GB)."""
        if self.MAX_CHUNK_BYTES:
            return self.MAX_CHUNK_BYTES
        free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        return max(1 << 30, int(free // 8))

    # ---- pair tables --------------------------------------------------
    def _pair_tables(self):
        """Padded device copies of every pair's PNOs (in its PAO domain),
        amplitudes and pair density, and of the singles."""
        if getattr(self, '_pt_src', None) is self.T:
            return
        self._pt_src = self.T
        pl = self.pl
        npair = len(pl)
        nv = np.array([x.shape[1] for x in self.X])
        npo = np.array([len(p) for p in self.pao])
        VP, PP = int(padded(nv.max())), int(padded(npo.max()))
        Xp = np.zeros((npair, PP, VP))
        dom = np.zeros((npair, PP), np.int64)
        dmk = np.zeros((npair, PP))
        Tp = np.zeros((npair, VP, VP))
        for p in range(npair):
            Xp[p, :npo[p], :nv[p]] = self.X[p]
            dom[p, :npo[p]] = self.pao[p]
            dmk[p, :npo[p]] = 1.
            Tp[p, :nv[p], :nv[p]] = self.T[p]
        Tp_d = cp.asarray(Tp)
        Tt = 2 * Tp_d - Tp_d.transpose(0, 2, 1)
        Dp = cp.matmul(Tt, Tp_d.transpose(0, 2, 1)) + cp.matmul(Tt.transpose(0, 2, 1), Tp_d)
        diag = np.array([i == j for (i, j) in pl.pairs])
        Dp[cp.asarray(diag)] *= .5
        T1 = np.zeros((self.n, VP))
        for i in range(self.n):
            T1[i, :len(self.T1[i])] = self.T1[i]
        self._pt = dict(VP=VP, PP=PP, X=cp.asarray(Xp), dom=cp.asarray(dom),
                        dmk=cp.asarray(dmk), T=Tp_d, D=Dp, T1=cp.asarray(T1))

    def _project(self, Z, bidx, pidx):
        """S(a_t, a_p)^T = X_p^T S_pao[d_p, d_t] X_t for couples (b, p),
        from Z[b] = S_pao[:, d_t] X_t.  -> (C, VP, V)."""
        pt = self._pt
        n = len(pidx)
        PP, VP, V = pt['PP'], pt['VP'], Z.shape[2]
        step = max(1, int((1 << 28) // (8 * PP * (VP + V))))
        from electrolites.dlpno.gather import gather_rows, xT_z_by_pointer
        bidx, pidx = cp.asarray(bidx), cp.asarray(pidx)
        # one row-gather pass and a pointer-batched GEMM on the pair table;
        # bounded temporaries: the gathered PAO rows are (couples, PP, V)
        out = cp.empty((n, VP, V))
        for c0 in range(0, n, step):
            b_, p_ = bidx[c0:c0 + step], pidx[c0:c0 + step]
            Zg = gather_rows(Z, b_, pt['dom'][p_], pt['dmk'][p_])
            out[c0:c0 + step] = xT_z_by_pointer(pt['X'], p_, Zg)
        return out

    # ---- TNOs ------------------------------------------------------------
    def tno_transform(self, t_cut):
        """Deferred: the TNOs of a chunk are built right before its (T0)
        evaluation in ``lccsd_t0`` and dropped afterwards (kept only when the
        iterative (T) needs them)."""
        self._t_cut_tno = t_cut

    def _tno_chunk(self, ch, Pv, t_cut):
        """TNOs of the triplets ``ch`` (all with padded PAO domain Pv).
        Returns X (b, Pv, Nt) in each triplet's PAO domain, e (b, Nt) host,
        nfin (b,) host, and the PAO index tables."""
        opt, pl = self.opt, self.pl
        pt = self._pt
        S_d, F_d = self.S_pao_d, self.F_pao_d
        trip = self._trip_arr
        b = len(ch)
        npao = self._npao_t[ch]
        idx = np.zeros((b, Pv), np.int64)
        msk = np.zeros((b, Pv))
        for s, t in enumerate(ch):
            idx[s, :npao[s]] = self.t_paos[t]
            msk[s, :npao[s]] = 1.
        gi = cp.asarray(idx)
        m = cp.asarray(msk)
        mm = m[:, :, None] * m[:, None, :]
        St = S_d[gi[:, :, None], gi[:, None, :]] * mm
        Ft = F_d[gi[:, :, None], gi[:, None, :]] * mm
        X, keep = orthonormal_domain(St, npao, opt['S_CUT'], getattr(self, 'PAO_ORTHO', 'pivchol'))
        Z = cp.matmul(S_d[:, gi].transpose(1, 0, 2) * m[:, None, :], X)   # (b, nbf, Pv)
        Dt = cp.zeros((b, Pv, Pv))
        tr = trip[ch]
        ar = cp.arange(b)
        for (a, c) in ((0, 1), (1, 2), (0, 2)):
            pp = cp.asarray(pl.ij[tr[:, a], tr[:, c]])
            sp = self._project(Z, ar, pp)                         # (b, VP, Pv)
            Dt += cp.matmul(sp.transpose(0, 2, 1), cp.matmul(pt['D'][pp], sp))
        del Z
        Dt /= 3.
        Dt -= cp.eye(Pv)[None] * (~keep)[:, :, None]
        occ, Vt = eigh_batched(Dt)
        del Dt
        occ, Vt = occ[:, ::-1], Vt[:, :, ::-1]
        nkeep = keep.sum(axis=1)
        cut = cp.asarray(self.tno_scale[ch] * t_cut)
        cnt = cp.sum((cp.abs(occ) >= cut[:, None])
                     & (cp.arange(Pv)[None, :] < nkeep[:, None]), axis=1)
        nkeep, cnt = nkeep.get(), cnt.get()
        nfin = np.maximum(1, np.maximum(cnt, np.minimum(opt['MIN_TNOS'], nkeep)))
        Nt = int(nfin.max())
        cm = (cp.arange(Nt)[None, :] < cp.asarray(nfin)[:, None]).astype(cp.float64)
        Uq = cp.matmul(X, Vt[:, :, :Nt]) * cm[:, None, :]
        Fq = cp.matmul(Uq.transpose(0, 2, 1), cp.matmul(Ft, Uq))
        Fq += 1e6 * cp.eye(Nt)[None] * (1 - cm)[:, :, None]
        et, Wt = eigh_batched(Fq)
        Xt = cp.matmul(Uq, Wt) * cm[:, None, :]
        return Xt, et.get(), nfin, idx, msk

    # ---- (T0) ----------------------------------------------------------
    def _trip_riatoms(self, prescreen):
        opt = self.opt
        mkn = opt['T_CUT_MKN_TRIPLES_PRE'] if prescreen else opt['T_CUT_MKN_TRIPLES']
        ria = [np.where(np.abs(self.mkn[:, i]) > mkn)[0] for i in range(self.n)]
        return [np.unique(np.concatenate([ria[i], ria[j], ria[k]])) for (i, j, k) in self.trip]

    def triples_sparsity(self, prescreen):
        super().triples_sparsity(prescreen)
        self.t_riatoms = self._trip_riatoms(prescreen)

    def lccsd_t0(self, save=False):
        """TNOs and (T0) energy of every triplet, chunk by chunk."""
        self.dfi.cache_clear()
        cp.get_default_memory_pool().free_all_blocks()
        self._pair_tables()
        if getattr(self, 'F_pao_d', None) is None:
            self.F_pao_d = cp.asarray(self.orb.F_pao)
        dfi = self.dfi
        ntrip = len(self.trip)
        self.e_ijk = np.zeros(ntrip)
        if save:
            self.W3, self.V3, self.T3 = [None] * ntrip, [None] * ntrip, [None] * ntrip
            self.X_tno, self.e_tno = [None] * ntrip, [None] * ntrip
        self._trip_arr = np.array(self.trip, dtype=np.int64).reshape(-1, 3)
        self._npao_t = np.array([len(p) for p in self.t_paos], dtype=np.int64)
        self._nl_t = np.array([len(l) for l in self.t_lmos], dtype=np.int64)
        self._naux_t = np.array([sum(len(dfi.atom_ribfs[a]) for a in at)
                                 for at in self.t_riatoms], dtype=np.int64)
        P = padded(self._npao_t)
        nbf = self.orb.nbf
        PP = self._pt['PP']
        from electrolites.dlpno import mgpu
        devs = mgpu.devices()
        items = []
        for Pv in np.unique(P):
            sel = np.where(P == Pv)[0]
            sel = sel[self.dfi.spatial_order([self.t_riatoms[t] for t in sel])]
            Pv = int(Pv)
            per = 8 * (10 * Pv * Pv + 2 * nbf * Pv + 3 * PP * Pv)
            step = max(1, int(self._chunk_bytes() // per))
            # several GPUs: smaller chunks, so that every device gets work
            if len(devs) > 1:
                step = max(1, min(step, -(-len(sel) // (2 * len(devs)))))
            items += [(Pv, sel[c0:c0 + step]) for c0 in range(0, len(sel), step)]
        run = lambda its: mgpu.run_queue(its, lambda w, it: w._t0_item(*it, save=save), devs,
                                         setup=lambda d: self._dev_view())
        if getattr(dfi, 'qab_direct', False):
            # direct (Q|uv): spatial order, regions evaluated once per atom
            order = dfi.spatial_order([np.unique(np.concatenate([self.t_riatoms[t] for t in it[1]]))
                                       for it in items])
            items = [items[k] for k in order]
            needs = []
            for _, ch in items:
                nd = {}
                for t in ch:
                    for a in self.t_riatoms[t]:
                        nd.setdefault(int(a), []).append(self.t_paos[t])
                needs.append({a: np.concatenate(v) for a, v in nd.items()})
            dfi.run_regions(items, needs, run, getattr(self, 'verbose', 0))
        else:
            # largest chunks first (load balance of the device queue)
            items.sort(key=lambda it: -len(it[1]) * it[0] ** 2)
            run(items)
        return float(self.e_ijk.sum())

    def _dev_view(self):
        """this object with its device tables on the current device."""
        from electrolites.dlpno import mgpu
        return mgpu.device_view(self, ('_pt', 'S_pao_d', 'F_pao_d'),
                                sub={'dfi': ('qia', 'qij', 'J2c', 'C_pao_d', 'mask_uv_d')})

    def _t0_item(self, Pv, ch, save):
        nbf = self.orb.nbf
        Xt, et, nfin, pidx, pm = self._tno_chunk(ch, Pv, self._t_cut_tno)
        if save:
            for s, t in enumerate(ch):
                self.X_tno[t] = Xt[s, :self._npao_t[t], :nfin[s]]
                self.e_tno[t] = cp.asarray(et[s, :nfin[s]])
        # (T0) in memory-bounded pieces of the chunk
        V = int(padded(nfin.max()))
        A = int(-(-self._naux_t[ch].max() // 32) * 32)
        L = int(padded(max(1, self._nl_t[ch].max())))
        per = 8 * (A * V * V + 3 * A * Pv + A * A + 14 * V ** 3
                   + 3 * L * V * V + nbf * V + 3 * A * V
                   + 3 * L * self._pt["PP"] * V)
        sub = max(1, int(self._chunk_bytes() // per))
        for s0 in range(0, len(ch), sub):
            sl = slice(s0, s0 + sub)
            Xs = Xt[sl]
            if Xs.shape[2] < V:
                Xs = cp.concatenate([Xs, cp.zeros(Xs.shape[:2] + (V - Xs.shape[2],))], axis=2)
            ev = np.full((len(ch[sl]), V), 1e3)
            for s, t in enumerate(ch[sl]):
                ev[s, :nfin[sl][s]] = et[sl][s, :nfin[sl][s]]
            self._t0_chunk(ch[sl], Xs, ev, nfin[sl], pidx[sl], pm[sl], save)

    def _t0_chunk(self, ch, X, ev, ntv, pidx, pm, save):
        pl, f = self.pl, self.orb.F_lmo
        dfi = self.dfi
        pt = self._pt
        fd = np.diag(f)
        ext = dfi.ext
        S_d = self.S_pao_d
        trip = self._trip_arr
        B = len(ch)
        P, V = X.shape[1], X.shape[2]
        nlv = self._nl_t
        L = int(padded(max(1, nlv[ch].max())))
        A = int(-(-self._naux_t[ch].max() // 32) * 32)
        lidx = np.zeros((B, L), np.int64)
        lmk = np.zeros((B, L))
        aidx = np.zeros((B, A), np.int64)
        amk = np.zeros((B, A))
        coef = np.zeros((B, 6))
        offs = []
        for s, t in enumerate(ch):
            lidx[s, :nlv[t]] = self.t_lmos[t]
            lmk[s, :nlv[t]] = 1.
            ax = dfi.aux_index(self.t_riatoms[t])
            aidx[s, :len(ax)] = ax
            amk[s, :len(ax)] = 1.
            coef[s] = _energy_coeffs(*trip[t])
            o, at_off = 0, {}
            for a in self.t_riatoms[t]:
                at_off[int(a)] = o
                o += len(dfi.atom_ribfs[a])
            offs.append(at_off)
        if True:
            gp, gpm = cp.asarray(pidx), cp.asarray(pm)
            gl, glm = cp.asarray(lidx), cp.asarray(lmk)
            ga, gam = cp.asarray(aidx), cp.asarray(amk)
            ijk = trip[ch]
            # ---- metric -------------------------------------------------
            J = dfi.J2c[ga[:, :, None], ga[:, None, :]] * (gam[:, :, None] * gam[:, None, :])
            J += cp.eye(A)[None] * (1 - gam)[:, :, None]
            Linv = chol_inv_batched(J)
            # ---- (Q|r a), (Q|r l) for r in i, j, k --------------------------
            qv, qo = [], []
            for r in range(3):
                lmo = cp.asarray(ijk[:, r])
                q = dfi.qia[ga[:, :, None], lmo[:, None, None], gp[:, None, :]]
                q *= gam[:, :, None] * gpm[:, None, :]
                q = cp.matmul(Linv, cp.matmul(q, X))                    # (B, A, V)
                qv.append(q)
                o = dfi.qij[ga[:, :, None], lmo[:, None, None], gl[:, None, :]]
                o *= gam[:, :, None] * glm[:, None, :]
                qo.append(cp.matmul(Linv, o))                           # (B, A, L)
            # ---- (Q|ab) in the TNO basis, one aux atom at a time -----------
            qvv = cp.zeros((B, A, V, V))
            atoms = sorted({int(a) for t in ch for a in self.t_riatoms[t]})
            for a in atoms:
                rows = np.array([s for s, t in enumerate(ch) if a in offs[s]], np.int64)
                off = np.array([offs[s][a] for s in rows], np.int64)
                qa, pmap = dfi.qab_block(a, [pidx[rows][pm[rows] > 0]])
                if qa is None:
                    continue
                na = qa.shape[0]
                pos = pmap[pidx[rows]]
                pos = np.where(pm[rows] > 0, pos, 0)
                assert (pos >= 0).all(), 'triplet PAO outside the extended domain'
                gr = cp.asarray(rows)
                gpos = cp.asarray(pos)
                mrow = gpm[gr]
                Xr = X[gr]
                step = max(1, int((1 << 28) // (8 * na * P * P)))
                for r0 in range(0, len(rows), step):
                    sl = slice(r0, r0 + step)
                    blk = gather_blocks(qa, gpos[sl], gpos[sl], mrow[sl], mrow[sl], ru=True)   # (b, P, na, P)
                    xr = Xr[sl]
                    tq = xtbx(blk, xr, xr)
                    oo = cp.asarray(off[sl])[:, None] + cp.arange(na)[None, :]
                    qvv[gr[sl][:, None], oo] = tq
            qvv = cp.matmul(Linv, qvv.reshape(B, A, V * V)).reshape(B, A, V, V)
            # (r a | b c) = sum_Q (L^-T L^-1 q)_Q,ra (Q|bc): qvv is half-fitted,
            # so contract with the half-fitted qv
            Kvvv = [cp.matmul(qv[r].transpose(0, 2, 1), qvv.reshape(B, A, V * V)).reshape(B, V, V, V)
                    for r in range(3)]
            Kov = {(a, c): cp.matmul(qo[a].transpose(0, 2, 1), qv[c])
                   for a in range(3) for c in range(3) if a != c}
            # ---- projected pair amplitudes --------------------------------
            Z = cp.matmul(S_d[:, gp].transpose(1, 0, 2) * gpm[:, None, :], X)   # (B, nbf, V)
            ar = cp.arange(B)
            W = cp.zeros((B, V, V, V))
            # projected amplitudes, shared by the permutations: T_kj (T_jk = T_kj^T,
            # same PNO space) and the domain amplitudes t_il of each position I
            Tc, Tl = {}, {}
            for r, (I, J_, K) in enumerate(_PERMS):
                key = (min(J_, K), max(J_, K))
                if key not in Tc:
                    pkj = cp.asarray(pl.ij[ijk[:, key[1]], ijk[:, key[0]]])
                    s = self._project(Z, ar, pkj)                       # (B, VP, V)
                    Tc[key] = cp.matmul(s.transpose(0, 2, 1), cp.matmul(pt['T'][pkj], s))
                Tkj = Tc[key] if (K, J_) == (key[1], key[0]) else Tc[key].transpose(0, 2, 1)
                w = cp.matmul(Kvvv[I].reshape(B, V * V, V), Tkj.transpose(0, 2, 1))
                # t_il^{ab} (jl|kc) over the triplet's LMO domain
                if I not in Tl:
                    pil = pl.ij[ijk[:, I][:, None], lidx]               # (B, L)
                    pil = np.where(lmk > 0, pil, 0)
                    bb = cp.asarray(np.repeat(np.arange(B), L))
                    gpil = cp.asarray(pil.reshape(-1))
                    s = self._project(Z, bb, gpil)                      # (B*L, VP, V)
                    Til = cp.matmul(s.transpose(0, 2, 1), cp.matmul(pt['T'][gpil], s))
                    Tl[I] = Til.reshape(B, L, V * V) * glm[:, :, None]
                    del s
                w -= cp.matmul(Tl[I].transpose(0, 2, 1), Kov[(J_, K)])
                W += w.reshape(B, V, V, V).transpose((0,) + tuple(a + 1 for a in _BACK[r]))
            # ---- V, T3, energy -------------------------------------------
            t1 = []
            for r in range(3):
                pdiag = cp.asarray(pl.ij[ijk[:, r], ijk[:, r]])
                s = self._project(Z, ar, pdiag)
                t1.append(cp.einsum('bpv,bp->bv', s, pt['T1'][cp.asarray(ijk[:, r])]))
            Kjk = cp.matmul(qv[1].transpose(0, 2, 1), qv[2])
            Kik = cp.matmul(qv[0].transpose(0, 2, 1), qv[2])
            Kij = cp.matmul(qv[0].transpose(0, 2, 1), qv[1])
            Vt = (W + t1[0][:, :, None, None] * Kjk[:, None]
                  + t1[1][:, None, :, None] * Kik[:, :, None, :]
                  + t1[2][:, None, None, :] * Kij[:, :, :, None])
            e = cp.asarray(ev)
            fs = cp.asarray(fd[ijk].sum(axis=1))
            Dn = (e[:, :, None, None] + e[:, None, :, None] + e[:, None, None, :]
                  - fs[:, None, None, None])
            T3 = -W / Dn
            eb = _batched_energy(Vt, T3, cp.asarray(coef)).get()
            self.e_ijk[ch] = eb
            if save:
                for s, t in enumerate(ch):
                    v = ntv[s]
                    self.W3[t] = W[s, :v, :v, :v].copy()
                    self.V3[t] = Vt[s, :v, :v, :v].copy()
                    self.T3[t] = T3[s, :v, :v, :v].copy()
        return float(self.e_ijk.sum())

    # ---- iterative (T) --------------------------------------------------
    def lccsd_t_iterations(self):
        """Iterative (T) (Jiang Eq. 111-112), batched: triplets live in
        padded buckets of equal TNO count; every coupling t <- u of a bucket
        pair and permutation case is one batched 3-mode transformation."""
        opt, f, n = self.opt, self.orb.F_lmo, self.n
        ntrip = len(self.trip)
        S_d = self.S_pao_d
        nv = np.array([x.shape[1] for x in self.X_tno], dtype=np.int64)
        Vp = padded(nv)
        buckets = {int(v): np.where(Vp == v)[0] for v in np.unique(Vp)}
        bslot = np.zeros(ntrip, np.int64)
        for v, mem in buckets.items():
            bslot[mem] = np.arange(len(mem))

        def pack(lst, v, mem, fill=0.):
            out = cp.full((len(mem), v, v, v), fill)
            for s, t in enumerate(mem):
                m = nv[t]
                out[s, :m, :m, :m] = lst[t]
            return out
        W = {v: pack(self.W3, v, m) for v, m in buckets.items()}
        Vv = {v: pack(self.V3, v, m) for v, m in buckets.items()}
        T = {v: pack(self.T3, v, m) for v, m in buckets.items()}
        D = {}
        coef = {}
        fd = np.diag(f)
        for v, mem in buckets.items():
            e = np.full((len(mem), v), 1e3)
            for s, t in enumerate(mem):
                e[s, :nv[t]] = cp.asnumpy(self.e_tno[t])
            e = cp.asarray(e)
            fs = cp.asarray(fd[self._trip_arr[mem]].sum(axis=1))
            D[v] = (e[:, :, None, None] + e[:, None, :, None] + e[:, None, None, :]
                    - fs[:, None, None, None])
            coef[v] = cp.asarray(np.array([_energy_coeffs(*self.trip[t]) for t in mem]))
        self.W3 = self.V3 = self.T3 = None
        # couplings t <- u (one LMO of t replaced by l), vectorised over triplets
        tri = self._trip_arr
        ts = np.sort(tri, axis=1)
        code = lambda M: (M[:, 0] * n + M[:, 1]) * n + M[:, 2]
        cs = code(ts)
        corder = np.argsort(cs)
        csorted = cs[corder]
        CT, CU, CF, CC = [], [], [], []
        for r in (2, 1, 0):
            x = tri[:, r]
            fm = f[:, x].T                                   # (ntrip, n): f[l, x_t]
            ok = (np.abs(fm) >= opt['F_CUT_T']) & (np.arange(n)[None, :] != x[:, None])
            ti, li = np.nonzero(ok)
            key = tri[ti].copy()
            key[:, r] = li
            c_ = code(np.sort(key, axis=1))
            pos = np.minimum(np.searchsorted(csorted, c_), len(csorted) - 1)
            hit = csorted[pos] == c_
            a1, a2, a3 = key[hit, 0], key[hit, 1], key[hit, 2]
            case = np.select([(a1 <= a2) & (a2 <= a3), (a1 <= a3) & (a3 <= a2), (a2 <= a1) & (a1 <= a3),
                              (a2 <= a3) & (a3 <= a1), (a3 <= a1) & (a1 <= a2)], [0, 1, 2, 3, 4], 5)
            CT.append(ti[hit])
            CU.append(corder[pos[hit]])
            CF.append(fm[ti[hit], li[hit]])
            CC.append(case)
        CT, CU = np.concatenate(CT), np.concatenate(CU)
        CF, CC = np.concatenate(CF), np.concatenate(CC)
        # TNO overlaps of every coupling, padded: S_tu = X_t^T S_pao[d_t, d_u] X_u
        # (device overlap generator over the triplets' compact TNO coefficients,
        # once per distinct couple; then padded per group by pointer gathers)
        from electrolites.dlpno.ovgen import OverlapGen
        from electrolites.dlpno.gather import gather2d
        tu = CT * ntrip + CU
        ucode, sid = np.unique(tu, return_inverse=True)
        sid = sid.reshape(-1)
        tt0, uu0 = ucode // ntrip, ucode % ntrip
        gen = OverlapGen(S_d, self.t_paos, [cp.asnumpy(x) for x in self.X_tno])
        LU = Layout(nv[tt0], nv[uu0]) if len(ucode) else None
        SU = Ragged(LU) if len(ucode) else None
        if len(ucode):
            for P_ in gen.prepare(tt0, uu0, out_layout=LU, out_idx=np.arange(len(ucode))):
                gen.run(P_, SU)
        del gen
        plan = []
        if len(CT):
            shp = np.array(LU.shapes, np.int64).reshape(-1, 2)
            base = np.array([d.data.ptr for d in SU.data], np.int64)
            gk = (Vp[CT] * 4096 + Vp[CU]) * 8 + CC
            order = np.argsort(gk, kind='stable')
            ukeys, starts = np.unique(gk[order], return_index=True)
            ends = np.append(starts[1:], len(order))
            pieces = []
            for k0, k1 in zip(starts, ends):
                m0 = order[k0:k1]
                vmax = int(max(Vp[CT[m0[0]]], Vp[CU[m0[0]]]))
                step = max(1, int((1 << 28) // (8 * 4 * vmax ** 3)))      # bounded sweep temporaries
                pieces += [m0[i0:i0 + step] for i0 in range(0, len(m0), step)]
            for mem in pieces:
                tt, uu = CT[mem], CU[mem]
                vt, vu, case = int(Vp[tt[0]]), int(Vp[uu[0]]), int(CC[mem[0]])
                ids = sid[mem]
                g_ = LU.group[ids]
                ptr = base[g_] + 8 * LU.slot[ids] * shp[g_, 0] * shp[g_, 1]
                S = gather2d(cp.asarray(ptr), cp.asarray(shp[g_, 1].astype(np.int32)),
                             cp.asarray(nv[tt].astype(np.int32)), cp.asarray(nv[uu].astype(np.int32)), vt, vu)
                plan.append(dict(vt=vt, vu=vu, case=case, S=S,
                                 st=cp.asarray(bslot[tt]), su=cp.asarray(bslot[uu]), w=cp.asarray(CF[mem])))
        e_old = np.zeros(ntrip)
        e_prev = 0.
        ax = lambda c: (0,) + tuple(a + 1 for a in _CASES[c])
        for it in range(1, opt['MAXITER'] + 1):
            R = {v: W[v] + T[v] * D[v] for v in buckets}
            for g in plan:
                A = T[g['vu']][g['su']].transpose(ax(g['case']))      # (c, Vu, Vu, Vu)
                S = g['S']
                c_, vt, vu = A.shape[0], g['vt'], g['vu']
                A = cp.matmul(S, A.reshape(c_, vu, vu * vu)).reshape(c_, vt, vu, vu)
                A = cp.matmul(A.reshape(c_, vt * vu, vu), S.transpose(0, 2, 1)).reshape(c_, vt, vu, vt)
                A = cp.matmul(S[:, None], A).reshape(c_, vt, vt, vt)
                cupyx.scatter_add(R[vt], g['st'], -g['w'][:, None, None, None] * A)
            rms = 0.
            # every triplet is updated every sweep: Psi4 freezes triplets whose
            # energy changed by less than T_CUT_ITER, which saves nothing in a
            # batched sweep and, with Jacobi updates, stops ~1e-7 Eh short
            for v, mem in buckets.items():
                T[v] = T[v] - R[v] / D[v]
                cnt = cp.asarray(np.maximum(nv[mem], 1) ** 3, dtype=cp.float64)
                r = cp.sqrt(cp.sum(R[v] ** 2, axis=(1, 2, 3)) / cnt)
                rms = max(rms, float(r.max()))
            e_old = self.e_ijk.copy()
            for v, mem in buckets.items():
                self.e_ijk[mem] = _batched_energy(Vv[v], T[v], coef[v]).get()
            E = float(self.e_ijk.sum())
            self.log(f'  GPU LCCSD(T) iter {it:3d} {E:.12f} {E - e_prev:.3e} {rms:.3e}')
            if rms < opt['R_CONVERGENCE'] and abs(E - e_prev) < opt['E_CONVERGENCE']:
                break
            e_prev = E
        else:
            raise RuntimeError('(T) iterations did not converge')
        return E
