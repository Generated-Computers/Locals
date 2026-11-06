"""Qwenputer's computer walked on the GPU: one threadgroup per request runs the 11 computer layers token after token,
decides each next token itself and stops where Qwen is to predict. No host round trip between the steps of a run.

Why this is fast (the decoding pass is spent on what is decided, memory on what is visited):

- The computer's memory is the Gated DeltaNet state: row k of a head is the memory cell its one-hot key k names. A
  token's key writes one row and its query (one-hot, or two-hot for the ROM) reads one or two, so a step touches
  about 3 of a head's 128 rows (measured on every token: keys at most 1 nonzero entry, queries at most 2). Here a
  head updates those rows in place and leaves the other 125 alone: the same values as the full update, which adds
  exactly 0 to them (but for the heads whose state decays, layer 0's counters, updated whole).
- A step's other work is 1.02M multiply-adds over sparse rows (the weights' nonzeros, each a 16-bit column and a
  bf16 value in one word) with the residual (1,327 numbers) in threadgroup memory.
- A token is decided on the device (the output code -> the next token -> its embedding row), so a run of thousands
  of tokens is one launch (bounded: ``walk`` relaunches every ``chunk`` steps, as macOS ends GPU work that holds the
  GPU for seconds while the display needs it).

Every rounding to bf16 is where transformers rounds; ``tests/test_qwenputer.py`` checks the walk against the MLX path
(equal to transformers' modules on every value), state and residual, token for token.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import List, Optional

import mlx.core as mx
import numpy as np

from .qwenputer import CHIP_VOCAB, CONTROL_NAMES, CTRL, FAMILIES, FAMILY_SIZES, HEAD, Chip, ChipState, FusedStep

# A run's tokens at most (then it failed): the toolbox's longest, a 900-token call's text written backwards, takes
# about 200,000 (a minute on a core); its text counted, 80,000 to 110,000
MAX_RUN = 250_000

f32, bf16 = mx.float32, mx.bfloat16

_HEADER = """
inline float r16(float x) { return static_cast<float>(static_cast<bfloat16_t>(x)); }
inline float zval(uint z) { return as_type<float>((z & 0xFFFFu) << 16); }
inline int zcol(uint z) { return int(z >> 16); }
// a sparse row's sum, its entries added one after another in order, read eight (then four) at a time (the reads of
// a step wait on memory one after another otherwise)
template <typename S>
inline float dot_row(const device uint* zz, int j, int je, S src) {
    float acc = 0.0f;
    for (; j + 8 <= je; j += 8) {
        uint w[8];
        float v[8];
        for (int e = 0; e < 8; e++) w[e] = zz[j + e];
        for (int e = 0; e < 8; e++) v[e] = src[zcol(w[e])];
        for (int e = 0; e < 8; e++) acc += zval(w[e]) * v[e];
    }
    for (; j + 4 <= je; j += 4) {
        uint w0 = zz[j], w1 = zz[j + 1], w2 = zz[j + 2], w3 = zz[j + 3];
        float v0 = src[zcol(w0)], v1 = src[zcol(w1)], v2 = src[zcol(w2)], v3 = src[zcol(w3)];
        acc += zval(w0) * v0;
        acc += zval(w1) * v1;
        acc += zval(w2) * v2;
        acc += zval(w3) * v3;
    }
    for (; j < je; j++) { uint w = zz[j]; acc += zval(w) * src[zcol(w)]; }
    return acc;
}
"""

# meta per layer (int32): see Walker._pack
_M = 26

_SOURCE = """
const uint tid = thread_position_in_threadgroup.x;
const int MAXIN = prm[0], MAXOUT = prm[1];
const uint b = threadgroup_position_in_grid.y;
const uint lane = tid % 32, sg = tid / 32;
const uint NSG = TG / 32;
threadgroup float xs[R];
threadgroup float hn[R];
threadgroup float red[32];
threadgroup int tl[NSG * 8];
threadgroup int ctl[4];
device float* S = (device float*)state + b * SSTRIDE;
device float* CS = (device float*)convs + b * CSTRIDE;
device float* P = scr + b * PSTRIDE;
const int nin = nins[b];
const bool free_run = frees[b] != 0;
auto tin = tokens + b * MAXIN;

int cur = tin[0];
int steps = 0;
while (true) {
    // ---- the embedding row of the token being read
    {
        int t = cur;
        if (t < V0) {
            int lo = 0, hi = NSPEC - 1, si = -1;
            while (lo <= hi) { int m = (lo + hi) / 2; int v = I[SPEC_IDS + m]; if (v == t) { si = m; break; } if (v < t) lo = m + 1; else hi = m - 1; }
            for (uint r = tid; r < R; r += TG) {
                float v;
                if (si >= 0) v = Fl[SPEC_ROWS + si * R + r];
                else {
                    int k = I[BITOF + r];
                    v = k >= 0 ? float((bits[t * 16 + k / 8] >> (7 - k % 8)) & 1) : Fl[PLAIN + r];
                }
                xs[r] = v;
            }
        } else {
            int c = t - V0, e = -1, f = 0, p = 0;
            if (c >= EXPL_A && c < EXPL_A + EXPL_N) e = c - EXPL_A;
            else if (c >= CTRL) e = EXPL_N + c - CTRL;
            else { for (int i = 0; i < NFAMS; i++) if (c >= I[FBASE + i]) f = i; p = c - I[FBASE + f]; }
            for (uint r = tid; r < R; r += TG) {
                float v;
                if (e >= 0) v = Fl[EXPL_ROWS + e * R + r];
                else {
                    v = Fl[FAM_BASE + f * R + r];
                    for (int i = 0; i < 16; i++) if ((p >> i) & 1) v += Fl[FAM_DELTA + (f * 16 + i) * R + r];
                }
                xs[r] = r16(v);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int L = 0; L < NL; L++) {
        const device int* m = I + META + L * MSTRIDE;
        for (int part = 0; part < 2; part++) {
            bool attn = part == 0;
            if (attn ? m[0] == 0 : m[16] == 0) continue;
            // the norm (gain 1, over the full hidden size)
            float s = 0.0f;
            for (uint r = tid; r < R; r += TG) s += xs[r] * xs[r];
            s = simd_sum(s);
            if (lane == 0) red[sg] = s;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float ss = 0.0f;
            for (uint i = 0; i < NSG; i++) ss += red[i];
            float inv = metal::precise::rsqrt(ss / float(H) + 1e-6f);
            for (uint r = tid; r < R; r += TG) hn[r] = r16(xs[r] * inv);  // the normed residual
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (attn) {
                const int C = m[1], NH = m[2], NK = m[3], N = m[4];
                const device int* ptr = I + m[5];
                const device uint* z = Z + m[6];
                const device float* cw = Fl + m[7];
                device float* cs = CS + m[25];
                for (uint r = tid; r < uint(N); r += TG) {
                    float acc = 0.0f;
                    for (int j = ptr[r]; j < ptr[r + 1]; j++) { uint w = z[j]; acc += zval(w) * hn[zcol(w)]; }
                    float y = r16(acc);
                    if (r < uint(C)) {
                        float s0 = cs[r], s1 = cs[C + r], s2 = cs[2 * C + r];
                        float c = 0.0f;
                        c = c + s0 * cw[4 * r];
                        c = c + s1 * cw[4 * r + 1];
                        c = c + s2 * cw[4 * r + 2];
                        c = c + y * cw[4 * r + 3];
                        float cb = r16(c);
                        P[r] = r16(cb / (1.0f + metal::precise::exp(-cb)));
                        cs[r] = s1; cs[C + r] = s2; cs[2 * C + r] = y;
                    } else P[r] = y;
                }
                threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
                // the heads: one simdgroup a head; lane l holds entries 4l..4l+3 (keys) and values 4l..4l+3
                device float* O = P + N;
                for (int h = sg; h < NH; h += NSG) {
                    int kh = I[m[8] + h];
                    device float* qs = O + NH * 128 + h * 256;  // (the normed query and key)
                    device float* ks = qs + 128;
                    float qv[4], kv4[4], sq = 0.0f, sk = 0.0f;
                    for (int i = 0; i < 4; i++) {
                        qv[i] = P[kh * 128 + 4 * lane + i];
                        kv4[i] = P[NK * 128 + kh * 128 + 4 * lane + i];
                        sq += qv[i] * qv[i];
                        sk += kv4[i] * kv4[i];
                    }
                    sq = simd_sum(sq);
                    sk = simd_sum(sk);
                    float rq = metal::precise::rsqrt(sq + 1e-6f), rk = metal::precise::rsqrt(sk + 1e-6f);
                    for (int i = 0; i < 4; i++) { qs[4 * lane + i] = qv[i] * rq / 11.313708498984761f; ks[4 * lane + i] = kv4[i] * rk; }
                    simdgroup_barrier(mem_flags::mem_device);
                    float bb = P[C + NH * 128 + h], aa = P[C + NH * 128 + NH + h];
                    float y0 = 1.0f / (1.0f + metal::precise::exp(metal::abs(bb)));
                    float beta = r16((bb < 0) ? y0 : 1.0f - y0);
                    float xa = aa + Fl[m[10] + h];
                    float sp = (xa > 20.0f) ? xa : log1p(metal::precise::exp(metal::min(xa, 20.0f)));
                    float decay = metal::precise::exp(-metal::precise::exp(Fl[m[9] + h]) * sp);
                    device float* st = S + m[24] + h * 16384;  // [key row][value column]
                    float out[4];
                    // the rows this token touches: nonzero key or query entries, ascending
                    threadgroup int* rows = tl + sg * 8;
                    int nr = 0;
                    if (decay == 1.0f) {
                        for (int i = 0; i < 4; i++) {
                            bool touch = ks[4 * lane + i] != 0.0f || qs[4 * lane + i] != 0.0f;
                            uint64_t mask = static_cast<uint64_t>(simd_ballot(touch));
                            for (uint l = 0; l < 32; l++) if ((mask >> l) & 1ull) { if (nr < 8 && lane == 0) rows[nr] = 4 * l + i; nr++; }
                        }
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                    }
                    if (decay == 1.0f && nr <= 8) {
                        // sort the touched rows (at most 8, collected by entry then lane)
                        if (lane == 0) for (int a = 1; a < nr; a++) { int v = rows[a], c = a - 1; while (c >= 0 && rows[c] > v) { rows[c + 1] = rows[c]; c--; } rows[c + 1] = v; }
                        simdgroup_barrier(mem_flags::mem_threadgroup);
                        for (int i = 0; i < 4; i++) {
                            int dv = 4 * lane + i;
                            float kvm = 0.0f;
                            for (int a = 0; a < nr; a++) { int dk = rows[a]; float kk = ks[dk]; if (kk != 0.0f) kvm += st[dk * 128 + dv] * kk; }
                            float vv = P[2 * NK * 128 + h * 128 + dv];
                            float delta = (vv - kvm) * beta;
                            float o = 0.0f;
                            for (int a = 0; a < nr; a++) {
                                int dk = rows[a];
                                float kk = ks[dk], qq = qs[dk];
                                float sv = st[dk * 128 + dv];
                                if (kk != 0.0f) { sv = sv + kk * delta; st[dk * 128 + dv] = sv; }
                                if (qq != 0.0f) o += sv * qq;
                            }
                            out[i] = o;
                        }
                    } else {
                        // every row (a decaying head): as the full update does it, row by row of values
                        for (int dv = 0; dv < 128; dv++) {
                            float sv[4], kvm = 0.0f;
                            for (int i = 0; i < 4; i++) { sv[i] = st[(4 * lane + i) * 128 + dv] * decay; kvm += sv[i] * ks[4 * lane + i]; }
                            kvm = simd_sum(kvm);
                            float delta = (P[2 * NK * 128 + h * 128 + dv] - kvm) * beta;
                            float o = 0.0f;
                            for (int i = 0; i < 4; i++) { sv[i] = sv[i] + ks[4 * lane + i] * delta; o += sv[i] * qs[4 * lane + i]; st[(4 * lane + i) * 128 + dv] = sv[i]; }
                            o = simd_sum(o);
                            if (dv / 4 == int(lane)) out[dv % 4] = o;
                        }
                    }
                    // the gated RMS norm over the head's 128 outputs
                    float s2 = 0.0f;
                    for (int i = 0; i < 4; i++) { out[i] = r16(out[i]); s2 += out[i] * out[i]; }
                    s2 = simd_sum(s2);
                    float rn = metal::precise::rsqrt(s2 / 128.0f + 1e-6f);
                    for (int i = 0; i < 4; i++) {
                        int dv = 4 * lane + i;
                        float zz = P[C + h * 128 + dv];
                        float n = r16(out[i] * rn);
                        n = r16(Fl[m[11] + dv] * n);
                        O[h * 128 + dv] = r16(n * (zz / (1.0f + metal::precise::exp(-zz))));
                    }
                }
                threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
                const device int* orow = I + m[13];
                const device int* optr = I + m[14];
                const device uint* oz = Z + m[15];
                for (uint i = tid; i < uint(m[12]); i += TG) {
                    float acc = 0.0f;
                    for (int j = optr[i]; j < optr[i + 1]; j++) acc += zval(oz[j]) * O[zcol(oz[j])];
                    int r = orow[i];
                    xs[r] = r16(xs[r] + r16(acc));
                }
            } else {
                const int U = m[17];
                const device int* ptr = I + m[18];
                const device uint* z = Z + m[19];
                for (uint u = tid; u < uint(U); u += TG) {
                    float g = 0.0f, up = 0.0f;
                    for (int j = ptr[u]; j < ptr[u + 1]; j++) { uint w = z[j]; g += zval(w) * hn[zcol(w)]; }
                    for (int j = ptr[U + u]; j < ptr[U + u + 1]; j++) { uint w = z[j]; up += zval(w) * hn[zcol(w)]; }
                    g = r16(g);
                    P[u] = r16(r16(g / (1.0f + metal::precise::exp(-g))) * r16(up));
                }
                threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
                const device int* drow = I + m[21];
                const device int* dptr = I + m[22];
                const device uint* dz = Z + m[23];
                for (uint i = tid; i < uint(m[20]); i += TG) {
                    float acc = 0.0f;
                    for (int j = dptr[i]; j < dptr[i + 1]; j++) acc += zval(dz[j]) * P[zcol(dz[j])];
                    int r = drow[i];
                    xs[r] = r16(xs[r] + r16(acc));
                }
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
    }
    if (DEBUG) for (uint r = tid; r < R; r += TG) dbg[(b * MAXOUT + steps) * R + r] = xs[r];
    // ---- the next token: the output code
    int fid = 0, pay = 0;
    bool set = false;
    for (int i = 0; i < NFAM + NPAY; i++) {
        float c = xs[I[CODE + i]];
        set = set || (c != 0.0f);
        int bit = c > 0.0f ? 1 : 0;
        if (i < NFAM) fid |= bit << i; else pay |= bit << (i - NFAM);
    }
    int tok = -1;
    if (set) {
        if (fid < NFAMS) tok = pay < I[FSIZE + fid] ? V0 + I[FBASE + fid] + pay : -2;
        else if (fid < NFAMS + NCTRL) tok = V0 + CTRL + (fid - NFAMS);
        else if (fid == NFAMS + NCTRL && pay < NPRINT) tok = I[PRINT + pay];
        else tok = -2;
    }
    if (tok == -1 && cur >= V0) tok = -3;
    if (tid == 0) pred[b * MAXOUT + steps] = tok;
    steps++;
    int nxt;
    if (steps < nin) nxt = tin[steps];
    else if (free_run && tok >= 0 && steps < MAXOUT) nxt = tok;
    else break;
    cur = nxt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
if (tid == 0) done[b] = steps;
"""


class Walker:
    """The walk kernel and its tables (from a ``Chip``); states as one buffer per batch (in place)."""

    TG = 512  # (the kernel allows at most 704)

    def __init__(self, chip: Chip):
        self.chip = chip
        fused = FusedStep(chip)  # (its sparse rows: the same order of every sum)
        self.R, self.H = fused.R, chip.H
        I: List[np.ndarray] = []
        Z: List[np.ndarray] = []
        Fl: List[np.ndarray] = []
        nI, nZ, nF = [0], [0], [0]

        def put(lst, n, a):
            off = n[0]
            lst.append(a)
            n[0] += a.size
            return off

        def ints(a):
            return put(I, nI, np.asarray(a, np.int32).reshape(-1))

        def floats(a):
            return put(Fl, nF, np.asarray(a, np.float32).reshape(-1))

        def packed(col, val):
            v = np.asarray(val, np.float32)
            bits = (v.view(np.uint32) >> 16).astype(np.uint32)
            assert np.array_equal((bits << 16).view(np.float32), v), "a weight is not a bf16 value"
            return put(Z, nZ, (np.asarray(col, np.uint32) << 16) | bits)

        a = lambda x: np.array(x)  # noqa: E731
        meta = np.zeros((chip.n_layers, _M), np.int32)
        sofs = cofs = 0
        self.heads, self.convs = [], []
        for L, lay in enumerate(fused.layers):
            m = meta[L]
            if "proj" in lay:
                ptr, col, val = (a(x) for x in lay["proj"])
                m[0:5] = [1, lay["C"], lay["nh"], lay["nk"], lay["n"]]
                m[5], m[6] = ints(ptr), packed(col, val)
                m[7] = floats(a(lay["conv"]))
                m[8], m[9], m[10], m[11] = ints(a(lay["kmap"])), floats(a(lay["alog"])), floats(a(lay["dtb"])), floats(a(lay["nw"]))
                wrow, optr, ocol, oval = (a(x) for x in lay["out"])
                rows = np.nonzero(wrow >= 0)[0]
                order = wrow[rows]
                orow = np.empty(len(rows), np.int32)
                orow[order] = rows
                m[12], m[13], m[14], m[15] = len(rows), ints(orow), ints(optr), packed(ocol, oval)
                m[24], m[25] = sofs, cofs
                self.heads.append((L, lay["nh"], sofs))
                self.convs.append((L, lay["C"], cofs))
                sofs += lay["nh"] * HEAD * HEAD
                cofs += 3 * lay["C"]
            if "mlp" in lay:
                ptr, col, val = (a(x) for x in lay["mlp"])
                m[16], m[17], m[18], m[19] = 1, lay["units"], ints(ptr), packed(col, val)
                wrow, dptr, dcol, dval = (a(x) for x in lay["down"])
                rows = np.nonzero(wrow >= 0)[0]
                order = wrow[rows]
                drow = np.empty(len(rows), np.int32)
                drow[order] = rows
                m[20], m[21], m[22], m[23] = len(rows), ints(drow), ints(dptr), packed(dcol, dval)
        self.sstride, self.cstride = sofs, cofs
        self.pstride = max(max(lay.get("n", 0) + 3 * lay.get("nh", 0) * HEAD, lay.get("units", 0)) for lay in fused.layers)
        e = chip._emb
        expl = np.asarray(e["explicit_ids"])
        qs, ctrl = FAMILIES["QS"][0], CTRL
        n_qsd = FAMILIES["QS"][1] + FAMILIES["QD"][1]
        assert FAMILIES["QD"][0] == qs + FAMILIES["QS"][1]
        assert np.array_equal(expl, np.r_[np.arange(qs, qs + n_qsd), np.arange(ctrl, ctrl + len(CONTROL_NAMES))]), \
            "trace tokens kept whole are not QS, QD and the controls"
        bitof = np.full(self.R, -1, np.int32)
        bitof[np.asarray(e["bitcols"])] = np.arange(len(e["bitcols"]))
        consts = dict(
            META=ints(meta), SPEC_IDS=ints(e["special_ids"]), BITOF=ints(bitof), FBASE=ints([b_ for b_, _ in FAMILIES.values()]),
            FSIZE=ints([s for _, s in FAMILY_SIZES]), CODE=ints(np.array(chip.code)), PRINT=ints(chip.printable),
            SPEC_ROWS=floats(e["special_rows"]), PLAIN=floats(e["plain"]), EXPL_ROWS=floats(e["explicit_rows"]),
            FAM_BASE=floats(e["fam_base"]), FAM_DELTA=floats(e["fam_delta"]))
        self.I = mx.array(np.concatenate(I))
        self.Z = mx.array(np.concatenate(Z))
        self.Fl = mx.array(np.concatenate(Fl))
        self.bits = mx.array(np.asarray(e["bits"], np.uint8).reshape(-1))
        self.consts = dict(consts, R=self.R, H=self.H, V0=chip.v0, CTRL=CTRL, NFAM=chip.nfam, NPAY=chip.npay,
                           NFAMS=len(FAMILY_SIZES), NCTRL=len(CONTROL_NAMES), NPRINT=len(chip.printable),
                           NSPEC=len(e["special_ids"]), EXPL_A=qs, EXPL_N=n_qsd, NL=chip.n_layers, MSTRIDE=_M,
                           SSTRIDE=self.sstride, CSTRIDE=self.cstride, PSTRIDE=self.pstride, TG=self.TG)
        self.kernel = mx.fast.metal_kernel(
            name="qwenputer_walk", input_names=["prm", "tokens", "nins", "frees", "I", "Z", "Fl", "bits", "state", "convs"],
            output_names=["pred", "done", "scr", "dbg"], source=_SOURCE, header=_HEADER)

    # states: a ChipState (the MLX layout) <-> the walk's buffers
    def pack(self, st: ChipState):
        """(state, convs) buffers of a ChipState: per row, every head's [key row][value column] state, then the
        convolutions' last three inputs (fp32)."""
        B = next(r for r in st.rec if r is not None).shape[0]
        s = mx.concatenate([st.rec[L].transpose(0, 1, 3, 2).reshape(B, -1) for L, _, _ in self.heads], axis=1)
        c = mx.concatenate([st.conv[L].astype(f32).reshape(B, -1) for L, _, _ in self.convs], axis=1)
        s, c = mx.array(s), mx.array(c)  # (copies of their own: the walk writes them in place)
        mx.eval(s, c)
        return s, c

    def masks(self, s: mx.array) -> mx.array:
        """Per row and head, the state rows that are not all 0 (4 words of bits a head): the decaying heads update
        those only."""
        a = np.array(s).reshape(s.shape[0], -1, HEAD, HEAD)  # (B, heads, key row, value column)
        nz = (a != 0).any(-1)  # (B, heads, 128)
        bits = np.packbits(nz.reshape(s.shape[0], -1, 4, 32)[..., ::-1], axis=-1, bitorder="big")
        words = bits.view(">u4").astype(np.uint32).reshape(s.shape[0], -1)
        return mx.array(np.ascontiguousarray(words))

    def unpack(self, s: mx.array, c: mx.array, st: ChipState) -> None:
        B = s.shape[0]
        for L, nh, off in self.heads:
            st.rec[L] = s[:, off:off + nh * HEAD * HEAD].reshape(B, nh, HEAD, HEAD).transpose(0, 1, 3, 2)
        for L, C, off in self.convs:
            st.conv[L] = c[:, off:off + 3 * C].reshape(B, 3, C).astype(bf16)

    def __call__(self, s: mx.array, c: mx.array, tokens: List[List[int]], free: List[bool], max_out: int,
                 debug: bool = False):
        """Read each row's ``tokens`` (teacher-forced), then, if ``free``, go on from the predicted tokens while the
        computer predicts; ``s``, ``c`` (from ``pack``) are updated in place. Returns (pred (B, max_out): the token
        predicted after each token read, -1 where Qwen predicts, -2 / -3 a failed code; steps (B,); residuals
        after the computer layers (B, max_out, R) if ``debug``)."""
        B = len(tokens)
        maxin = max(len(t) for t in tokens)
        tok = np.zeros((B, maxin), np.int32)
        for i, t in enumerate(tokens):
            tok[i, :len(t)] = t
        consts = dict(self.consts, DEBUG=bool(debug))
        pred, done, _, dbg = self.kernel(
            inputs=[mx.array(np.array([maxin, max_out], np.int32)), mx.array(tok), mx.array(np.array([len(t) for t in tokens], np.int32)), mx.array(np.array(free, np.int32)),
                    self.I, self.Z, self.Fl, self.bits, s, c],
            template=list(consts.items()), grid=(self.TG, B, 1), threadgroup=(self.TG, 1, 1),
            output_shapes=[(B, max_out), (B,), (B, self.pstride), (B, max_out, self.R) if debug else (1,)],
            output_dtypes=[mx.int32, mx.int32, f32, f32], init_value=-4)
        return pred, done, (dbg if debug else None)



def _head(V: str, Z: str, BB: str, AA: str, ALOG: str, DTB: str, NW: str, ON: str, OUT: str) -> str:
    """A value head's step, by one simdgroup (lane l: key entries and values 4l..4l+3), from the raw query and key in
    threadgroup ``qa``, ``ka``. The state ``st`` ([key row][value column]) is updated in place on the rows the token
    touches (nonzero key or query entries: at most 3 measured; more than 8 take every row) and, where the head
    decays, on the rows its mask ``mk`` (4 words) has seen written: the others are 0 and stay 0. ``OUT`` stores
    ``val``, the output for value ``dv``."""
    return f"""
{{
    float sq = 0.0f, sk = 0.0f;
    for (int i = 0; i < 4; i++) {{ sq += qa[4 * lane + i] * qa[4 * lane + i]; sk += ka[4 * lane + i] * ka[4 * lane + i]; }}
    sq = simd_sum(sq);
    sk = simd_sum(sk);
    float rq = metal::precise::rsqrt(sq + 1e-6f), rk = metal::precise::rsqrt(sk + 1e-6f);
    for (int i = 0; i < 4; i++) {{ qa[4 * lane + i] = qa[4 * lane + i] * rq / 11.313708498984761f; ka[4 * lane + i] = ka[4 * lane + i] * rk; }}
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float bb = {BB}, aa = {AA};
    float y0 = 1.0f / (1.0f + metal::precise::exp(metal::abs(bb)));
    float beta = r16((bb < 0) ? y0 : 1.0f - y0);
    float xa = aa + {DTB};
    float sp = (xa > 20.0f) ? xa : log1p(metal::precise::exp(metal::min(xa, 20.0f)));
    float decay = metal::precise::exp(-metal::precise::exp({ALOG}) * sp);
    bool on = {ON};
    float out[4];
    // the touched rows, ascending: lane l holds rows 4l..4l+3, so a prefix sum over the lanes orders them
    int mine = 0;
    for (int i = 0; i < 4; i++) mine += (ka[4 * lane + i] != 0.0f || qa[4 * lane + i] != 0.0f) ? 1 : 0;
    int at = simd_prefix_exclusive_sum(mine);
    int nr = simd_sum(mine);
    for (int i = 0; i < 4; i++) if (ka[4 * lane + i] != 0.0f || qa[4 * lane + i] != 0.0f) {{ if (at < 8) rows[at] = 4 * lane + i; at++; }}
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (nr <= 8) {{
        uint vm[4] = {{mk[0], mk[1], mk[2], mk[3]}};
        for (int i = 0; i < 4; i++) {{
            int dv = 4 * lane + i;
            float kvm = 0.0f;
            for (int a = 0; a < nr; a++) {{ int dk = rows[a]; float kk = ka[dk]; if (kk != 0.0f) kvm += (st[dk * 128 + dv] * decay) * kk; }}
            float delta = ({V} - kvm) * beta;
            float o = 0.0f;
            for (int a = 0; a < nr; a++) {{
                int dk = rows[a];
                float kk = ka[dk], qq = qa[dk];
                float sv = st[dk * 128 + dv] * decay;
                if (kk != 0.0f) sv = sv + kk * delta;
                if (on && (kk != 0.0f || decay != 1.0f)) st[dk * 128 + dv] = sv;
                if (qq != 0.0f) o += sv * qq;
            }}
            if (on && decay != 1.0f) {{
                for (int w = 0; w < 4; w++) {{
                    uint bits = vm[w];
                    while (bits) {{
                        int dk = 32 * w + ctz(bits);
                        bits &= bits - 1;
                        bool touched = false;
                        for (int a = 0; a < nr; a++) touched = touched || rows[a] == dk;
                        if (!touched) st[dk * 128 + dv] = st[dk * 128 + dv] * decay;
                    }}
                }}
            }}
            out[i] = o;
        }}
        if (on && lane == 0) for (int a = 0; a < nr; a++) if (ka[rows[a]] != 0.0f) mk[rows[a] / 32] |= 1u << (rows[a] % 32);
    }} else {{
        for (int dv = 0; dv < 128; dv++) {{
            float sv[4], kvm = 0.0f;
            for (int i = 0; i < 4; i++) {{ sv[i] = st[(4 * lane + i) * 128 + dv] * decay; kvm += sv[i] * ka[4 * lane + i]; }}
            kvm = simd_sum(kvm);
            float delta = ({V} - kvm) * beta;
            float o = 0.0f;
            for (int i = 0; i < 4; i++) {{ sv[i] = sv[i] + ka[4 * lane + i] * delta; o += sv[i] * qa[4 * lane + i]; if (on) st[(4 * lane + i) * 128 + dv] = sv[i]; }}
            o = simd_sum(o);
            if (dv / 4 == int(lane)) out[dv % 4] = o;
        }}
        if (on && lane == 0) for (int dk = 0; dk < 128; dk++) if (ka[dk] != 0.0f) mk[dk / 32] |= 1u << (dk % 32);
    }}
    float s2 = 0.0f;
    for (int i = 0; i < 4; i++) {{ out[i] = r16(out[i]); s2 += out[i] * out[i]; }}
    s2 = simd_sum(s2);
    float rn = metal::precise::rsqrt(s2 / 128.0f + 1e-6f);
    for (int i = 0; i < 4; i++) {{
        int dv = 4 * lane + i;
        float z = {Z};
        float n = r16(out[i] * rn);
        n = r16({NW} * n);
        float val = r16(n * (z / (1.0f + metal::precise::exp(-z))));
        {OUT}
    }}
}}
"""


# -- the computer spread over the GPU: two kernels a layer, the tokens decided on the device ------------------------
# One threadgroup cannot hide the memory latency of a step (2.3 ms a step above: the walk is exact, its shape is
# wrong). Here a layer's attention is one kernel with a threadgroup per value head (its own 514 projection rows,
# its recurrence on the rows it touches, in place) and its MLP one kernel over the units. Each kernel first adds
# the previous kernel's output to the residual (a few thousand multiply-adds, done by every threadgroup), so no
# kernel only adds; then 20 kernels and the token's kernel make a step, and the GPU runs ``sync`` steps between two
# looks of the host. Buffers live on the device and are updated in place; a 1-element output chains the kernels in
# order.

_RESID = """
for (uint r = tid; r < R; r += TGS) xs[r] = xin[b * R + r];
threadgroup_barrier(mem_flags::mem_threadgroup);
if (PN > 0) {
    // a thread a short row (most have 1 to 20 entries: a simdgroup a row walked them one after another, 166 rows a
    // simdgroup for the last layer's 1,326), a simdgroup a long one (to 979 entries); each row writes its own dim
    for (uint k = tid; k < uint(PSN); k += TGS) {  // (each: its entries' start and end, its dim)
        int j = I[PSL + 3 * k], je = I[PSL + 3 * k + 1], r = I[PSL + 3 * k + 2];
        float acc = dot_row(Z + PNNZ, j, je, scr + b * SCR + PSRC);
        xs[r] = r16(xs[r] + r16(acc));
    }
    for (uint k = tid / 32; k < uint(PLN); k += TGS / 32) {
        int i = I[PLL + k];
        const int je = I[PPTR + i + 1];
        const device uint* zz = Z + PNNZ;
        const device float* src = scr + b * SCR + PSRC;
        float acc = 0.0f;
        int j = I[PPTR + i] + int(tid % 32);
        for (; j + 96 < je; j += 128) {  // (four of the lane's entries read at once, added in order)
            uint w0 = zz[j], w1 = zz[j + 32], w2 = zz[j + 64], w3 = zz[j + 96];
            float v0 = src[zcol(w0)], v1 = src[zcol(w1)], v2 = src[zcol(w2)], v3 = src[zcol(w3)];
            acc += zval(w0) * v0;
            acc += zval(w1) * v1;
            acc += zval(w2) * v2;
            acc += zval(w3) * v3;
        }
        for (; j < je; j += 32) { uint w = zz[j]; acc += zval(w) * src[zcol(w)]; }
        acc = simd_sum(acc);
        if (tid % 32 == 0) { int r = I[PROWS + i]; xs[r] = r16(xs[r] + r16(acc)); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""

_NORM = """
{
    float s = 0.0f;
    for (uint r = tid; r < R; r += TGS) s += xs[r] * xs[r];
    s = simd_sum(s);
    if (lane == 0) red[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0.0f;
    for (uint i = 0; i < TGS / 32; i++) ss += red[i];
    float inv = metal::precise::rsqrt(ss / float(H) + 1e-6f);
    for (uint r = tid; r < R; r += TGS) hn[r] = r16(xs[r] * inv);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""

# A value head's step in the stepper: _head's operations, each in its order, over 128 threads (a value column each)
# instead of one simdgroup walking four columns after one another; a thread reads the state of the rows the token
# touches once (they are read again for the update in _head: the same values) and writes them back. Simdgroup 0
# normalizes the query and key, lists the touched rows and makes the gates, and normalizes the output, as _head.
_HEAD_STEP = """
{
    device float* st = (device float*)state + b * SSTRIDE + m[24] + h * 16384;
    device uint* mk = (device uint*)masks + b * MASKS + (m[24] / 16384 + h) * 4;
    threadgroup float gates[2], ov[128];
    threadgroup int nrows[1];
    if (sg == 0) {
        float sq = 0.0f, sk = 0.0f;
        for (int i = 0; i < 4; i++) { sq += qa[4 * lane + i] * qa[4 * lane + i]; sk += ka[4 * lane + i] * ka[4 * lane + i]; }
        sq = simd_sum(sq);
        sk = simd_sum(sk);
        float rq = metal::precise::rsqrt(sq + 1e-6f), rk = metal::precise::rsqrt(sk + 1e-6f);
        for (int i = 0; i < 4; i++) { qa[4 * lane + i] = qa[4 * lane + i] * rq / 11.313708498984761f; ka[4 * lane + i] = ka[4 * lane + i] * rk; }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        int mine = 0;
        for (int i = 0; i < 4; i++) mine += (ka[4 * lane + i] != 0.0f || qa[4 * lane + i] != 0.0f) ? 1 : 0;
        int at = simd_prefix_exclusive_sum(mine);
        int nr = simd_sum(mine);
        for (int i = 0; i < 4; i++) if (ka[4 * lane + i] != 0.0f || qa[4 * lane + i] != 0.0f) { if (at < 8) rows[at] = 4 * lane + i; at++; }
        if (lane == 0) {
            float bb = ab[0], aa = ab[1];
            float y0 = 1.0f / (1.0f + metal::precise::exp(metal::abs(bb)));
            gates[0] = r16((bb < 0) ? y0 : 1.0f - y0);
            float xa = aa + Fl[m[10] + h];
            float sp = (xa > 20.0f) ? xa : log1p(metal::precise::exp(metal::min(xa, 20.0f)));
            gates[1] = metal::precise::exp(-metal::precise::exp(Fl[m[9] + h]) * sp);
            nrows[0] = nr;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float beta = gates[0], decay = gates[1];
    const int nr = nrows[0];
    const bool on = act[b] != 0;
    if (nr <= 8) {
        if (tid < 128) {
            const int dv = int(tid);
            const uint vm0 = mk[0], vm1 = mk[1], vm2 = mk[2], vm3 = mk[3];
            float sv[8];
            #pragma unroll
            for (int a = 0; a < 8; a++) if (a < nr) sv[a] = st[rows[a] * 128 + dv];
            float kvm = 0.0f;
            #pragma unroll
            for (int a = 0; a < 8; a++) if (a < nr) { float kk = ka[rows[a]]; if (kk != 0.0f) kvm += (sv[a] * decay) * kk; }
            float delta = (va[dv] - kvm) * beta;
            float o = 0.0f;
            #pragma unroll
            for (int a = 0; a < 8; a++) if (a < nr) {
                int dk = rows[a];
                float kk = ka[dk], qq = qa[dk];
                float v = sv[a] * decay;
                if (kk != 0.0f) v = v + kk * delta;
                if (on && (kk != 0.0f || decay != 1.0f)) st[dk * 128 + dv] = v;
                if (qq != 0.0f) o += v * qq;
            }
            if (on && decay != 1.0f) {
                for (int w = 0; w < 4; w++) {
                    uint bits = w == 0 ? vm0 : w == 1 ? vm1 : w == 2 ? vm2 : vm3;
                    while (bits) {
                        int dk = 32 * w + ctz(bits);
                        bits &= bits - 1;
                        bool touched = false;
                        for (int a = 0; a < nr; a++) touched = touched || rows[a] == dk;
                        if (!touched) st[dk * 128 + dv] = st[dk * 128 + dv] * decay;
                    }
                }
            }
            ov[dv] = o;
        }
    } else if (sg == 0) {  // (more than 8 rows: every row, as _head)
        for (int dv = 0; dv < 128; dv++) {
            float sv[4], kvm = 0.0f;
            for (int i = 0; i < 4; i++) { sv[i] = st[(4 * lane + i) * 128 + dv] * decay; kvm += sv[i] * ka[4 * lane + i]; }
            kvm = simd_sum(kvm);
            float delta = (va[dv] - kvm) * beta;
            float o = 0.0f;
            for (int i = 0; i < 4; i++) { sv[i] = sv[i] + ka[4 * lane + i] * delta; o += sv[i] * qa[4 * lane + i]; if (on) st[(4 * lane + i) * 128 + dv] = sv[i]; }
            o = simd_sum(o);
            if (lane == 0) ov[dv] = o;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // (every column's masks read before they change)
    if (sg == 0) {
        if (on && lane == 0) {
            if (nr <= 8) { for (int a = 0; a < nr; a++) if (ka[rows[a]] != 0.0f) mk[rows[a] / 32] |= 1u << (rows[a] % 32); }
            else for (int dk = 0; dk < 128; dk++) if (ka[dk] != 0.0f) mk[dk / 32] |= 1u << (dk % 32);
        }
        float out[4], s2 = 0.0f;
        for (int i = 0; i < 4; i++) { out[i] = r16(ov[4 * lane + i]); s2 += out[i] * out[i]; }
        s2 = simd_sum(s2);
        float rn = metal::precise::rsqrt(s2 / 128.0f + 1e-6f);
        for (int i = 0; i < 4; i++) {
            int dv = 4 * lane + i;
            float z = za[dv];
            float n = r16(out[i] * rn);
            n = r16(Fl[m[11] + dv] * n);
            sc[b * SCR + OOFF + h * 128 + dv] = r16(n * (z / (1.0f + metal::precise::exp(-z))));
        }
    }
}
"""

_ATTN = ("""
const uint tid = thread_position_in_threadgroup.x;
const uint h = threadgroup_position_in_grid.y, b = threadgroup_position_in_grid.z;
const uint lane = tid % 32, sg = tid / 32;
threadgroup float xs[R], hn[R], qa[128], ka[128], va[128], za[128], red[32], ab[2];
threadgroup int rows[8];
const device int* m = I + META + L * MSTRIDE;
const bool on = act[b] != 0;
device float* xo = (device float*)xout;
device float* sc = (device float*)scr;
""" + _RESID + """
if (h == 0) for (uint r = tid; r < R; r += TGS) xo[b * R + r] = xs[r];
""" + _NORM + """
const int C = m[1], NH = m[2], NK = m[3];
const device int* ptr = I + m[5];
const device uint* zz = Z + m[6];
const device float* cw = Fl + m[7];
const int kh = I[m[8] + h];
const bool kw = I[KWRITE + h] != 0;
const device float* csi = cin + b * CSTRIDE + m[25];
device float* cso = (device float*)cout + b * CSTRIDE + m[25];
for (uint i = tid; i < 514; i += TGS) {
    int r, kind;
    if (i < 128) { r = kh * 128 + i; kind = 0; }
    else if (i < 256) { r = NK * 128 + kh * 128 + (i - 128); kind = 1; }
    else if (i < 384) { r = 2 * NK * 128 + h * 128 + (i - 256); kind = 2; }
    else if (i < 512) { r = C + h * 128 + (i - 384); kind = 3; }
    else { r = C + NH * 128 + (i - 512) * NH + h; kind = 4; }
    float y = r16(dot_row(zz, ptr[r], ptr[r + 1], hn));
    if (kind <= 2) {
        float s0 = csi[r], s1 = csi[C + r], s2 = csi[2 * C + r];
        float c = 0.0f;
        c = c + s0 * cw[4 * r];
        c = c + s1 * cw[4 * r + 1];
        c = c + s2 * cw[4 * r + 2];
        c = c + y * cw[4 * r + 3];
        float cb = r16(c);
        float v = r16(cb / (1.0f + metal::precise::exp(-cb)));
        if (kind == 2 || kw) { cso[r] = on ? s1 : s0; cso[C + r] = on ? s2 : s1; cso[2 * C + r] = on ? y : s2; }
        if (kind == 0) qa[i] = v; else if (kind == 1) ka[i - 128] = v; else va[i - 256] = v;
    } else if (kind == 3) za[i - 384] = y;
    else ab[i - 512] = y;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
""" + _HEAD_STEP + """
if (tid == 0 && h == 0) dep2[0] = 0.0f;
""")

_MLP = """
const uint tid = thread_position_in_threadgroup.x;
const uint blk = threadgroup_position_in_grid.y, b = threadgroup_position_in_grid.z;
const uint lane = tid % 32, sg = tid / 32;
threadgroup float xs[R], hn[R], red[32];
const device int* m = I + META + L * MSTRIDE;
device float* xo = (device float*)xout;
device float* sc = (device float*)scr;
""" + _RESID + """
if (blk == 0) for (uint r = tid; r < R; r += TGS) xo[b * R + r] = xs[r];
""" + _NORM + """
const int U = m[17];
const device int* ptr = I + m[18];
const device uint* zz = Z + m[19];
uint u = blk * TGS + tid;
if (u < uint(U)) {
    float g = dot_row(zz, ptr[u], ptr[u + 1], hn), up = dot_row(zz, ptr[U + u], ptr[U + u + 1], hn);
    g = r16(g);
    sc[b * SCR + POFF + u] = r16(r16(g / (1.0f + metal::precise::exp(-g))) * r16(up));
}
if (tid == 0 && blk == 0) dep2[0] = 0.0f;
"""

# the token: the residual after the computer (last MLP's output added), its output code, the next input row
_TOKEN = """
const uint tid = thread_position_in_threadgroup.x;
const uint b = threadgroup_position_in_grid.y;
const uint lane = tid % 32, sg = tid / 32;
threadgroup float xs[R];
threadgroup int nx[3];
device float* xo = (device float*)xout;
device int* cur = (device int*)curs;
device int* actw = (device int*)act;
device int* cnt = (device int*)cnts;
device int* pr = (device int*)pred;
""" + _RESID + """
if (tid == 0) {
    int fid = 0, pay = 0;
    bool set = false;
    for (int i = 0; i < NFAM + NPAY; i++) {
        float c = xs[I[CODE + i]];
        set = set || (c != 0.0f);
        int bit = c > 0.0f ? 1 : 0;
        if (i < NFAM) fid |= bit << i; else pay |= bit << (i - NFAM);
    }
    int tok = -1;
    if (set) {
        if (fid < NFAMS) tok = pay < I[FSIZE + fid] ? V0 + I[FBASE + fid] + pay : -2;
        else if (fid < NFAMS + NCTRL) tok = V0 + CTRL + (fid - NFAMS);
        else if (fid == NFAMS + NCTRL && pay < NPRINT) tok = I[PRINT + pay];
        else tok = -2;
    }
    if (tok == -1 && cur[b] >= V0) tok = -3;
    bool on = actw[b] != 0;
    int fz = on ? forced[b * KMAX + cnt[b]] : -1;
    if (on) { pr[b * KMAX + cnt[b]] = tok; cnt[b] = cnt[b] + 1; }
    if (fz >= 0) tok = fz;
    bool go = on && tok >= 0;
    actw[b] = go ? 1 : 0;
    if (go) cur[b] = tok;
    nx[0] = go ? tok : -1;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const int t = nx[0];
if (t >= 0) {
    if (t < V0) {
        int lo = 0, hi = NSPEC - 1, si = -1;
        while (lo <= hi) { int mm = (lo + hi) / 2; int v = I[SPEC_IDS + mm]; if (v == t) { si = mm; break; } if (v < t) lo = mm + 1; else hi = mm - 1; }
        for (uint r = tid; r < R; r += TGS) {
            float v;
            if (si >= 0) v = Fl[SPEC_ROWS + si * R + r];
            else { int k = I[BITOF + r]; v = k >= 0 ? float((bits[t * 16 + k / 8] >> (7 - k % 8)) & 1) : Fl[PLAIN + r]; }
            xo[b * R + r] = v;
        }
    } else {
        int c = t - V0, e = -1, f = 0, p = 0;
        if (c >= EXPL_A && c < EXPL_A + EXPL_N) e = c - EXPL_A;
        else if (c >= CTRL) e = EXPL_N + c - CTRL;
        else { for (int i = 0; i < NFAMS; i++) if (c >= I[FBASE + i]) f = i; p = c - I[FBASE + f]; }
        for (uint r = tid; r < R; r += TGS) {
            float v;
            if (e >= 0) v = Fl[EXPL_ROWS + e * R + r];
            else {
                v = Fl[FAM_BASE + f * R + r];
                for (int i = 0; i < 16; i++) if ((p >> i) & 1) v += Fl[FAM_DELTA + (f * 16 + i) * R + r];
            }
            xo[b * R + r] = r16(v);
        }
    }
}
if (tid == 0 && b == 0) dep2[0] = 0.0f;
"""


_COMPILE = os.environ.get("LOCALS_QWENPUTER_COMPILE", "1") != "0"


def _padded(v) -> np.ndarray:
    """An int32 buffer of at least 64 entries: MLX passes smaller inputs in the constant address space, which a
    kernel cannot write."""
    a = np.zeros(max(len(v), 64), np.int32)
    a[:len(v)] = np.asarray(v, np.int32)
    return a


class Stepper:
    """The computer's decoding on the device (see above), on a ``Walker``'s tables and state buffers."""

    TGS = 256

    def __init__(self, walker: Walker):
        self.w = W = walker
        fused_layers = FusedStep(W.chip).layers
        I0 = np.array(W.I)
        meta = I0[W.consts["META"]:W.consts["META"] + W.chip.n_layers * _M].reshape(-1, _M)
        extra: List[int] = []
        base = int(W.I.size)
        # the scratch per row: each attention layer's head outputs, each MLP layer's units
        scr, self.kernels = 0, []
        prev = None  # (rows, ptr, nnz, count, scratch offset) of the previous kernel's output
        mk = mx.fast.metal_kernel
        common = ["dep", "I", "Z", "Fl", "xin", "xout", "scr"]
        self.k_attn = mk(name="qwenputer_attn", input_names=common + ["state", "cin", "cout", "act", "masks"], output_names=["dep2"],
                         source=_ATTN, header=_HEADER)
        self.k_mlp = mk(name="qwenputer_mlp", input_names=common, output_names=["dep2"], source=_MLP, header=_HEADER)
        self.k_token = mk(name="qwenputer_token", input_names=common + ["bits", "curs", "act", "cnts", "pred", "forced"],
                          output_names=["dep2"], source=_TOKEN, header=_HEADER)
        def split(pend):
            """The rows a kernel adds to the residual: the short ones (a thread each), the long ones (a simdgroup)."""
            _, ptr, _, n, _ = pend
            lens = np.diff(I0[ptr:ptr + n + 1]) if n else np.zeros(0, np.int32)
            short, long_ = [i for i in range(n) if lens[i] <= 32], [i for i in range(n) if lens[i] > 32]
            out = dict(PSN=len(short), PSL=base + len(extra))
            rows_ = I0[pend[0]:pend[0] + n] if n else []
            for i in short:  # (start, end, dim: the row's place read at once)
                extra.extend([int(I0[ptr + i]), int(I0[ptr + i + 1]), int(rows_[i])])
            out.update(PLN=len(long_), PLL=base + len(extra))
            extra.extend(long_)
            return out

        for L, lay in enumerate(fused_layers):
            m = meta[L]
            pend = prev or (0, 0, 0, 0, 0)
            if m[0]:
                nh = int(m[2])
                kmap = np.array(lay["kmap"]).tolist()
                writer = [1 if kmap.index(kmap[h]) == h else 0 for h in range(nh)]
                kw = base + len(extra)
                extra += writer
                self.kernels.append(("attn", L, nh, dict(L=L, OOFF=scr, KWRITE=kw, PROWS=pend[0], PPTR=pend[1], PNNZ=pend[2], PN=pend[3], PSRC=pend[4],
                                                         **split(pend))))
                prev = (int(m[13]), int(m[14]), int(m[15]), int(m[12]), scr)
                scr += nh * HEAD
                pend = prev
            if m[16]:
                U = int(m[17])
                self.kernels.append(("mlp", L, -(-U // self.TGS), dict(L=L, POFF=scr, PROWS=pend[0], PPTR=pend[1], PNNZ=pend[2], PN=pend[3],
                                                                       PSRC=pend[4], **split(pend))))
                prev = (int(m[21]), int(m[22]), int(m[23]), int(m[20]), scr)
                scr += U
        self.token = dict(PROWS=prev[0], PPTR=prev[1], PNNZ=prev[2], PN=prev[3], PSRC=prev[4], **split(prev))
        assert len(self.kernels) % 2 == 0, "the residual's two buffers alternate: an even number of layer kernels"
        self.scr = scr
        self.I = mx.concatenate([W.I, mx.array(np.array(extra, np.int32))]) if extra else W.I
        self.consts = dict(W.consts, SCR=scr, TGS=self.TGS, MASKS=W.sstride // (HEAD * HEAD) * 4)
        self._tmpl = None
        self._blocks: dict = {}  # steps a block -> the block compiled (mx.compile: one per batch size, cached)

    def start(self, s: mx.array, c: mx.array, first: List[int], rows: Optional[List[int]] = None, mk: Optional[mx.array] = None):
        """Device buffers for runs from each row's token in ``first`` (rows: the ones that run); ``mk``: the
        state's row masks (Walker.masks), updated in place."""
        B = len(first)
        on = [True] * B if rows is None else [i in set(rows) for i in range(B)]
        x0 = np.array(self.w.chip.embed([[t if on[i] else 0] for i, t in enumerate(first)]).astype(f32)).reshape(B, -1)
        own = lambda a: mx.array(np.ascontiguousarray(a))  # noqa: E731 (a buffer of its own: the kernels write it in place)
        bufs = dict(s=s, c=[c, own(np.zeros(c.shape, np.float32))], x=[own(x0), own(np.zeros_like(x0))],
                    scr=own(np.zeros((B, max(self.scr, 1)), np.float32)), act=own(_padded(on)), cur=own(_padded(first)),
                    B=B, dep=own(np.zeros(1, np.float32)), parity=0, mk=self.w.masks(s) if mk is None else mk)
        mx.eval([v for k, v in bufs.items() if isinstance(v, mx.array)] + bufs["c"] + bufs["x"])
        return bufs

    def steps(self, bufs: dict, k: int, forced: Optional[np.ndarray] = None):
        """Enqueue k steps; returns (dep, pred (B, k), count (B,)): evaluate dep, then read pred and count. ``forced``
        (B, k): a token >= 0 is read next in place of the predicted one (teacher forcing)."""
        B = bufs["B"]
        pred = mx.array(np.full((B, max(k, 64)), -4, np.int32))
        cnt = mx.array(_padded([0] * B))
        f = np.full((B, max(k, 64)), -1, np.int32)
        if forced is not None:
            f[:, :forced.shape[1]] = forced
        frc = mx.array(f)
        mx.eval(pred, cnt, frc)
        p = bufs["parity"]
        c0, c1 = bufs["c"][p], bufs["c"][1 - p]
        args = (bufs["dep"], bufs["x"][0], bufs["x"][1], bufs["scr"], bufs["s"], c0, c1, bufs["act"], bufs["mk"], bufs["cur"], cnt, pred, frc)
        if k % 2 == 0 and k <= 64 and _COMPILE:
            # a block of steps compiled once (for its length and batch): enqueueing it took the host 255 us a step
            # (21 kernels), held the GIL that the batch engine's thread needs, and left the GPU waiting
            f = self._blocks.get(k)
            if f is None:
                f = self._blocks[k] = mx.compile(lambda *a, k=k: self._enqueue(*a, k=k))
            dep = f(*args)
        else:
            dep = self._enqueue(*args, k=k)
            bufs["parity"] = (p + k) % 2
        bufs["dep"] = dep
        return dep, pred, cnt

    def _enqueue(self, dep, x0, x1, sc, s, c0, c1, act, mk, cur, cnt, pred, frc, k: int):
        """k steps' kernels after ``dep``, the convolutions' state in c0 (then c1, alternately)."""
        W, B = self.w, x0.shape[0]
        if self._tmpl is None:  # (each kernel's template made once)
            g = list(self.consts.items())
            self._tmpl = [g + list(kc.items()) for _, _, _, kc in self.kernels] + [g + list(self.token.items())]
        tm, one, tg, f1 = self._tmpl, [(1,)], (self.TGS, 1, 1), [f32]
        tok_t = tm[-1] + [("KMAX", max(k, 64))]
        I, Z, Fl = self.I, W.Z, W.Fl
        cs = (c0, c1)
        for step in range(k):
            cin, cout = cs[step % 2], cs[1 - step % 2]
            for i, (kind, L, n, kc) in enumerate(self.kernels):
                xin, xout = (x0, x1) if i % 2 == 0 else (x1, x0)
                if kind == "attn":
                    dep = self.k_attn(inputs=[dep, I, Z, Fl, xin, xout, sc, s, cin, cout, act, mk], template=tm[i],
                                      grid=(self.TGS, n, B), threadgroup=tg, output_shapes=one, output_dtypes=f1)[0]
                else:
                    dep = self.k_mlp(inputs=[dep, I, Z, Fl, xin, xout, sc], template=tm[i],
                                     grid=(self.TGS, n, B), threadgroup=tg, output_shapes=one, output_dtypes=f1)[0]
            dep = self.k_token(inputs=[dep, I, Z, Fl, x0, x0, sc, W.bits, cur, act, cnt, pred, frc], template=tok_t,
                               grid=(self.TGS, B, 1), threadgroup=tg, output_shapes=one, output_dtypes=f1)[0]
        return dep

    def run(self, s: mx.array, c: mx.array, first: List[int], rows: Optional[List[int]] = None, max_tokens: int = MAX_RUN,
            sync: int = 32, mk: Optional[mx.array] = None):
        """As FusedStep.run, on the walker's buffers (s and mk updated in place): (the rows' runs, the convolutions'
        state after them)."""
        bufs = self.start(s, c, first, rows, mk)
        B = bufs["B"]
        live = set(range(B)) if rows is None else set(rows)
        out = {i: [] for i in live}
        n, k = 0, 4
        while live and n < max_tokens:
            dep, pred, cnt = self.steps(bufs, k)
            mx.eval(dep)
            n += k
            k = min(2 * k, sync)
            P, N = np.array(pred), np.array(cnt)
            for i in list(live):
                for t in P[i, :N[i]].tolist():
                    if t < 0:
                        if t != -1:
                            out[i].append(t)
                        live.discard(i)
                        break
                    out[i].append(t)
        for i in live:  # (cut at max_tokens: failed, as the Pool and the CPU end one)
            out[i].append(-5)
        return [out[i] for i in sorted(out)], bufs["c"][bufs["parity"]]


# -- a block of tokens in one pass (a prompt, Qwen's text): the same sums, over T tokens at once -----------------------
_B_NORM = """
const int TP = (tn[0] + 3) / 4 * 4;
uint tid = thread_position_in_threadgroup.x, t = threadgroup_position_in_grid.y;
uint lane = tid % 32, sg = tid / 32;
threadgroup float red[32];
float s = 0.0f;
for (uint r = tid; r < R; r += TGS) { float v = x[t * R + r]; s += v * v; }
s = simd_sum(s);
if (lane == 0) red[sg] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
float ss = 0.0f;
for (uint i = 0; i < TGS / 32; i++) ss += red[i];
float inv = metal::precise::rsqrt(ss / float(H) + 1e-6f);
for (uint r = tid; r < R; r += TGS) hn[r * TP + t] = r16(x[t * R + r] * inv);  // (transposed: tokens of a column together)
"""

_B_PROJ = """
// thread: one row, 4 tokens; a simdgroup: one row, 128 tokens, so a column's 128 values come in one read
const int TN = tn[0], TP = (TN + 3) / 4 * 4;
uint q = thread_position_in_grid.x, r = thread_position_in_grid.y;
if (r >= uint(NR) || 4 * q >= uint(TP)) return;
const device int* ptr = I + PTR;
const device uint* zz = Z + NNZ;
const device float4* h4 = (const device float4*)hn;
float4 acc = 0.0f;
for (int j = ptr[r]; j < ptr[r + 1]; j++) { uint w = zz[j]; acc += zval(w) * h4[zcol(w) * (TP / 4) + q]; }
float4 up = 0.0f;
if (GATE) for (int j = ptr[NR + r]; j < ptr[NR + r + 1]; j++) { uint w = zz[j]; up += zval(w) * h4[zcol(w) * (TP / 4) + q]; }
for (int i = 0; i < 4; i++) {
    uint t = 4 * q + i;
    if (t >= uint(TN)) break;
    if (GATE) {
        float g = r16(acc[i]);
        y[t * NR + r] = r16(r16(g / (1.0f + metal::precise::exp(-g))) * r16(up[i]));
    } else y[t * NR + r] = r16(acc[i]);
}
"""

_B_CONV = """
const int T = tn[0];
uint c = thread_position_in_grid.x, t = thread_position_in_grid.y;
if (c >= uint(CC)) return;
float v[4];
for (int k = 0; k < 4; k++) { int at = int(t) - 3 + k; v[k] = at < 0 ? cs[(at + 3) * CC + c] : y[at * NR + c]; }
float s = 0.0f;
s = s + v[0] * cw[4 * c];
s = s + v[1] * cw[4 * c + 1];
s = s + v[2] * cw[4 * c + 2];
s = s + v[3] * cw[4 * c + 3];
float cb = r16(s);
p[t * CC + c] = r16(cb / (1.0f + metal::precise::exp(-cb)));
if (t == uint(T) - 1) for (int k = 0; k < 3; k++) { int at = int(T) - 3 + k; cso[k * CC + c] = at < 0 ? cs[(at + 3) * CC + c] : y[at * NR + c]; }
"""

_B_HEAD = ("""
const int T = tn[0];
uint lane = thread_position_in_threadgroup.x, h = threadgroup_position_in_grid.y;
threadgroup float qa[128], ka[128];
threadgroup int rows[8];
const device int* m = I + META + L * MSTRIDE;
const int C = m[1], NH = m[2], NK = m[3], N = m[4];
const int kh = I[m[8] + h];
device float* st = (device float*)state + m[24] + h * 16384;
device uint* mk = (device uint*)masks + (m[24] / 16384 + h) * 4;
for (int t = 0; t < T; t++) {
    for (int i = 0; i < 4; i++) { qa[4 * lane + i] = p[t * C + kh * 128 + 4 * lane + i]; ka[4 * lane + i] = p[t * C + NK * 128 + kh * 128 + 4 * lane + i]; }
    simdgroup_barrier(mem_flags::mem_threadgroup);
""" + _head("p[t * C + 2 * NK * 128 + h * 128 + dv]", "y[t * N + C + h * 128 + dv]", "y[t * N + C + NH * 128 + h]",
            "y[t * N + C + NH * 128 + NH + h]", "Fl[m[9] + h]", "Fl[m[10] + h]", "Fl[m[11] + dv]", "true",
            "ho[t * NH * 128 + h * 128 + dv] = val;") + """
    simdgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);  // (the state and mask, for the next token)
}
""")

_B_ADD = """
uint r = thread_position_in_grid.x, t = thread_position_in_grid.y;
if (r >= uint(R)) return;
int i = I[WMAP + r];
float v = x[t * R + r];
if (i >= 0) {
    const device int* ptr = I + PTR;
    const device uint* zz = Z + NNZ;
    float acc = 0.0f;
    for (int j = ptr[i]; j < ptr[i + 1]; j++) { uint w = zz[j]; acc += zval(w) * src[t * K + zcol(w)]; }
    v = r16(v + r16(acc));
}
xo[t * R + r] = v;
"""


# The heads of a block, spread over (head, value column) instead of one simdgroup a head walking every token: the
# computer's memory is a set of independent rows (a one-hot key writes one, a query reads one or two), so each value
# column of a head can follow the tokens alone, touching at most 3 rows a token, with every operation and its order
# as in _head. Four kernels: the queries' and keys' norms and the rows each token touches (by token and head), each
# head's active tokens (_BH_ACT, below), the recurrence (by head and column), the gated norm (by token and head). A
# token touching more than 8 rows (none measured) sends the layer back to the walking kernel (_B_HEAD).
_BH_PREP = """
uint lane = thread_position_in_threadgroup.x, h = threadgroup_position_in_grid.y, t = threadgroup_position_in_grid.z;
const device int* m = I + META + L * MSTRIDE;
const int C = m[1], NH = m[2], NK = m[3], N = m[4];
const int kh = I[m[8] + h];
float qa[4], ka[4], sq = 0.0f, sk = 0.0f;
for (int i = 0; i < 4; i++) {
    qa[i] = p[t * C + kh * 128 + 4 * lane + i];
    ka[i] = p[t * C + NK * 128 + kh * 128 + 4 * lane + i];
    sq += qa[i] * qa[i];
    sk += ka[i] * ka[i];
}
sq = simd_sum(sq);
sk = simd_sum(sk);
float rq = metal::precise::rsqrt(sq + 1e-6f), rk = metal::precise::rsqrt(sk + 1e-6f);
for (int i = 0; i < 4; i++) { qa[i] = qa[i] * rq / 11.313708498984761f; ka[i] = ka[i] * rk; }
int mine = 0;
for (int i = 0; i < 4; i++) mine += (ka[i] != 0.0f || qa[i] != 0.0f) ? 1 : 0;
int at = simd_prefix_exclusive_sum(mine);
int nr = simd_sum(mine);
uint o = (t * NH + h) * 8;
for (int i = 0; i < 4; i++) if (ka[i] != 0.0f || qa[i] != 0.0f) {
    if (at < 8) { rows[o + at] = 4 * lane + i; kv[o + at] = ka[i]; qv[o + at] = qa[i]; }
    at++;
}
if (lane == 0) {
    float bb = y[t * N + C + NH * 128 + h], aa = y[t * N + C + NH * 128 + NH + h];
    float y0 = 1.0f / (1.0f + metal::precise::exp(metal::abs(bb)));
    float xa = aa + Fl[m[10] + h];
    float sp = (xa > 20.0f) ? xa : log1p(metal::precise::exp(metal::min(xa, 20.0f)));
    bd[2 * (t * NH + h)] = r16((bb < 0) ? y0 : 1.0f - y0);
    bd[2 * (t * NH + h) + 1] = metal::precise::exp(-metal::precise::exp(Fl[m[9] + h]) * sp);
    cnt[t * NH + h] = nr;
}
"""

# Most of a head's tokens touch none of its rows (8 of a layer's 4,096 head outputs are nonzero a token, measured), and
# such a token leaves a head whose state does not decay as it was, its output 0: each head walks only its active
# tokens (touching a row, or every token where the state decays), listed in order by _BH_ACT; the recurrence then
# reads a tile of their values and rows at once into threadgroup memory instead of one dependent read after another.
_BH_ACT = """
const int T = tn[0];
uint tid = thread_position_in_threadgroup.x, h = threadgroup_position_in_grid.y;
const device int* m = I + META + L * MSTRIDE;
const int NH = m[2];
const int PER = (T + 1023) / 1024, t0 = int(tid) * PER;
int mine = 0;
for (int k = 0; k < PER; k++) {
    int t = t0 + k;
    if (t < T) { uint e = t * NH + h; mine += (cnt[e] > 0 || bd[2 * e + 1] != 1.0f) ? 1 : 0; }
}
threadgroup int sums[32];
threadgroup atomic_uint rm[4];
uint lane = tid % 32, sg = tid / 32;
if (tid < 4) atomic_store_explicit(&rm[tid], 0u, memory_order_relaxed);
int at = simd_prefix_exclusive_sum(mine);
int tot = simd_sum(mine);
if (lane == 0) sums[sg] = tot;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
    int v = sums[lane];
    int all = simd_sum(v);
    sums[lane] = simd_prefix_exclusive_sum(v);
    if (lane == 0) nact[h] = all;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
at += sums[sg];
for (int k = 0; k < PER; k++) {
    int t = t0 + k;
    if (t < T) {
        uint e = t * NH + h;
        if (cnt[e] > 0 || bd[2 * e + 1] != 1.0f) act[h * T + at++] = t;
        for (int a = 0; a < min(cnt[e], 8); a++) atomic_fetch_or_explicit(&rm[rows[8 * e + a] / 32], 1u << (rows[8 * e + a] % 32), memory_order_relaxed);
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid < 4) rmask[h * 4 + tid] = atomic_load_explicit(&rm[tid], memory_order_relaxed);
"""

_BH_REC = """
const int T = tn[0];
uint dv = thread_position_in_grid.x, h = thread_position_in_grid.y;
const device int* m = I + META + L * MSTRIDE;
const int C = m[1], NH = m[2], NK = m[3];
device float* st = (device float*)state + m[24] + h * 16384;
device uint* mk = (device uint*)masks + (m[24] / 16384 + h) * 4;
// (four words, never an array indexed by data: such an array lives in memory, and every token waited for it)
uint vm0 = mk[0], vm1 = mk[1], vm2 = mk[2], vm3 = mk[3];
#define PICK(a, w) ((w) == 0 ? a##0 : (w) == 1 ? a##1 : (w) == 2 ? a##2 : a##3)
#define SEEN(r) { int w_ = (r) >> 5; uint b_ = 1u << ((r) & 31); vm0 |= w_ == 0 ? b_ : 0u; vm1 |= w_ == 1 ? b_ : 0u; vm2 |= w_ == 2 ? b_ : 0u; vm3 |= w_ == 3 ? b_ : 0u; }
const int n = nact[h];
threadgroup int tt[TILE], tc[TILE], tr[TILE * 8];
threadgroup float tb[TILE], tdc[TILE], tk[TILE * 8], tq[TILE * 8], tv[TILE * 128];
// the rows the block touches (its tokens' rows, and the rows written before, which a decaying state shrinks): in
// threadgroup memory while it runs when there are at most ROWS of them, else read and written where they are
threadgroup float sc[ROWS * 128];
const uint U0 = rmask[h * 4] | vm0, U1 = rmask[h * 4 + 1] | vm1, U2 = rmask[h * 4 + 2] | vm2, U3 = rmask[h * 4 + 3] | vm3;
const int pc0 = 0, pc1 = popcount(U0), pc2 = pc1 + popcount(U1), pc3 = pc2 + popcount(U2), nu = pc3 + popcount(U3);
const bool held = nu <= ROWS;
#define SLOT(dk) (PICK(pc, (dk) >> 5) + popcount(PICK(U, (dk) >> 5) & ((1u << ((dk) & 31)) - 1u)))
#define AT(dk) (held ? sc[SLOT(dk) * 128 + dv] : st[(dk) * 128 + dv])
#define PUT(dk, v) { if (held) sc[SLOT(dk) * 128 + dv] = (v); else st[(dk) * 128 + dv] = (v); }
if (held) for (int w = 0; w < 4; w++) { uint bits = PICK(U, w); while (bits) { int dk = 32 * w + ctz(bits); bits &= bits - 1; sc[SLOT(dk) * 128 + dv] = st[dk * 128 + dv]; } }
for (int base = 0; base < n; base += TILE) {
    const int len = min(TILE, n - base);
    threadgroup_barrier(mem_flags::mem_threadgroup);  // (the last tile read)
    if (dv < uint(len)) {
        int t = act[h * T + base + dv];
        uint e = t * NH + h;
        tt[dv] = t;
        tc[dv] = min(cnt[e], 8);  // (past 8 the pass is made again by the walking kernel)
        tb[dv] = bd[2 * e];
        tdc[dv] = bd[2 * e + 1];
        for (int a = 0; a < 8; a++) { tr[dv * 8 + a] = rows[8 * e + a]; tk[dv * 8 + a] = kv[8 * e + a]; tq[dv * 8 + a] = qv[8 * e + a]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < len; i++) tv[i * 128 + dv] = p[tt[i] * C + 2 * NK * 128 + h * 128 + dv];  // (its own column)
    for (int i = 0; i < len; i++) {
        const int t = tt[i], nr = tc[i];
        const float beta = tb[i], decay = tdc[i];
        float kvm = 0.0f;
        for (int a = 0; a < nr; a++) { float kk = tk[8 * i + a]; if (kk != 0.0f) kvm += (AT(tr[8 * i + a]) * decay) * kk; }
        float delta = (tv[i * 128 + dv] - kvm) * beta;
        float o = 0.0f;
        for (int a = 0; a < nr; a++) {
            int dk = tr[8 * i + a];
            float kk = tk[8 * i + a], qq = tq[8 * i + a];
            float sv = AT(dk) * decay;
            if (kk != 0.0f) sv = sv + kk * delta;
            if (kk != 0.0f || decay != 1.0f) PUT(dk, sv);
            if (qq != 0.0f) o += sv * qq;
        }
        if (decay != 1.0f) {
            for (int w = 0; w < 4; w++) {
                uint bits = PICK(vm, w);
                while (bits) {
                    int dk = 32 * w + ctz(bits);
                    bits &= bits - 1;
                    bool touched = false;
                    for (int a = 0; a < nr; a++) touched = touched || tr[8 * i + a] == dk;
                    if (!touched) PUT(dk, AT(dk) * decay);
                }
            }
        }
        yr[t * NH * 128 + h * 128 + dv] = o;
        for (int a = 0; a < nr; a++) if (tk[8 * i + a] != 0.0f) SEEN(tr[8 * i + a]);
    }
}
if (held) for (int w = 0; w < 4; w++) { uint bits = PICK(U, w); while (bits) { int dk = 32 * w + ctz(bits); bits &= bits - 1; st[dk * 128 + dv] = sc[SLOT(dk) * 128 + dv]; } }
if (dv == 0) { mk[0] = vm0; mk[1] = vm1; mk[2] = vm2; mk[3] = vm3; }
#undef PUT
#undef AT
#undef SLOT
#undef SEEN
#undef PICK
"""

_BH_NORM = """
uint lane = thread_position_in_threadgroup.x, h = threadgroup_position_in_grid.y, t = threadgroup_position_in_grid.z;
const device int* m = I + META + L * MSTRIDE;
const int C = m[1], NH = m[2], N = m[4];
float out[4], s2 = 0.0f;
const uint e = t * NH + h;
const bool on = cnt[e] > 0 || bd[2 * e + 1] != 1.0f;  // (a token the recurrence skipped: its output is 0)
for (int i = 0; i < 4; i++) { out[i] = r16(on ? yr[t * NH * 128 + h * 128 + 4 * lane + i] : 0.0f); s2 += out[i] * out[i]; }
s2 = simd_sum(s2);
float rn = metal::precise::rsqrt(s2 / 128.0f + 1e-6f);
for (int i = 0; i < 4; i++) {
    int dv = 4 * lane + i;
    float z = y[t * N + C + h * 128 + dv];
    float n = r16(out[i] * rn);
    n = r16(Fl[m[11] + dv] * n);
    ho[t * NH * 128 + h * 128 + dv] = r16(n * (z / (1.0f + metal::precise::exp(-z))));
}
"""


class Block:
    """T tokens through the computer layers in one pass (``Walker``'s tables and state buffers, updated in place):
    returns the residual after the computer layers at every position."""

    TGS = 256

    def __init__(self, walker: Walker):
        self.w = W = walker
        mk = mx.fast.metal_kernel
        self.k_norm = mk(name="qp_b_norm", input_names=["x", "tn"], output_names=["hn"], source=_B_NORM, header=_HEADER)
        self.k_proj = mk(name="qp_b_proj", input_names=["hn", "I", "Z", "tn"], output_names=["y"], source=_B_PROJ, header=_HEADER)
        self.k_conv = mk(name="qp_b_conv", input_names=["y", "cs", "cw", "tn"], output_names=["p", "cso"], source=_B_CONV, header=_HEADER)
        self.k_head = mk(name="qp_b_head", input_names=["p", "y", "I", "Fl", "state", "masks", "tn"], output_names=["ho"], source=_B_HEAD, header=_HEADER)
        self.k_add = mk(name="qp_b_add", input_names=["src", "I", "Z", "x"], output_names=["xo"], source=_B_ADD, header=_HEADER)
        self.k_prep = mk(name="qp_bh_prep", input_names=["p", "y", "I", "Fl"], output_names=["rows", "kv", "qv", "bd", "cnt"],
                         source=_BH_PREP, header=_HEADER)
        self.k_act = mk(name="qp_bh_act", input_names=["cnt", "bd", "rows", "I", "tn"], output_names=["act", "nact", "rmask"],
                        source=_BH_ACT, header=_HEADER)
        self.k_rec = mk(name="qp_bh_rec", input_names=["p", "I", "rows", "kv", "qv", "bd", "cnt", "act", "nact", "rmask", "state",
                                                       "masks", "tn"], output_names=["yr"], source=_BH_REC, header=_HEADER)
        self.k_hnorm = mk(name="qp_bh_norm", input_names=["yr", "y", "I", "Fl", "cnt", "bd"], output_names=["ho"], source=_BH_NORM,
                          header=_HEADER)
        I0 = np.array(W.I)
        self.meta = I0[W.consts["META"]:W.consts["META"] + W.chip.n_layers * _M].reshape(-1, _M)
        maps, self.wmap = [], {}
        for L, m in enumerate(self.meta):  # residual dim -> its row in the output projection (-1: not written)
            for kind, n, rows in (("out", m[12], m[13]), ("down", m[20], m[21])):
                if n:
                    wm = np.full(W.R, -1, np.int32)
                    wm[I0[rows:rows + n]] = np.arange(n)
                    self.wmap[(L, kind)] = int(W.I.size) + sum(a.size for a in maps)
                    maps.append(wm)
        self.I = mx.concatenate([W.I, mx.array(np.concatenate(maps))])
        self.walk = False  # the heads by the walking kernel (a pass made again after a token touched more than 8 rows)
        self.most = None

    def _heads(self, L, p, y, s, mk, tn, T, NH, g):
        """A layer's heads over the block, by head and value column (the rows each token touches); with ``walk``, by
        the walking kernel. Returns (outputs, the most rows a token touched: past 8 the pass is to be made again with
        ``walk``)."""
        W, lt = self.w, [("L", L), ("META", self.w.consts["META"]), ("MSTRIDE", _M)]
        if self.walk:
            return self.k_head(inputs=[p, y, W.I, W.Fl, s, mk, tn], template=g + lt, grid=(32, NH, 1), threadgroup=(32, 1, 1),
                               output_shapes=[(T, NH * HEAD)], output_dtypes=[f32])[0], None
        rows, kv, qv, bd, cnt = self.k_prep(inputs=[p, y, W.I, W.Fl], template=g + lt, grid=(32, NH, T), threadgroup=(32, 1, 1),
                                            output_shapes=[(T * NH * 8,), (T * NH * 8,), (T * NH * 8,), (T * NH * 2,), (T * NH,)],
                                            output_dtypes=[mx.int32, f32, f32, f32, mx.int32])
        act, nact, rmask = self.k_act(inputs=[cnt, bd, rows, W.I, tn], template=g + lt, grid=(1024, NH, 1), threadgroup=(1024, 1, 1),
                                      output_shapes=[(NH * T,), (NH,), (NH * 4,)], output_dtypes=[mx.int32, mx.int32, mx.uint32])
        yr = self.k_rec(inputs=[p, W.I, rows, kv, qv, bd, cnt, act, nact, rmask, s, mk, tn],
                        template=g + lt + [("TILE", 16), ("ROWS", 32)], grid=(HEAD, NH, 1), threadgroup=(HEAD, 1, 1),
                        output_shapes=[(T * NH * HEAD,)], output_dtypes=[f32])[0]
        return self.k_hnorm(inputs=[yr, y, W.I, W.Fl, cnt, bd], template=g + lt, grid=(32, NH, T), threadgroup=(32, 1, 1),
                            output_shapes=[(T, NH * HEAD)], output_dtypes=[f32])[0], mx.max(cnt)

    def __call__(self, s: mx.array, c: mx.array, tokens: List[int], mk: Optional[mx.array] = None):
        """One row: ``s``, ``c`` (Walker.pack, B = 1) and the row masks ``mk`` (Walker.masks) updated in place;
        returns (residuals (T, R), new c)."""
        mk = self.w.masks(s) if mk is None else mk
        W, R, T, TG = self.w, self.w.R, len(tokens), self.TGS
        TP = -(-T // 4) * 4
        g = [("R", R), ("H", W.H), ("TGS", TG)]
        x = mx.array(np.ascontiguousarray(np.array(W.chip.embed([tokens]).astype(f32))[0]))
        cnew = []
        self.most = None  # (the most rows a token touched in the pass's heads)
        tn = mx.array(np.array([T], np.int32))
        for L in range(W.chip.n_layers):
            m = self.meta[L]
            if m[0]:
                C, NH, NK, N = int(m[1]), int(m[2]), int(m[3]), int(m[4])
                hn = self.k_norm(inputs=[x, tn], template=g, grid=(TG, T, 1), threadgroup=(TG, 1, 1), output_shapes=[(R * TP,)], output_dtypes=[f32], init_value=0)[0]
                y = self.k_proj(inputs=[hn, W.I, W.Z, tn], template=g + [("NR", N), ("PTR", int(m[5])), ("NNZ", int(m[6])), ("GATE", False)],
                                grid=(-(-TP // 128) * 32, N, 1), threadgroup=(32, 1, 1), output_shapes=[(T, N)], output_dtypes=[f32])[0]
                off = int(m[25])
                cs = c[0, off:off + 3 * C]
                cw = W.Fl[int(m[7]):int(m[7]) + 4 * C]
                p, cso = self.k_conv(inputs=[y, cs, cw, tn], template=g + [("CC", C), ("NR", N)],
                                     grid=(-(-C // TG) * TG, T, 1), threadgroup=(TG, 1, 1), output_shapes=[(T, C), (3 * C,)], output_dtypes=[f32, f32])
                cnew.append((off, cso))
                o, most = self._heads(L, p, y, s, mk, tn, T, NH, g)
                if most is not None:
                    self.most = most if self.most is None else mx.maximum(self.most, most)
                x = self.k_add(inputs=[o, self.I, W.Z, x], template=g + [("WMAP", self.wmap[(L, "out")]), ("PTR", int(m[14])), ("NNZ", int(m[15])), ("K", NH * HEAD)],
                               grid=(-(-R // TG) * TG, T, 1), threadgroup=(TG, 1, 1), output_shapes=[(T, R)], output_dtypes=[f32])[0]
            if m[16]:
                U = int(m[17])
                hn = self.k_norm(inputs=[x, tn], template=g, grid=(TG, T, 1), threadgroup=(TG, 1, 1), output_shapes=[(R * TP,)], output_dtypes=[f32], init_value=0)[0]
                pm = self.k_proj(inputs=[hn, W.I, W.Z, tn], template=g + [("NR", U), ("PTR", int(m[18])), ("NNZ", int(m[19])), ("GATE", True)],
                                 grid=(-(-TP // 128) * 32, U, 1), threadgroup=(32, 1, 1), output_shapes=[(T, U)], output_dtypes=[f32])[0]
                x = self.k_add(inputs=[pm, self.I, W.Z, x], template=g + [("WMAP", self.wmap[(L, "down")]), ("PTR", int(m[22])), ("NNZ", int(m[23])), ("K", U)],
                               grid=(-(-R // TG) * TG, T, 1), threadgroup=(TG, 1, 1), output_shapes=[(T, R)], output_dtypes=[f32])[0]
        return x, mx.concatenate([cso for _, cso in cnew])[None]


# -- decoding with the device paths ------------------------------------------------------------------------------
def _arrays(obj, seen=None) -> list:
    """The MLX arrays an object of ours holds (in its attributes, dicts, lists and tuples)."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, mx.array):
        return [obj]
    if isinstance(obj, dict):
        items = obj.values()
    elif isinstance(obj, (list, tuple)):
        items = obj
    elif hasattr(obj, "__dict__") and type(obj).__module__.startswith("locals"):
        items = vars(obj).values()
    else:
        return []
    return [a for v in items for a in _arrays(v, seen)]


class Device:
    """What every conversation's computer shares: the chip pack's tables and kernels, and the firmware as a packed
    state (made once)."""

    def __init__(self, chip: Chip, walker: Optional[Walker] = None):
        self.chip = chip
        self.w = walker or Walker(chip)
        self.stepper = Stepper(self.w)
        self.block = Block(self.w)
        self._firmware = None
        self._prefixes: "OrderedDict[tuple, tuple]" = OrderedDict()  # a prompt prefix -> the state after it
        # evaluated in the thread that makes them: MLX's streams are a thread's own, and an array left lazy here,
        # evaluated first in the server's computer thread, aborted the server ("no Stream(cpu, 0) in current thread")
        mx.eval(_arrays(self))

    def firmware(self):
        """(state, masks) of the parked firmware, as numpy arrays (the computer's after any prompt)."""
        if self._firmware is None:
            st = self.chip.new_state(1)
            self.chip.inject(st)
            s, _ = self.w.pack(st)
            self._firmware = (np.array(s), np.array(self.w.masks(s)))
        return self._firmware

    def zero(self):
        """(state, convolutions, masks) before any token."""
        s, c = self.w.pack(self.chip.new_state(1))
        return s, c, self.w.masks(s)

    def after(self, tokens: tuple):
        """Copies of the state kept after a prompt prefix (``keep``), or None."""
        got = self._prefixes.get(tokens)
        if got is None:
            return None
        self._prefixes.move_to_end(tokens)
        out = tuple(a + 0 for a in got)
        mx.eval(out)  # (before a pass writes them in place)
        return out

    def keep(self, tokens: tuple, state) -> None:
        """Keep the state after a prompt prefix that later prompts start with (a few: 13 MB each)."""
        self._prefixes[tokens] = tuple(a + 0 for a in state)
        mx.eval(self._prefixes[tokens])
        while len(self._prefixes) > 8:
            self._prefixes.popitem(last=False)


class Computer:
    """One conversation's computer on the device: the packed state (``s``, ``c``, masks ``mk``) and the three ways
    to move it: ``read`` (tokens given: the block pass) and ``step`` (one token, then the run it starts, each token
    the one it predicts, decided on the GPU)."""

    def __init__(self, chip, walker: Optional[Walker] = None):
        self.dev = chip if isinstance(chip, Device) else Device(chip, walker)
        self.chip, self.w, self.stepper, self.block = self.dev.chip, self.dev.w, self.dev.stepper, self.dev.block
        self.s = self.c = self.mk = None

    def start(self, prompt: List[int], cuts=()) -> None:
        """The prompt from zero states (block passes), then the firmware: the snapshot replaces every recurrent
        state, so what the prompt leaves is the convolutions' last inputs. ``cuts``: prompt positions that other
        prompts share (the server's checkpoints: the calc tool's system turn is 369 tokens of a question's 415); the
        state there is kept on the device, and a prompt that starts with a kept prefix reads only the rest."""
        firmware = self.dev.firmware()
        cuts = sorted(n for n in set(cuts) if 0 < n < len(prompt))
        done = next((n for n in reversed(cuts) if self.dev.after(tuple(prompt[:n])) is not None), 0)
        self.s, self.c, self.mk = self.dev.after(tuple(prompt[:done])) if done else self.dev.zero()
        for n in [n for n in cuts if n > done] + [len(prompt)]:
            for i in range(done, n, 2048):
                self.read(prompt[i:min(i + 2048, n)])
            if n < len(prompt):
                self.dev.keep(tuple(prompt[:n]), (self.s, self.c, self.mk))
            done = n
        self.s, self.mk = mx.array(firmware[0]), mx.array(firmware[1])

    def predict(self, x: mx.array) -> np.ndarray:
        return self.chip.predict(x.astype(bf16)[None])[0]

    def read(self, tokens: List[int]) -> np.ndarray:
        """Read ``tokens``; the token predicted after each (-1: Qwen's turn)."""
        before = (self.s + 0, self.c + 0, self.mk + 0)  # (copies on the GPU, for a pass made again: made now, before
        mx.eval(before)  # the pass writes the state in place)
        x, c = self.block(self.s, self.c, tokens, self.mk)
        p = self.predict(x)
        if self.block.most is not None and int(self.block.most.item()) > 8:  # a token touched more than 8 rows
            self.s, self.c, self.mk = before
            self.block.walk = True
            try:
                x, c = self.block(self.s, self.c, tokens, self.mk)
            finally:
                self.block.walk = False
            p = self.predict(x)
        self.c = c
        return np.where((p == -1) & (np.asarray(tokens) >= self.chip.v0), -3, p)

    def peek(self, t: int) -> int:
        """Read one token: the token the computer predicts after it (-1: Qwen's turn)."""
        bufs = self.stepper.start(self.s, self.c, [t], mk=self.mk)
        dep, pred, _ = self.stepper.steps(bufs, 1)
        mx.eval(dep)
        self.c = bufs["c"][bufs["parity"]]
        return int(np.array(pred)[0, 0])

    def step(self, t: int, max_tokens: int = MAX_RUN) -> List[int]:
        """Read one token, then go on while the computer predicts: the tokens of the run (empty: Qwen's turn)."""
        out, self.c = self.stepper.run(self.s, self.c, [t], max_tokens=max_tokens, mk=self.mk)
        return out[0]


class Pool:
    """The runs of several conversations' computers stepped as one batch on the GPU, which runs join and leave
    between two looks of the host (every ``sync`` steps), as the batch engine's rows do: a calc call that closes
    while other runs go on joins them at the next look instead of waiting for them all to end. Each run is the one
    ``Computer.step`` makes: every token the computer's own prediction."""

    def __init__(self, dev: Device, sync: int = 32, max_tokens: int = MAX_RUN):
        self.dev, self.st, self.sync, self.max_tokens = dev, dev.stepper, sync, max_tokens
        self.rows: list = []  # [computer, its run so far, tag]
        self.new: list = []  # (computer, first token, tag): joining at the next look
        self.ended: set = set()  # rows whose runs ended (leaving once the blocks in flight are read)
        self.flight: list = []  # blocks of steps queued on the GPU, oldest first
        self.bufs = None

    def __len__(self) -> int:
        return len(self.rows) + len(self.new)

    def add(self, comp: Computer, first: int, tag) -> None:
        """A run from ``first`` (the token ``comp`` predicted after a call); ``advance`` returns it with ``tag``."""
        self.new.append((comp, first, tag))

    def advance(self) -> list:
        """One look: the new runs join, ``sync`` steps, the runs that ended leave (each computer given back its state
        after its run). Returns (tag, run) of each run that ended (a run ending in a negative code failed). The GPU is
        kept a block of steps ahead (queued while the host reads the last block's tokens); runs join and leave once
        the blocks in flight are read."""
        if not self.flight and self.new:
            self._join([i for i in range(len(self.rows)) if i not in self.ended])
        if not self.rows:
            return []
        while len(self.flight) < 2 and not self.new and not self.ended:
            self.flight.append(self.st.steps(self.bufs, self.sync))
            mx.async_eval(self.flight[-1][0])
        dep, pred, cnt = self.flight.pop(0)
        mx.eval(dep)
        P, N = np.array(pred), np.array(cnt)
        for i, (comp, run, tag) in enumerate(self.rows):
            if i in self.ended:
                continue
            for t in P[i, :N[i]].tolist():
                if t < 0:
                    if t != -1:
                        run.append(t)
                    self.ended.add(i)
                    break
                run.append(t)
            if i not in self.ended and len(run) > self.max_tokens:
                run.append(-5)  # (too long)
                self.ended.add(i)
        if self.flight or not (self.ended or self.new):
            return []
        out = []
        b, p = self.bufs, self.bufs["parity"]
        for i in sorted(self.ended):
            comp, run, tag = self.rows[i]
            comp.s, comp.c, comp.mk = b["s"][i:i + 1] + 0, b["c"][p][i:i + 1] + 0, b["mk"][i:i + 1] + 0
            mx.eval(comp.s, comp.c, comp.mk)
            out.append((tag, run))
        keep = [i for i in range(len(self.rows)) if i not in self.ended]
        self.ended = set()
        if keep or self.new:
            self._join(keep)
        else:
            self.rows, self.bufs = [], None
        return out

    def _join(self, keep: List[int]) -> None:
        """The batch's buffers again: the rows ``keep`` as they are (state, convolutions, next input, current token,
        masks), then the new runs from their computers' states."""
        W = self.dev.w
        s, c, x, mk, cur, rows = [], [], [], [], [], []
        if self.bufs is not None and keep:
            b, p = self.bufs, self.bufs["parity"]
            idx = mx.array(np.array(keep, np.int32))
            s.append(b["s"][idx])
            c.append(b["c"][p][idx])
            x.append(b["x"][0][idx])
            mk.append(b["mk"][idx])
            now = np.array(b["cur"])
            cur += [int(now[i]) for i in keep]
            rows += [self.rows[i] for i in keep]
        for comp, first, tag in self.new:
            s.append(comp.s)
            c.append(comp.c)
            mk.append(comp.mk)
            x.append(mx.array(np.array(W.chip.embed([[first]]).astype(f32)).reshape(1, -1)))
            cur.append(first)
            rows.append([comp, [first], tag])
        self.new = []
        B = len(rows)
        own = lambda a: mx.array(np.ascontiguousarray(a))  # noqa: E731 (a buffer of its own: the kernels write it in place)
        cat = lambda parts: mx.concatenate(parts) + 0  # noqa: E731 (a copy, even of one part)
        S, C, X, MK = cat(s), cat(c), cat(x), cat(mk)
        self.bufs = dict(s=S, c=[C, own(np.zeros(C.shape, np.float32))], x=[X, own(np.zeros(X.shape, np.float32))],
                         scr=own(np.zeros((B, max(self.st.scr, 1)), np.float32)), act=own(_padded([1] * B)),
                         cur=own(_padded(cur)), B=B, dep=own(np.zeros(1, np.float32)), parity=0, mk=MK)
        mx.eval([v for v in self.bufs.values() if isinstance(v, mx.array)] + self.bufs["c"] + self.bufs["x"])
        self.rows = rows


def generate_fast(model, comp: Computer, prompt: List[int], max_tokens: int = 1000, on_token=None,
                  lazy: bool = True) -> dict:
    """Greedy decoding as ``qwenputer.generate``, with the computer on the device. The computer reads each of Qwen's
    tokens (``lazy``: not one by one but in one block pass when Qwen closes a tool call, the only place the computer
    takes over: the pass checks it predicts nothing before); a run it starts is stepped; Qwen reads the run's echoed
    text in one pass."""
    import time

    from mlx_lm.models.cache import make_prompt_cache

    chip = comp.chip
    v0 = chip.v0
    t0 = time.perf_counter()
    cache = make_prompt_cache(model)
    for i in range(0, len(prompt), 2048):
        logits = model(mx.array([prompt[i:i + 2048]]), cache=cache)[:, -1]
        mx.eval(logits)
    comp.start(prompt)
    t_prompt = time.perf_counter() - t0
    nxt = int(mx.argmax(logits[0, :v0]))
    out, n_text, n_chip, t_chip = [], 0, 0, 0.0
    pending: List[int] = []  # Qwen's tokens the computer has not read yet (lazy)
    while n_text < max_tokens:
        out.append(nxt)
        if on_token is not None:
            on_token(nxt)
        if nxt in chip.info["eos"]:
            break
        n_text += 1
        t1 = time.perf_counter()
        logits = model(mx.array([[nxt]]), cache=cache)[:, -1]  # (Qwen reads it while the computer does)
        if lazy:
            pending.append(nxt)
            if nxt != chip.info["tool_call"][1]:  # (not </tool_call>: Qwen goes on)
                t_chip += time.perf_counter() - t1
                nxt = int(mx.argmax(logits[0, :v0]))
                continue
            pred = comp.read(pending)
            if (pred[:-1] != -1).any():
                i = int(np.nonzero(pred[:-1] != -1)[0][0])
                raise RuntimeError(f"the computer predicts {int(pred[i])} after Qwen's {pending[i]}, before a </tool_call>")
            first, pending = int(pred[-1]), []
        else:
            first = comp.peek(nxt)  # the computer reads it: a token of its own follows, or Qwen's turn
        run = []
        if first < -1:
            raise RuntimeError(f"the computer failed ({first}) after {nxt}")
        if first >= 0:
            run = [first] + comp.step(first)
        t_chip += time.perf_counter() - t1
        if not run:
            nxt = int(mx.argmax(logits[0, :v0]))
            continue
        if run[-1] < 0:
            raise RuntimeError(f"the computer failed ({run[-1]}) after {len(run) - 1} tokens of a run")
        n_chip += sum(1 for t in run if t >= v0)
        for t in run:
            out.append(t)
            if on_token is not None:
                on_token(t)
        echo = [t for t in run if t < v0]
        if echo:  # Qwen reads the computer's text in one pass
            logits = model(mx.array([echo]), cache=cache)[:, -1]
        nxt = int(mx.argmax(logits[0, :v0]))
    return {"tokens": out, "text_tokens": n_text, "chip_tokens": n_chip, "seconds": time.perf_counter() - t0, "prompt_seconds": t_prompt, "chip_seconds": t_chip}


def _called(text: str) -> Optional[str]:
    """The function a Qwen3.5 tool call names."""
    import re

    m = re.search(r"<function=([^>\s]+)>", text)
    return m.group(1) if m else None


def _note_op(op: str, text: str) -> Optional[tuple]:
    """(op, key, value) of a remember / forget call's text (a tab or a newline in them a space; a key in quotes
    without them, as notes.c reads it); None for recall."""
    import re

    if op == "recall":
        return None
    params = dict(re.findall(r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", text, re.S))
    clean = lambda v: " ".join(v.split())  # noqa: E731
    key = clean(params.get("key") or params.get("name") or "")
    if len(key) >= 2 and key[0] in "\"'" and key[-1] == key[0]:
        key = key[1:-1]
    return op, key, clean(params.get("value") or "")


def _notes_state(ops) -> str:
    """The notes a conversation's notes calls (``_note_op``'s, in order) leave, as the computer replays them: a line
    each, remember, key, value (tab-separated), keys as notes.c finds them (case ignored, 60 characters at most); the
    newest that fit its 2,048 bytes of notes (3 bytes, the key and the value each)."""
    from collections import OrderedDict

    state: "OrderedDict[str, tuple]" = OrderedDict()
    for op, key, value in ops:
        if not key or len(key) > 60:
            continue
        state.pop(key.lower(), None)
        if op == "remember" and value:
            state[key.lower()] = (key, value)
    keep, size = [], 0
    for key, value in reversed(state.values()):
        size += 3 + len(key.encode("utf-8")) + len(value.encode("utf-8"))
        if size > 2048:
            break
        keep.append(f"remember\t{key}\t{value}")
    return "\n".join(reversed(keep))


# The bridge's text buffer (chipformer.rv.spec): a row of each of a call's first ROWS tokens, its first ROW bytes
ROW, ROWS = 16, 80
# a long toolbox call's parts and bytes at most: the toolbox's text holds 15 KB, and writing 7,000 bytes backwards
# (the slowest tool, about 30 run tokens a byte) takes about 210,000 run tokens, within MAX_RUN
PARTS, BYTES = 13, 7000
# (copy) a number of the answer's longer than this many tokens is shown as written, not held for copy_check (which
# reads 64 digits at most): a model that writes ",000" on and on, held, was hidden from the server's loop guard
NUM_HOLD = 100
CUTS = 2  # runs cut at MAX_RUN in a reply before a toolbox call gets an error without a run (CalcController)
# (a run's errors that say it used its whole budget of work, the last thing it printed: counted as cut runs; a program
# over run_python's steps took about a minute to fail, and the model wrote another, and another)
_SPENT = ("TimeoutError: ", "too many steps")
TOO_LONG = "too_long"  # (the call the toolbox reads in place of a longer one: an error it prints, qwenputer_tools.s)
# (check_call's errors that are the computer's, not the call's: the call goes as written)
_NOT_THE_CALLS = ("too long for the computer", "out of memory", "too many steps", "no such tool")
_VOCABS: dict = {}


def _vocab(tok):
    """(the tokenizers.Tokenizer, the added tokens' ids, the byte-level alphabet) of a tokenizer, made once."""
    if id(tok) not in _VOCABS:
        from ..vocab import _bytes_to_unicode

        hf = getattr(tok, "_tokenizer", tok)
        _VOCABS[id(tok)] = (getattr(hf, "backend_tokenizer", hf), frozenset(getattr(hf, "added_tokens_decoder", None) or ()),
                            _bytes_to_unicode())
    return _VOCABS[id(tok)]


def _rows(tok, body: List[int]) -> List[int]:
    """A toolbox call's text as the computer is to read it: a token of more than ROW bytes (a run of spaces or dashes,
    a Chinese word: its row held a cut piece of it, and a text's length or the text backwards came out wrong) as the
    vocabulary's tokens of at most ROW bytes it is made of, the longest first. Qwen's own text stays as written."""
    from ..vocab import _BYTE_DECODER

    bt, added, enc = _vocab(tok)
    out = []
    for t in body:
        piece = bt.id_to_token(t)
        b = bytes(_BYTE_DECODER[c] for c in piece) if piece and t not in added and all(c in _BYTE_DECODER for c in piece) else b""
        if len(b) <= ROW:
            out.append(t)
            continue
        j = 0
        while j < len(b):
            for k in range(min(ROW, len(b) - j), 0, -1):  # (each byte is a token: k = 1 always is one)
                i = bt.token_to_id("".join(enc[x] for x in b[j:j + k]))
                if i is not None and i not in added:
                    break
            out.append(i)
            j += k
    return out


def _parts(tok, body: List[int]) -> List[List[int]]:
    """A toolbox call's text (the tokens between <tool_call> and </tool_call>) as the calls the computer reads, each
    of at most ROWS tokens: the call (``_rows``); or, longer, its parts, each a first row (the byte 2 for the first
    part, 3 for the last, 1 for one between) and ROWS - 1 of its tokens: the toolbox adds a part's tokens to the
    call's text and parks without a word, and runs the call at its last part; or, longer than PARTS parts or BYTES
    bytes, a call of too_long."""
    body = _rows(tok, body)
    if len(body) <= ROWS:
        return [body]
    if not _fits(tok, body):
        return [tok.encode(f"\n<function={TOO_LONG}>\n</function>\n", add_special_tokens=False)]
    k = ROWS - 1
    n = -(-len(body) // k)
    bt, _, enc = _vocab(tok)
    mark = [bt.token_to_id(enc[2])] + [bt.token_to_id(enc[1])] * (n - 2) + [bt.token_to_id(enc[3])]
    return [[mark[j]] + body[j * k:(j + 1) * k] for j in range(n)]


def _fits(tok, body: List[int]) -> bool:
    """Whether a toolbox call's text, as the computer reads it (``_rows``), is one it can read: in a row each of
    ROWS tokens, or in PARTS parts at most and BYTES bytes."""
    return len(body) <= ROWS or (-(-len(body) // (ROWS - 1)) <= PARTS and len(tok.decode(body).encode("utf-8")) <= BYTES)


class Presenter:
    """The reply as a client reads it, in Qwen3.5's own form: a calc exchange the model ran before writing any
    content is reasoning (in the think block: the model's own when it thinks, one opened for it when it does not;
    its closing tag waits until text other than an exchange follows); one after content has begun is left out (the
    model still read it). A call to another tool and everything else pass through. ``feed`` takes turn's events and
    returns the token ids to show.

    States: think (the model's reasoning), gap (no content yet: a held </think> and blank lines), exchange (a calc
    call shown as reasoning, then its run), content, skip (a calc call left out, then its run), client (with
    ``checked``: a call of the client's held until the computer has checked it, ``checked_call``); a call is held in
    ``call`` until it names its function."""

    def __init__(self, tokenizer, thinking: bool, v0: int, internal=("calc",), checked: bool = False):
        ids = {s: tokenizer.convert_tokens_to_ids(s) for s in ("<think>", "</think>", "<tool_call>", "</tool_call>")}
        self.internal = frozenset(internal)  # the tools the computer runs (calc, and the toolbox's)
        self.open, self.close, self.tco, self.tcc = ids["<think>"], ids["</think>"], ids["<tool_call>"], ids["</tool_call>"]
        self.tok, self.v0 = tokenizer, v0
        self.state = "think" if thinking else "gap"
        self.opened = thinking  # a think block is open in what was shown
        self.held: List[int] = []  # shown when the reply turns out to go on with content
        self.call: Optional[List[int]] = None  # a call whose function is not known yet
        self.checked = checked
        self.pending: Optional[List[int]] = None  # (client) the client's call, until checked
        self.before = "gap"  # (client) the state the call came in

    def _blank(self, t: int) -> bool:
        return not self.tok.decode([t]).strip()

    def save(self) -> tuple:
        return (self.state, self.opened, list(self.held), None if self.call is None else list(self.call),
                None if self.pending is None else list(self.pending), self.before)

    def load(self, saved: tuple) -> None:
        self.state, self.opened, held, call, pending, self.before = saved
        self.held, self.call = list(held), None if call is None else list(call)
        self.pending = None if pending is None else list(pending)

    def feed(self, kind: str, toks) -> List[int]:
        if kind == "run":
            if self.state == "skip":
                self.state = "content"
                return []
            if self.state == "think":  # (a call inside the model's reasoning: all of it reasoning)
                return [t for t in toks if t < self.v0]
            keep = []  # the echo up to its tool response's end (its closing empty think block is left out)
            for t in (t for t in toks if t < self.v0):
                keep.append(t)
                if "</tool_response>" in self.tok.decode(keep):
                    break
            self.state = "gap"
            return keep
        t = toks
        if self.call is not None:  # a call: held until it names its function
            self.call.append(t)
            name = _called(self.tok.decode(self.call))
            if name is None and t != self.tcc:
                return []
            call, self.call = self.call, None
            if name in self.internal and self.state == "gap":  # an exchange before content: reasoning
                head = [] if self.opened else [self.open]
                self.opened = True
                self.held = [x for x in self.held if x == self.close]  # (the model's own </think> waits)
                self.state = "exchange"
                return head + call
            if name in self.internal:  # after content: left out
                self.state = "skip"
                return []
            if self.checked:  # (the client's: held until the computer has checked it)
                self.pending, self.before, self.state = call, self.state, "client"
                return []
            out = self._flush() + call
            self.state = "content"
            return out
        if self.state == "client":
            self.pending.append(t)
            return []
        if self.state == "exchange":
            return [t]
        if self.state == "skip":
            return []
        if self.state == "think":
            if t == self.close:
                self.state = "gap"
                self.held = [t]
                return []
            return [t]
        if t == self.tco:
            self.call = [t]
            return []
        if self.state == "gap":
            if self._blank(t):
                self.held.append(t)
                return []
            out = self._flush() + [t]
            self.state = "content"
            return out
        return [t]

    def checked_call(self, ok: bool) -> List[int]:
        """(client) The computer has checked the client's call: shown as written if it is right; if not, the call and
        the computer's response (fed as a run next) are an exchange of the computer's, as a calc call is (reasoning
        before content, left out after), and the model writes the call again."""
        call, self.pending = self.pending or [], None
        if ok:
            self.state = "content"
            return self._flush() + call
        if self.before == "gap":
            head = [] if self.opened else [self.open]
            self.opened = True
            self.held = [x for x in self.held if x == self.close]
            self.state = "exchange"
            return head + call
        self.state = "skip"
        return []

    def _flush(self) -> List[int]:
        """Before content: the think block shown closes (the model's own </think>, or one for the block opened)."""
        held, self.held = self.held, []
        if not self.opened:
            return held
        self.opened = False
        if self.close not in held:
            return [self.close] + [x for x in held if x != self.close]
        return held

    def finish(self) -> List[int]:
        """At the reply's end: a think block shown still open is closed."""
        out = []
        if self.call is not None:
            out, self.call = self.call, None
        if self.pending is not None:  # (a client's call not checked: shown as written)
            out, self.pending = out + self.pending, None
        return self._flush() + out if self.opened else self.held + out

# -- the computer in Locals's batch engine -----------------------------------------------------------------------
def _controller_base():
    from ..engine import Controller

    return Controller


class CalcController(_controller_base()):
    """A request's controller on Qwenputer, in the batch engine: the computer makes the tool response of each calc
    call. It reads nothing until a calc call closes (it takes over nowhere else); then it reads the prompt (the first
    time) and Qwen's tokens since, in block passes, makes its run token by token, each the one it predicts (stepped
    on the GPU), and the echoed text is the forced span after ``</tool_call>``. A reply without a calc call costs the
    computer nothing. ``drain`` gives the tokens to show the client (``Presenter``).

    With ``split`` the batch never waits for the computer: the row stops at the call's ``</tool_call>`` (``held``;
    the engine keeps its cache), the service's computer thread makes the run (``read_call``, then the run on a CPU
    core or stepped in a ``Pool`` with the other replies' runs; ``ran``), and the reply goes on as a request that
    continues the kept cache, Qwen reading the echo in one pass. With ``park`` the row stays in the batch instead,
    reading nothing (``parked``) until ``release`` gives the engine the echo, its forced span.

    Nothing is drafted: the batch engine's prompt-lookup drafts are off for these rows (``drafts``)."""

    drafts = False

    def __init__(self, dev: Device, prompt: List[int], tokenizer, eos, endless: bool = False, split: bool = False,
                 cuts=(), park: bool = False, internal=("calc",), computer: Optional[dict] = None):
        import threading

        self.eos_ids = () if endless else list(eos)
        self.dev, self.prompt, self.tok = dev, list(prompt), tokenizer
        self.v0 = dev.chip.v0
        self.tco, self.tcc = dev.chip.info["tool_call"]
        # the tools the computer runs for this request (calc, and the toolbox's its prompt offers): another tool's
        # call is the client's
        self.internal = frozenset(internal)
        # the server's work for the computer (``computer``): "checks", the client's tools' schemas by name (and
        # "tools", their names) for check_call, which checks each call of the client's before the client gets it (one
        # that is wrong goes back to the model with what is wrong, twice at most); "notes", the conversation's notes
        # calls before this request (remember / forget, a line each), given to its computer with its first notes
        # call; "copy", numbers the answer copies from the computer's results checked by copy_check (``on_copy``)
        computer = computer or {}
        self.checks: Optional[dict] = computer.get("checks") if (split or park) else None
        self.tools_line: str = computer.get("tools", "")
        self.notes: str = computer.get("notes") or ""
        self.copy = bool(computer.get("copy")) and (split or park)
        self.checking: Optional[str] = None  # the client's tool whose call the computer checks
        self.check_tries = 0
        self.notes_given = False
        self.note: Optional[tuple] = None  # (op, key, value) of the notes call being run
        self.note_ops: List[tuple] = []  # this reply's notes calls that ran (the server keeps them for the next)
        self.results: List[str] = []  # the values the computer gave in this reply
        self.on_copy = None  # (the service's) a copy check for the computer's thread: (controller, results, number)
        self.num: List[int] = []  # (copy) a number being written in the answer
        self.prompt_text: Optional[str] = None  # (copy) the prompt as text, once a number is to be checked
        self.run_on = False  # (copy) in a number too long to check (``NUM_HOLD``)
        self.num_held: List[int] = []
        self.num_tail: List[int] = []
        self.wait: Optional[List[tuple]] = None  # (copy) what came after a number being checked: (token, answer)
        self.on_end = None  # (copy) the reply's end, once its last check is done
        self.pres = Presenter(tokenizer, tokenizer.decode(prompt[-3:]).endswith("<think>\n"), self.v0, internal,
                              checked=self.checks is not None)
        self.shown: List[int] = []
        self.sent = 0
        self.comp: Optional[Computer] = None  # (made at the first calc call)
        self.unread: List[int] = []  # Qwen's tokens the computer has not read
        self.seen: List[int] = []  # the reply's text so far (Qwen's tokens and the echoes it read): a computer made
        # again (after a run cut at MAX_RUN) reads it
        self.call: Optional[List[int]] = None  # Qwen's tokens since <tool_call>
        self.runs = self.run_tokens = 0
        # runs cut at MAX_RUN in this reply, or that used up their work (``_SPENT``): after CUTS of them, a toolbox
        # call gets an error without a run (``refused``): the model had asked again and again, at minutes a run
        self.cut_runs, self.refused = 0, False
        self.split, self.held = split, False
        self.tool: Optional[str] = None  # the function of the latest call
        self.parts: List[List[int]] = []  # a long toolbox call's parts the computer has not read (``_parts``)
        # park: the row stays in the batch and reads nothing (``parked``, hybrid_batch's parked rows) until the run
        # is made; ``on_park`` (the service's) starts it, and ``ready`` takes the echo ``release`` gives the engine
        self.park, self.parked, self.ready, self.on_park = park, False, None, None
        self.cuts = list(cuts)  # prompt positions other prompts share (Computer.start)
        self.lock = threading.Lock()

    def accept(self, token: int) -> None:
        answer = self.call is None and token != self.tco
        with self.lock:
            if token == self.tco and self.num and self.wait is None:  # (a number right before a call ends there)
                self._end_number([])
            self._show(self.pres.feed("qwen", token), answer and self.pres.state == "content")
        self.unread.append(token)
        self.seen.append(token)
        if token == self.tco:
            self.call = []
        elif self.call is not None:
            self.call.append(token)
            if (self.split or self.park) and token == self.tcc:
                body, self.call = self.call[:-1], None
                name = _called(self.tok.decode(body))
                self.tool = name
                # a call of the client's the presenter holds (one with a name, outside the reasoning), when the
                # computer can read it with its schema (a longer call goes as written)
                check = (name is not None and name not in self.internal and self.checks is not None
                         and self.check_tries < 2 and self.pres.state == "client"
                         and _fits(self.tok, _rows(self.tok, self._check_head(name) + body)))
                self.checking = name if check else None
                if self.pres.state == "client" and not check:  # (not checked: it goes as written)
                    with self.lock:
                        self._show(self.pres.checked_call(True), False)
                if (name in self.internal or check) and self.park:
                    self.parked = True
                    if self.on_park is not None:
                        self.on_park()
                else:
                    self.held = name in self.internal or check

    def may_force(self) -> bool:
        return self.call is not None  # (inside a call: its </tool_call> may start a run)

    def finished(self) -> bool:
        return self.held

    def forced(self, token: int):
        if token != self.tcc or self.call is None or self.split or self.park:
            return [], False
        name, self.call = _called(self.tok.decode(self.call)), None
        self.tool = name
        if name not in self.internal:
            return [], False
        while True:  # (a long toolbox call: a run a part)
            first = self.read_call()
            echo = self.ran([first] + self.comp.step(first) if first >= 0 else [])
            if not self.parts:
                return echo, False

    def release(self) -> Optional[List[int]]:
        """(park) The echo the row goes on with, once the run is made; None until then."""
        span = self.ready
        if span is None:
            return None
        self.ready, self.parked = None, False
        return span

    def read_call(self) -> int:
        """The computer (made and given the prompt the first time) reads Qwen's tokens since its last run; the token
        it predicts after the call (the run's first; < 0: none)."""
        if self.cut_runs >= CUTS and self.checking is None and self.tool not in (None, "calc") and not self.parts:
            # (no run: the computer made again at the next call, from the reply so far, this call in it)
            self.comp, self.unread, self.notes_given, self.note, self.refused = None, [], False, None, True
            return -1
        if self.comp is None:  # (a computer made again after a cut run: the reply so far as part of its prompt)
            self.comp = Computer(self.dev)
            done = self.seen[:len(self.seen) - len(self.unread)]
            self.comp.start(self.prompt + done, self.cuts)
        unread = self.unread
        if self.parts:  # the next part of a long toolbox call
            unread = [self.tco] + self.parts.pop(0) + [self.tcc]
        elif self.tool not in (None, "calc") and unread and unread[-1] == self.tcc and self.tco in unread:
            # a toolbox call as the computer reads it: a row a whole token, and a call its rows cannot hold (it read
            # the first 79 tokens and then only the last) in parts, each a run (``_parts``)
            i = len(unread) - 1 - unread[::-1].index(self.tco)
            body = unread[i + 1:-1]
            if self.checking is not None:  # (the client's call, after the schema it is checked against)
                body = self._check_head(self.checking) + body
            elif self.tool in ("remember", "recall", "forget"):
                text = self.tok.decode(body)
                self.note = _note_op(self.tool, text)
                j = text.rfind("</function>")
                notes = self.notes.split("\n") if self.notes and not self.notes_given and j >= 0 else []
                while notes:  # (this request's computer has none of them yet: the newest that fit in the call)
                    with_notes = self.tok.encode(text[:j] + "<parameter=_replay>\n" + "\n".join(notes)
                                                 + "\n</parameter>\n" + text[j:], add_special_tokens=False)
                    if _fits(self.tok, _rows(self.tok, with_notes)):
                        body = with_notes
                        break
                    notes = notes[len(notes) // 4 + 1:]
                self.notes_given = True
            parts = _parts(self.tok, body)
            unread, self.parts = unread[:i + 1] + parts[0] + unread[-1:], parts[1:]
        pred = self.comp.read(unread)
        self.unread = []
        if (pred[:-1] != -1).any():
            i = int(np.nonzero(pred[:-1] != -1)[0][0])
            raise RuntimeError(f"the computer predicts {int(pred[i])} inside Qwen's text")
        return int(pred[-1])

    def _check_head(self, name: str) -> List[int]:
        """The text before a client's call that the computer checks: check_call with the schema of the tool it names
        (the client's tools' names, for a tool of none of them)."""
        schema = self.checks[name] if name in self.checks else self.tools_line
        return self.tok.encode(f"\n<function=check_call>\n{schema}\n</function>", add_special_tokens=False)

    def unchecked(self) -> List[int]:
        """(the computer failed checking the client's call: its run failed, or it could not read the call) The call
        goes as written; the computer is made again at the next call, from the reply so far, the call in it."""
        self.comp, self.parts, self.unread, self.held, self.notes_given, self.note = None, [], [], False, False, None
        self.checking = None
        with self.lock:
            self._show(self.pres.checked_call(True), False)
        return []

    def ran(self, run: List[int]) -> List[int]:
        """The computer's run shown; its echoed text (Qwen reads it next). A held row goes on after it. A run of a
        part of a long call (``parts`` still to read) is not shown, nor read by Qwen: [] (the next part is read)."""
        if self.refused:  # (runs cut CUTS times: no run, an error)
            self.refused = False
            return self._error("the computer's runs went past its limit twice in this reply: work it out without "
                               "the computer")
        if run and run[-1] == -5:  # cut at MAX_RUN: an error for the model, and a computer made again for the rest
            return self._cut(run)
        if run and run[-1] < 0:
            if self.checking is not None:
                return self.unchecked()
            raise RuntimeError(f"the computer failed ({run[-1]}) after {len(run) - 1} tokens of a run")
        if self.parts and not run:  # (no run for a part: the call goes no further, as a call with no run)
            self.parts = []
        if self.parts:
            self.runs += 1
            self.run_tokens += len(run)
            return []
        self.held = False
        self.runs += 1
        self.run_tokens += len(run)
        echo = [t for t in run if t < self.v0]
        value = self._value(echo)
        if self.checking is not None:  # (the computer's check of the client's call)
            self.checking = None
            with self.lock:
                # the call goes to the client, the model going on after it: one that is right, or one the computer
                # could not check (an error of the computer's own, not of the call's)
                if not value.startswith("error (") or any(w in value[:80] for w in _NOT_THE_CALLS):
                    self._show(self.pres.checked_call(True), False)
                    return []
                self.check_tries += 1
                self._show(self.pres.checked_call(False) + self.pres.feed("run", run), False)
            echo = self._as_text(echo)
            self.seen += echo
            return echo
        with self.lock:
            self._show(self.pres.feed("run", run), False)
        at = value.rfind("error (")  # (after what the program printed)
        if at >= 0 and self.tool != "calc" and any(w in value[at:at + 80] for w in _SPENT):
            self.cut_runs += 1
        if value and not value.startswith("error ("):
            self.results.append(value)
            if self.note is not None and value.startswith(("remembered", "updated", "forgot")):
                self.note_ops.append(self.note)
        self.note = None
        if self.tool != "calc":
            echo = self._as_text(echo)
        self.seen += echo
        return echo

    def _cut(self, run: List[int]) -> List[int]:
        """A run cut at MAX_RUN, its computer mid-program: the computer is made again from the reply so far (its notes
        from the conversation's earlier calls, read again at its next notes call); a check of the client's call lets the
        call go; any other call gets an error as its response, read by the model."""
        self.runs += 1
        self.run_tokens += len(run)
        self.cut_runs += 1
        self.comp, self.parts, self.notes_given, self.note = None, [], False, None
        if self.checking is not None:
            self.checking, self.held = None, False
            with self.lock:
                self._show(self.pres.checked_call(True), False)
            return []
        return self._error("the computer's run went past its limit: give it a smaller job")

    def _error(self, why: str) -> List[int]:
        """An error of the computer's as the call's response (no run): shown as a run's echo is, and read by Qwen."""
        self.held = False
        echo = self.tok.encode(f"\n<tool_response>\nerror ({why})\n</tool_response>\n<think>\n\n</think>\n\n",
                               add_special_tokens=False)
        with self.lock:
            self._show(self.pres.feed("run", echo), False)
        self.seen += echo
        return echo

    def _value(self, echo: List[int]) -> str:
        """The text of a run's tool response."""
        text = self.tok.decode(echo)
        a, b = text.find("<tool_response>\n"), text.find("\n</tool_response>")
        return text[a + 16:b] if 0 <= a < b else ""

    # -- (copy) numbers the answer copies from the computer's results: held until copy_check has read them
    def _show(self, toks: List[int], answer: bool) -> None:
        """Tokens to show (the lock held): a number of the answer's (``answer``), and what follows it, wait while the
        computer checks it."""
        for t in toks:
            if self.wait is not None:
                self.wait.append((t, answer))
            elif answer and self.copy and self.results:
                self._number(t)
            else:
                self.shown.append(t)

    def _number(self, t: int) -> None:
        piece = self.tok.decode([t])
        digit = piece.isdigit() and piece.isascii()
        if self.run_on:  # (the rest of a number too long to check: as written)
            self.run_on = digit or piece in (",", ".")
            self.shown.append(t)
        elif digit or (self.num and piece in (",", ".")):
            self.num.append(t)
            if len(self.num) > NUM_HOLD:  # longer than copy_check reads: shown as written (a loop guard sees it)
                self.shown += self.num
                self.num, self.run_on = [], True
        elif self.num:
            self._end_number([(t, True)])
        else:
            self.shown.append(t)

    def _end_number(self, after: List[tuple]) -> None:
        num, self.num = self.num, []
        tail = []
        while num and self.tok.decode(num[-1:]) in (",", "."):  # (a full stop or a comma after it)
            tail.insert(0, num.pop())
        written = self.tok.decode(num)
        if (self.on_copy is not None and sum(c.isdigit() for c in written) >= 6
                and any(sum(c.isdigit() for c in r) >= 6 for r in self.results)):
            self.wait, self.num_held, self.num_tail = list(after), num, tail
            self.on_copy(self, "\n".join(self.results), written)
            return
        self.shown += num + tail + [t for t, _ in after]  # (what ended the number is not one)

    def in_prompt(self, written: str) -> bool:
        """(the computer's thread) Whether the question wrote the number: not one copied from a result (the prompt
        read as text once, off the batch's thread)."""
        if self.prompt_text is None:
            self.prompt_text = self.tok.decode(self.prompt)
        return written in self.prompt_text

    def copied(self, fixed: Optional[str]) -> None:
        """(the computer's thread) copy_check's answer: the number as written (None) or as the result it miscopied;
        then what waited after it (which may hold the next number to check), and the reply's end if it was waiting."""
        end = None
        with self.lock:
            self.shown += (self.tok.encode(fixed, add_special_tokens=False) if fixed else self.num_held) + self.num_tail
            wait, self.wait = self.wait or [], None
            for t, answer in wait:
                self._show([t], answer)
            if self.wait is None and self.on_end is not None:
                if self.num:  # (the reply ended in a number: it may be the next to check)
                    self._end_number([])
                if self.wait is None:
                    end, self.on_end = self.on_end, None
        if end is not None:
            end()

    def finish_after(self, end) -> bool:
        """At the reply's end: False if nothing waits for a check (the caller ends it now); else True, and ``end`` is
        called once the check is done."""
        with self.lock:
            if self.num and self.wait is None:
                self._end_number([])
            if self.wait is None:
                return False
            self.on_end = end
            return True

    def _as_text(self, echo: List[int]) -> List[int]:
        """A toolbox tool's echo with its value (the line after <tool_response>) in the tokenizer's own tokens, as Qwen
        reads a client's tool result: the computer prints a byte a token, and Qwen copied text it read so (a word
        written backwards) a letter wrong. calc's echo stays as printed (digits are a token each either way)."""
        nl = self.tok.encode("\n", add_special_tokens=False)
        open_, close = (self.tok.convert_tokens_to_ids(t) for t in ("<tool_response>", "</tool_response>"))
        try:
            a = echo.index(open_) + 1 + len(nl)
            b = echo.index(close, a) - len(nl)
        except ValueError:
            return echo
        if b <= a:
            return echo
        text = self.tok.encode(self.tok.decode(echo[a:b]), add_special_tokens=False)
        if not _vocab(self.tok)[1].isdisjoint(text):  # (a value that spells <|im_end|>: a byte a token, as printed)
            return echo
        return echo[:a] + text + echo[b:]

    def drain(self, cb=None) -> List[int]:
        """The tokens to show that were not shown yet, given to ``cb`` in order (the batch's thread and the
        computer's both drain)."""
        with self.lock:
            out = self.shown[self.sent:]
            self.sent = len(self.shown)
            if cb is not None and out:
                cb(out)
        return out

    def finish(self, cb=None) -> List[int]:
        self.shown += self.pres.finish()
        return self.drain(cb)
