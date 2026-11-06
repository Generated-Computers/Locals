"""Qwenputer's computer on a CPU core: a run made by the stepper's operations (qwenputer_walk: _RESID, _ATTN,
_HEAD_STEP, _MLP, _TOKEN) in their order, in C, on a ``Device``'s tables and a ``Computer``'s packed state. The run
and the state after it are the GPU's, bit for bit (tests/test_qwenputer.py).

Next to Qwen's batch on the GPU, the computer's kernels waited on memory Qwen's weights keep busy (a run stepped at
740 tokens/s there, 1,700 alone on an M2 Pro); a performance core makes a run at 3,300 tokens/s whatever the GPU does,
and the cores make several at once. What makes a core this fast is the computer's sparsity, kept exact:

* a sparse product walks only the columns whose input is nonzero (7 to 23% of the normed residual, 16 of a layer's
  4,096 head outputs), in ascending order: every row stores its entries in ascending column order (checked), so a
  row's terms are added in the stepper's order, and a zero input adds only zeros, which change no sum;
* a head that no query or key entry touches and whose state does not decay changes nothing and outputs 0 (most of a
  layer's heads at a token): only its value channels' convolution state moves on;
* the queries and keys of a key head are made once for the value heads that share it;
* the attention layers' convolution channels are stepped four at a time (NEON), every channel's sums in the stepper's
  order; one whose inputs are all +0 gives +0 and keeps its +0 state, as the stepper's. SiLU is a table over the
  65,536 bf16 values its input takes, made by the same expression;
* a norm's sum of squares, added in order, is the ballast's alone when every other square is under half a unit in the
  last place of that sum (checked; else summed);
* the 62 ballast dims, which never change, come first in the column order: a product's sums over them are made once
  a run (and made again if their inputs ever differ, checked bit for bit) and each step goes on from them; a unit no
  other column touches keeps the output it had, and a key head whose queries and keys are the step before's, bit for
  bit, keeps their norms and the rows they touch.

A run works in place on the computer's own device buffers (``shared``: on Apple silicon the GPU's memory is the
CPU's, so the state is neither copied out for a run nor back after it).

A run alone can share its attention layers' key and value heads and its MLP layers' units out to a few cores
(``threads``: 4,100 tokens/s with 3 or 4 on an M2 Pro); the service does that when no other run is going.

The C is built once with the Mac's clang (Xcode's command line tools) into the chip pack's folder; ``CPU.make``
returns None where it cannot be (the computer then runs on the GPU)."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

SOURCE = r"""// Qwenputer's computer on a CPU core: the stepper's operations (locals/models/qwenputer_walk.py: _RESID, _ATTN,
// _HEAD_STEP, _MLP, _TOKEN) in their order, one run at a time, on the walker's tables and packed state.
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <sched.h>
#include <string.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// a sparse matrix by columns: the columns' entries (row, value), the columns in ascending order. Every row's entries
// are stored in ascending column order (checked), so a walk over the columns adds a row's terms in the order the
// stepper adds them; a column whose input is 0 adds only zeros, which change no sum (+0 + -0 is +0, and a sum starts
// at +0), and is left out: the sums are the stepper's, bit for bit.
typedef struct {
    const int32_t* cp;
    const int32_t* ri;
    const float* v;
    int ncols, nrows;
} Mat;

typedef struct {
    const int32_t* I;
    const uint32_t* Z;
    const float* Fl;
    const uint8_t* bits;
    const int32_t* prog;  // per layer kernel: kind (0 attn, 1 mlp), L, out offset, KWRITE, PROWS, PPTR, PNNZ, PN, PSRC
    const int32_t* tok;   // the token kernel's PROWS, PPTR, PNNZ, PN, PSRC
    const Mat* main;      // per layer kernel: its projection (attention) or its units' rows (MLP), by columns
    const Mat* res;       // per layer kernel, then the token kernel: the rows it adds to the residual, by columns
    int nk;
    int R, H, V0, CTRL, NFAM, NPAY, NFAMS, NCTRL, NPRINT, NSPEC, EXPL_A, EXPL_N, META, MSTRIDE, SCR, CSTRIDE;
    int SPEC_IDS, BITOF, FBASE, FSIZE, CODE, PRINT, SPEC_ROWS, PLAIN, EXPL_ROWS, FAM_BASE, FAM_DELTA;
} Tab;

static inline float r16(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) return x;  // (inf, nan)
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    memcpy(&x, &u, 4);
    return x;
}
static inline float pexp(float x) { return (float)exp((double)x); }
// exp(-x) where most x are 0 (a sparse computer): exp(0) is 1 exactly
static inline float pexpn(float x) { return x == 0.0f ? 1.0f : pexp(-x); }
static inline float prsqrt(float x) { return (float)(1.0 / sqrt((double)x)); }
static inline float plog1p(float x) { return (float)log1p((double)x); }

// SiLU of every bf16 value (each input here is a bf16 value: rounded just before), as the stepper computes it:
// x / (1 + exp(-x)), rounded to bf16 (convolutions, units) and not (the heads' output gate); made once
static float silu_r[65536], silu_f[65536];
static int silu_made = 0;
static void make_silu(void) {
    for (uint32_t b = 0; b < 65536; b++) {
        uint32_t u = b << 16;
        float x;
        memcpy(&x, &u, 4);
        float f = x / (1.0f + pexpn(x));
        silu_f[b] = f;
        silu_r[b] = r16(f);
    }
    silu_made = 1;
}
static inline uint32_t bits(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    return u;
}
static inline uint32_t bf(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    return u >> 16;
}

// The convolution step of n channels from r (n a multiple of 4; each channel's taps cw[4r..4r+3]): its input rounded
// to bf16, the sum of its four products in order, rounded, through SiLU's table (out), and its state shifted on in
// place, four channels at a time. A channel whose inputs are all +0 gets +0 (every sum stays +0) and its state stays
// +0, as the stepper's, so no channel is left out.
#if defined(__aarch64__)
static inline float32x4_t r16v(float32x4_t x) {
    const uint32x4_t u = vreinterpretq_u32_f32(x), e = vdupq_n_u32(0x7F800000u);
    const uint32x4_t special = vceqq_u32(vandq_u32(u, e), e);  // (inf, nan: as they are)
    const uint32x4_t lsb = vandq_u32(vshrq_n_u32(u, 16), vdupq_n_u32(1u));
    const uint32x4_t rnd = vandq_u32(vaddq_u32(vaddq_u32(u, vdupq_n_u32(0x7FFFu)), lsb), vdupq_n_u32(0xFFFF0000u));
    return vreinterpretq_f32_u32(vbslq_u32(special, u, rnd));
}
#endif
static void conv_step(const float* yall, float* cs, int C, const float* cw, int r, int n, const float* tab, float* out) {
#if defined(__aarch64__)
    const float32x4_t z = vdupq_n_f32(0.0f);
    for (int i = 0; i < n; i += 4) {
        const int q = r + i;
        const float32x4_t y = r16v(vld1q_f32(yall + q));
        const float32x4_t s0 = vld1q_f32(cs + q), s1 = vld1q_f32(cs + C + q), s2 = vld1q_f32(cs + 2 * C + q);
        const float32x4x4_t w = vld4q_f32(cw + 4 * q);
        float32x4_t c = vaddq_f32(z, vmulq_f32(s0, w.val[0]));
        c = vaddq_f32(c, vmulq_f32(s1, w.val[1]));
        c = vaddq_f32(c, vmulq_f32(s2, w.val[2]));
        c = vaddq_f32(c, vmulq_f32(y, w.val[3]));
        const uint32x4_t b = vshrq_n_u32(vreinterpretq_u32_f32(r16v(c)), 16);
        out[i] = tab[vgetq_lane_u32(b, 0)];
        out[i + 1] = tab[vgetq_lane_u32(b, 1)];
        out[i + 2] = tab[vgetq_lane_u32(b, 2)];
        out[i + 3] = tab[vgetq_lane_u32(b, 3)];
        vst1q_f32(cs + q, s1);
        vst1q_f32(cs + C + q, s2);
        vst1q_f32(cs + 2 * C + q, y);
    }
#else
    for (int i = 0; i < n; i++) {
        const int q = r + i;
        const float y = r16(yall[q]), s0 = cs[q], s1 = cs[C + q], s2 = cs[2 * C + q];
        float c = 0.0f;
        c = c + s0 * cw[4 * q];
        c = c + s1 * cw[4 * q + 1];
        c = c + s2 * cw[4 * q + 2];
        c = c + y * cw[4 * q + 3];
        out[i] = tab[bf(r16(c))];
        cs[q] = s1; cs[C + q] = s2; cs[2 * C + q] = y;
    }
#endif
}

// A main product's sums over the leading columns that do not change (the 62 ballast dims of the normed residual come
// first in the column order): made once and kept with the inputs they were made from (checked at every step, bit
// for bit; made again where they differ), then each step goes on from them over the other columns. Rows the other
// columns touch are listed (``mark``, ``list``): a unit no varying column touched has the output it had.
#define PRE 62
typedef struct {
    float* acc0;
    float hn0[PRE];
    int valid, changed, n, np;
    unsigned char* mark;
    int* list;   // rows touched this step
    int* ulist;  // units written this step (MLP)
    int* plist;  // and the step before
    unsigned char* umark;
    float* cout; // a unit's output from acc0 alone
} Pre;

static void spmv_pre(const Mat* M, const float* src, float* acc, Pre* P, int track) {
    const int b = M->ncols < PRE ? M->ncols : PRE;
    if (!P->valid || memcmp(P->hn0, src, sizeof(float) * b) != 0) {
        memset(P->acc0, 0, sizeof(float) * M->nrows);
        for (int c = 0; c < b; c++) {
            const float x = src[c];
            if (x == 0.0f) continue;
            for (int e = M->cp[c]; e < M->cp[c + 1]; e++) P->acc0[M->ri[e]] = P->acc0[M->ri[e]] + M->v[e] * x;
        }
        memcpy(P->hn0, src, sizeof(float) * b);
        P->valid = 1;
        P->changed = 1;
    }
    // the rows read densely (attention's) start from a copy of the kept sums; the ones read where touched (the
    // units') from their kept sum when a varying column first touches them (``val`` reads the others' kept sums)
    if (!track) memcpy(acc, P->acc0, sizeof(float) * M->nrows);
    P->n = 0;
    for (int c = b; c < M->ncols; c++) {
        const float x = src[c];
        if (x == 0.0f) continue;
        for (int e = M->cp[c]; e < M->cp[c + 1]; e++) {
            const int r = M->ri[e];
            if (track && !P->mark[r]) { P->mark[r] = 1; P->list[P->n++] = r; acc[r] = P->acc0[r]; }
            acc[r] = acc[r] + M->v[e] * x;
        }
    }
}
static inline float val(const Pre* P, const float* acc, int r) { return P->mark[r] ? acc[r] : P->acc0[r]; }

static void spmv(const Mat* M, const float* src, float* acc) {
    memset(acc, 0, sizeof(float) * M->nrows);
    for (int c = 0; c < M->ncols; c++) {
        const float x = src[c];
        if (x == 0.0f) continue;
        for (int e = M->cp[c]; e < M->cp[c + 1]; e++) acc[M->ri[e]] = acc[M->ri[e]] + M->v[e] * x;
    }
}

// the embedding row of token t into x
static void embed(const Tab* T, int t, float* x) {
    const int32_t* I = T->I;
    const float* Fl = T->Fl;
    int R = T->R;
    if (t < T->V0) {
        int lo = 0, hi = T->NSPEC - 1, si = -1;
        while (lo <= hi) {
            int m = (lo + hi) / 2, v = I[T->SPEC_IDS + m];
            if (v == t) { si = m; break; }
            if (v < t) lo = m + 1; else hi = m - 1;
        }
        for (int r = 0; r < R; r++) {
            float v;
            if (si >= 0) v = Fl[T->SPEC_ROWS + si * R + r];
            else {
                int k = I[T->BITOF + r];
                v = k >= 0 ? (float)((T->bits[t * 16 + k / 8] >> (7 - k % 8)) & 1) : Fl[T->PLAIN + r];
            }
            x[r] = v;
        }
    } else {
        int c = t - T->V0, e = -1, f = 0, p = 0;
        if (c >= T->EXPL_A && c < T->EXPL_A + T->EXPL_N) e = c - T->EXPL_A;
        else if (c >= T->CTRL) e = T->EXPL_N + c - T->CTRL;
        else { for (int i = 0; i < T->NFAMS; i++) if (c >= I[T->FBASE + i]) f = i; p = c - I[T->FBASE + f]; }
        for (int r = 0; r < R; r++) {
            float v;
            if (e >= 0) v = Fl[T->EXPL_ROWS + e * R + r];
            else {
                v = Fl[T->FAM_BASE + f * R + r];
                for (int i = 0; i < 16; i++) if ((p >> i) & 1) v += Fl[T->FAM_DELTA + (f * 16 + i) * R + r];
            }
            x[r] = r16(v);
        }
    }
}

// a kernel's start: the previous kernel's output rows added to the residual
static void resid(const Tab* T, const Mat* M, float* xs, const float* scr, int PROWS, int PN, int PSRC, float* acc) {
    const int32_t* I = T->I;
    spmv(M, scr + PSRC, acc);
    for (int i = 0; i < PN; i++) {
        int r = I[PROWS + i];
        xs[r] = r16(xs[r] + r16(acc[i]));
    }
}

static void norm(const Tab* T, const float* xs, float* hn) {
    // The ballast dims come first; the sum of squares, added in order, is theirs alone when every other square is
    // under half a unit in the last place of that sum (each addition then rounds back to it): checked, else summed
    float s = 0.0f;
    const int B = T->R < PRE ? T->R : PRE;
    for (int r = 0; r < B; r++) s += xs[r] * xs[r];
    const float hu = (nextafterf(s, INFINITY) - s) * 0.5f;
    int big = 0;
    for (int r = B; r < T->R; r++) big |= !(xs[r] * xs[r] < hu);  // (a NaN too)
    if (big) for (int r = B; r < T->R; r++) s += xs[r] * xs[r];
    float inv = prsqrt(s / (float)T->H + 1e-6f);
    for (int r = 0; r < T->R; r++) hn[r] = r16(xs[r] * inv);
}

// The key heads' queries and keys of an attention layer (made once for the value heads that share one), then its value
// heads: each the same work whether one thread does them all or several share them out (``tid`` of ``nth``).
typedef struct {
    float qk[32][256];  // normalized
    float raw[32][256]; // as the convolution and SiLU made them (what the normalized values follow from)
    int nrk[32], rowk[32][128], made[32];
    float eA[128];  // each value head's exp(A_log), made at its first step (eAm)
    unsigned char eAm[128];
} Keys;

typedef struct {
    const Tab* T;
    const int32_t* k;
    float* s;
    float* conv;
    uint32_t* mk;
    float* scr;
    const float* yall;
    Keys* K;
} AttnJob;

static void attn_keys(const AttnJob* J, int tid, int nth) {
    const Tab* T = J->T;
    const int32_t* I = T->I;
    const float* Fl = T->Fl;
    const int L = J->k[1];
    const int32_t* m = I + T->META + L * T->MSTRIDE;
    const int C = m[1], NK = m[3];
    const float* cw = Fl + m[7];
    float* cs = J->conv + m[25];  // (in place: each channel read and written once a step)
    const float* yall = J->yall;
    Keys* K = J->K;
    for (int kh = tid; kh < NK && kh < 32; kh += nth) {  // (every key head is used by a value head)
        int changed = !K->made[kh];
        float v[256];  // its queries' channels, then its keys'
        conv_step(yall, cs, C, cw, kh * 128, 128, silu_r, v);
        conv_step(yall, cs, C, cw, NK * 128 + kh * 128, 128, silu_r, v + 128);
        if (memcmp(v, K->raw[kh], sizeof(v)) != 0) { memcpy(K->raw[kh], v, sizeof(v)); changed = 1; }
        if (!changed) continue;  // (the same values as the step before: their norms and rows as they were)
        K->made[kh] = 1;
        for (int i = 0; i < 256; i++) K->qk[kh][i] = K->raw[kh][i];
        // normalized, and the rows they touch (the same for every head that shares the key head)
        float sq = 0.0f, sk = 0.0f;
        for (int i = 0; i < 128; i++) { sq += K->qk[kh][i] * K->qk[kh][i]; sk += K->qk[kh][128 + i] * K->qk[kh][128 + i]; }
        float rq = prsqrt(sq + 1e-6f), rk = prsqrt(sk + 1e-6f);
        for (int i = 0; i < 128; i++) { K->qk[kh][i] = K->qk[kh][i] * rq / 11.313708498984761f; K->qk[kh][128 + i] = K->qk[kh][128 + i] * rk; }
        K->nrk[kh] = 0;
        for (int i = 0; i < 128; i++) if (K->qk[kh][128 + i] != 0.0f || K->qk[kh][i] != 0.0f) K->rowk[kh][K->nrk[kh]++] = i;
    }
}

static void attn_heads(const AttnJob* J, int tid, int nth) {
    const Tab* T = J->T;
    const int32_t* I = T->I;
    const float* Fl = T->Fl;
    const int L = J->k[1], OOFF = J->k[2];
    const int32_t* m = I + T->META + L * T->MSTRIDE;
    const int C = m[1], NH = m[2], NK = m[3];
    const float* cw = Fl + m[7];
    float* cs = J->conv + m[25];
    float* s = J->s;
    uint32_t* mk = J->mk;
    float* scr = J->scr;
    const float* yall = J->yall;
    Keys* K = J->K;
    for (int h = tid; h < NH; h += nth) {
        float va[128], za[128], ab[2], ov[128];
        const int kh = I[m[8] + h];
        const float* qa = K->qk[kh];
        const float* ka = K->qk[kh] + 128;
        const int nr = K->nrk[kh];
        const int* rows = K->rowk[kh];
        for (int i = 0; i < 2; i++) ab[i] = r16(yall[C + NH * 128 + i * NH + h]);
        float* st = s + m[24] + h * 16384;
        uint32_t* mw = mk + (m[24] / 16384 + h) * 4;
        float bb = ab[0], aa = ab[1];
        if (h < 128 && !K->eAm[h]) { K->eA[h] = pexp(Fl[m[9] + h]); K->eAm[h] = 1; }
        const float eA = h < 128 ? K->eA[h] : pexp(Fl[m[9] + h]);
        float decay = 1.0f;  // (exp(-0 * sp) where exp(A_log) is 0: every head but layer 0's counters)
        if (eA != 0.0f) {
            float xa = aa + Fl[m[10] + h];
            float sp = (xa > 20.0f) ? xa : plog1p(pexp(fminf(xa, 20.0f)));
            decay = pexp(-eA * sp);
        }
        // a head no query or key entry touches, whose state does not decay, changes nothing and outputs 0 (most of a
        // layer's heads at a token): its value channels' convolution state still moves on
        const int idle = nr == 0 && decay == 1.0f;
        conv_step(yall, cs, C, cw, 2 * NK * 128 + h * 128, 128, silu_r, va);  // (an idle head's state moves on too)
        if (idle) {
            for (int dv = 0; dv < 128; dv++) scr[OOFF + h * 128 + dv] = 0.0f;
            continue;
        }
        for (int i = 0; i < 128; i++) za[i] = r16(yall[C + h * 128 + i]);
        float y0 = 1.0f / (1.0f + pexp(fabsf(bb)));
        const float beta = r16((bb < 0) ? y0 : 1.0f - y0);
        const uint32_t vm[4] = {mw[0], mw[1], mw[2], mw[3]};
#if defined(__aarch64__)
        if (nr <= 8 && decay == 1.0f) {
            // (no decay: every touched row's value columns four at a time, each column's sums in the scalar order)
            float kk[8], qq[8];
            for (int a = 0; a < nr; a++) { kk[a] = ka[rows[a]]; qq[a] = qa[rows[a]]; }
            const float32x4_t z = vdupq_n_f32(0.0f), bv = vdupq_n_f32(beta);
            for (int dv = 0; dv < 128; dv += 4) {
                float32x4_t sv[8];
                for (int a = 0; a < nr; a++) sv[a] = vld1q_f32(st + rows[a] * 128 + dv);
                float32x4_t kvm = z;
                for (int a = 0; a < nr; a++) if (kk[a] != 0.0f) kvm = vaddq_f32(kvm, vmulq_f32(sv[a], vdupq_n_f32(kk[a])));
                const float32x4_t delta = vmulq_f32(vsubq_f32(vld1q_f32(va + dv), kvm), bv);
                float32x4_t o = z;
                for (int a = 0; a < nr; a++) {
                    float32x4_t v = sv[a];
                    if (kk[a] != 0.0f) {
                        v = vaddq_f32(v, vmulq_f32(vdupq_n_f32(kk[a]), delta));
                        vst1q_f32(st + rows[a] * 128 + dv, v);
                    }
                    if (qq[a] != 0.0f) o = vaddq_f32(o, vmulq_f32(v, vdupq_n_f32(qq[a])));
                }
                vst1q_f32(ov + dv, o);
            }
            for (int a = 0; a < nr; a++) if (kk[a] != 0.0f) mw[rows[a] / 32] |= 1u << (rows[a] % 32);
        } else
#endif
        if (nr <= 8) {
            for (int dv = 0; dv < 128; dv++) {
                float sv[8];
                for (int a = 0; a < nr; a++) sv[a] = st[rows[a] * 128 + dv];
                float kvm = 0.0f;
                for (int a = 0; a < nr; a++) { float kk = ka[rows[a]]; if (kk != 0.0f) kvm += (sv[a] * decay) * kk; }
                float delta = (va[dv] - kvm) * beta;
                float o = 0.0f;
                for (int a = 0; a < nr; a++) {
                    int dk = rows[a];
                    float kk = ka[dk], qq = qa[dk];
                    float v = sv[a] * decay;
                    if (kk != 0.0f) v = v + kk * delta;
                    if (kk != 0.0f || decay != 1.0f) st[dk * 128 + dv] = v;
                    if (qq != 0.0f) o += v * qq;
                }
                if (decay != 1.0f) {
                    for (int w = 0; w < 4; w++) {
                        uint32_t b = vm[w];
                        while (b) {
                            int dk = 32 * w + __builtin_ctz(b);
                            b &= b - 1;
                            int touched = 0;
                            for (int a = 0; a < nr; a++) touched = touched || rows[a] == dk;
                            if (!touched) st[dk * 128 + dv] = st[dk * 128 + dv] * decay;
                        }
                    }
                }
                ov[dv] = o;
            }
            for (int a = 0; a < nr; a++) if (ka[rows[a]] != 0.0f) mw[rows[a] / 32] |= 1u << (rows[a] % 32);
        } else {
            for (int dv = 0; dv < 128; dv++) {
                float sv[128], kvm = 0.0f;
                for (int dk = 0; dk < 128; dk++) { sv[dk] = st[dk * 128 + dv] * decay; kvm += sv[dk] * ka[dk]; }
                float delta = (va[dv] - kvm) * beta;
                float o = 0.0f;
                for (int dk = 0; dk < 128; dk++) { sv[dk] = sv[dk] + ka[dk] * delta; o += sv[dk] * qa[dk]; st[dk * 128 + dv] = sv[dk]; }
                ov[dv] = o;
            }
            for (int dk = 0; dk < 128; dk++) if (ka[dk] != 0.0f) mw[dk / 32] |= 1u << (dk % 32);
        }
        float out[128], s2 = 0.0f;
        for (int dv = 0; dv < 128; dv++) { out[dv] = r16(ov[dv]); s2 += out[dv] * out[dv]; }
        float rn = prsqrt(s2 / 128.0f + 1e-6f);
#if defined(__aarch64__)
        const float32x4_t rv = vdupq_n_f32(rn);
        for (int dv = 0; dv < 128; dv += 4) {
            float32x4_t n = r16v(vmulq_f32(vld1q_f32(out + dv), rv));
            n = r16v(vmulq_f32(vld1q_f32(Fl + m[11] + dv), n));
            const float g[4] = {silu_f[bf(za[dv])], silu_f[bf(za[dv + 1])], silu_f[bf(za[dv + 2])], silu_f[bf(za[dv + 3])]};
            vst1q_f32(scr + OOFF + h * 128 + dv, r16v(vmulq_f32(n, vld1q_f32(g))));
        }
#else
        for (int dv = 0; dv < 128; dv++) {
            float z = za[dv];
            float n = r16(out[dv] * rn);
            n = r16(Fl[m[11] + dv] * n);
            scr[OOFF + h * 128 + dv] = r16(n * silu_f[bf(z)]);
        }
#endif
    }
}

// A run's thread team: threads 1..n-1 wait at a spin barrier for a phase (an attention layer's key heads, then its
// value heads), do their share, and meet again; thread 0 (the caller) does the rest of a step alone.
typedef struct {
    atomic_int count, sense;
    int n;
} Bar;

static void bar_wait(Bar* b, int* mine) {
    const int s = !*mine;
    *mine = s;
    if (atomic_fetch_add_explicit(&b->count, 1, memory_order_acq_rel) == b->n - 1) {
        atomic_store_explicit(&b->count, 0, memory_order_relaxed);
        atomic_store_explicit(&b->sense, s, memory_order_release);
    } else {
        // spin a while, then give the core up between looks (more threads than free cores must not stall a step)
        for (int i = 0; atomic_load_explicit(&b->sense, memory_order_acquire) != s; i++) {
            if (i < 4096) __asm__ volatile("yield"); else sched_yield();
        }
    }
}

// A thread's share of an MLP layer: its units [u0, u1), so its gate rows [u0, u1) and up rows [U + u0, U + u1), each
// column's entries for them (``split``: 4 per column, the start and end of each range; a column's entries are in row
// order) and its own lists. NULL ``split``: every row of every column (one thread).
typedef struct {
    int u0, u1, n, np;
    int *list, *ulist, *plist;
    int* split;
} Part;

typedef struct {
    const Tab* T;
    const int32_t* k;
    const Mat* M;
    Pre* P;
    Part* parts;
    const float* hn;
    float* scr;
    float* gall;
} MlpJob;

typedef struct {
    int n;
    Bar start, done;
    int phase;  // 1: key heads, 2: value heads, 3: an MLP layer's units, 0: the run is over
    AttnJob job;
    MlpJob mjob;
    pthread_t th[16];
} Team;

static void mlp_part(const MlpJob* J, Part* Q);

typedef struct {
    Team* team;
    int tid;
} Member;

static void* member(void* arg) {
    Member* me = (Member*)arg;
    Team* t = me->team;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);  // (a performance core: the barrier waits for the slowest)
    int s0 = 0, s1 = 0;
    for (;;) {
        bar_wait(&t->start, &s0);
        const int phase = t->phase;
        if (phase == 0) break;
        if (phase == 1) attn_keys(&t->job, me->tid, t->n);
        else if (phase == 2) attn_heads(&t->job, me->tid, t->n);
        else mlp_part(&t->mjob, t->mjob.parts + me->tid);
        bar_wait(&t->done, &s1);
    }
    return NULL;
}

static void attn_run(Team* t, const AttnJob* J, int* s0, int* s1) {
    if (t == NULL) {
        attn_keys(J, 0, 1);
        attn_heads(J, 0, 1);
        return;
    }
    t->job = *J;
    for (int phase = 1; phase <= 2; phase++) {
        t->phase = phase;
        bar_wait(&t->start, s0);
        if (phase == 1) attn_keys(J, 0, t->n); else attn_heads(J, 0, t->n);
        bar_wait(&t->done, s1);
    }
}

static inline float unit(float g, float up) {
    if (g == 0.0f || up == 0.0f) return 0.0f;  // (a zero product, added by no row)
    return r16(silu_r[bf(r16(g))] * r16(up));
}

// an MLP layer's units of a share (see Part): their rows' sums going on from the kept ones over the varying columns
// (each row's terms in the column order), the units those columns touch made, the others' outputs from the kept sums
static void mlp_part(const MlpJob* J, Part* Q) {
    const Tab* T = J->T;
    const Mat* M = J->M;
    Pre* P = J->P;
    const int32_t* m = T->I + T->META + J->k[1] * T->MSTRIDE;
    const int U = m[17];
    const int b = M->ncols < PRE ? M->ncols : PRE;
    const float* hn = J->hn;
    float* gall = J->gall;
    float* out = J->scr + J->k[2];
    Q->n = 0;
    for (int c = b; c < M->ncols; c++) {
        const float x = hn[c];
        if (x == 0.0f) continue;
        for (int seg = 0; seg < (Q->split ? 2 : 1); seg++) {
            const int e0 = Q->split ? Q->split[4 * c + 2 * seg] : M->cp[c];
            const int e1 = Q->split ? Q->split[4 * c + 2 * seg + 1] : M->cp[c + 1];
            for (int e = e0; e < e1; e++) {
                const int r = M->ri[e];
                if (!P->mark[r]) { P->mark[r] = 1; Q->list[Q->n++] = r; gall[r] = P->acc0[r]; }
                gall[r] = gall[r] + M->v[e] * x;
            }
        }
    }
    if (P->changed) {  // (the sums from the unchanging columns made again: every unit's output from them)
        for (int u = Q->u0; u < Q->u1; u++) out[u] = P->cout[u] = unit(P->acc0[u], P->acc0[U + u]);
        Q->np = 0;
    }
    int nu = 0;
    for (int i = 0; i < Q->n; i++) {  // units a varying column touched
        int r = Q->list[i], u = r < U ? r : r - U;
        if (P->umark[u]) continue;
        P->umark[u] = 1;
        Q->ulist[nu++] = u;
        out[u] = unit(val(P, gall, u), val(P, gall, U + u));
    }
    for (int i = 0; i < Q->n; i++) P->mark[Q->list[i]] = 0;
    for (int i = 0; i < Q->np; i++) {  // and those touched the step before, not now: their output from the sums kept
        int u = Q->plist[i];
        if (!P->umark[u]) out[u] = P->cout[u];
    }
    for (int i = 0; i < nu; i++) { P->umark[Q->ulist[i]] = 0; Q->plist[i] = Q->ulist[i]; }
    Q->np = nu;
}

// the kept sums over the ballast columns, made again where their inputs differ from the ones they were made from
static void ballast(const Mat* M, const float* src, Pre* P) {
    const int b = M->ncols < PRE ? M->ncols : PRE;
    if (P->valid && memcmp(P->hn0, src, sizeof(float) * b) == 0) return;
    memset(P->acc0, 0, sizeof(float) * M->nrows);
    for (int c = 0; c < b; c++) {
        const float x = src[c];
        if (x == 0.0f) continue;
        for (int e = M->cp[c]; e < M->cp[c + 1]; e++) P->acc0[M->ri[e]] = P->acc0[M->ri[e]] + M->v[e] * x;
    }
    memcpy(P->hn0, src, sizeof(float) * b);
    P->valid = 1;
    P->changed = 1;
}

// an MLP layer, its units shared out to the team (each share its own rows: the same work on whichever does it)
static void mlp_run(Team* t, const MlpJob* J, int* s0, int* s1) {
    ballast(J->M, J->hn, J->P);
    if (t == NULL) mlp_part(J, J->parts);
    else {
        t->mjob = *J;
        t->phase = 3;
        bar_wait(&t->start, s0);
        mlp_part(J, J->parts);
        bar_wait(&t->done, s1);
    }
    J->P->changed = 0;
}

// a share's entries in each column of an MLP layer's matrix (its rows' ranges: a column's entries are in row order)
static int* share(const Mat* M, int U, int u0, int u1) {
    int* sp = malloc(sizeof(int) * 4 * (M->ncols > 0 ? M->ncols : 1));
    for (int c = 0; c < M->ncols; c++) {
        const int lo[4] = {u0, u1, U + u0, U + u1};
        for (int j = 0; j < 4; j++) {  // the first entry of the column whose row is >= lo[j]
            int a = M->cp[c], z = M->cp[c + 1];
            while (a < z) { int h = (a + z) / 2; if (M->ri[h] < lo[j]) a = h + 1; else z = h; }
            sp[4 * c + j] = a;
        }
    }
    return sp;
}

// one run from token `first`: out[k] is the token predicted after the k-th token read (-1: Qwen's turn; < -1: a
// failed code); returns how many were written. s, mk updated in place; the convolutions' state after the run is in
// cA if the return's bit 30 is clear, else in cB (the count is the return's low bits).
int qp_run(const Tab* T, float* s, float* cA, float* cB, uint32_t* mk, float* scr, float* work, int first, int32_t* out,
           int max_out, int nthreads) {
    const int R = T->R;
    float x[4096], hn[4096];
    if (R > 4096) return -100;
    if (!silu_made) return -101;  // (qp_init first)
    Keys* keys = calloc(T->nk, sizeof(Keys));  // (a layer's: kept from step to step)
    Team* team = NULL;
    Member mem[16];
    int s0 = 0, s1 = 0;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    if (nthreads > 16) nthreads = 16;
    if (nthreads > 1) {
        team = calloc(1, sizeof(Team));
        team->n = nthreads;
        team->start.n = team->done.n = nthreads;
        for (int i = 1; i < nthreads; i++) {
            mem[i].team = team;
            mem[i].tid = i;
            pthread_create(&team->th[i], NULL, member, &mem[i]);
        }
    }
    Pre* pre = calloc(T->nk, sizeof(Pre));
    for (int i = 0; i < T->nk; i++) {
        const int n = T->main[i].nrows;
        pre[i].acc0 = malloc(sizeof(float) * n);
        pre[i].mark = calloc(n, 1);
        pre[i].list = malloc(sizeof(int) * n);
        pre[i].umark = calloc(n, 1);
        pre[i].ulist = malloc(sizeof(int) * n);
        pre[i].plist = malloc(sizeof(int) * n);
        pre[i].cout = malloc(sizeof(float) * n);
    }
    const int nparts = team != NULL ? nthreads : 1;
    Part* parts = calloc((size_t)T->nk * nparts, sizeof(Part));  // (an MLP layer's shares: one a thread)
    for (int i = 0; i < T->nk; i++) {
        const int32_t* k = T->prog + 9 * i;
        if (k[0] != 1) continue;
        const int U = T->I[T->META + k[1] * T->MSTRIDE + 17];
        for (int j = 0; j < nparts; j++) {
            Part* q = parts + i * nparts + j;
            q->u0 = (int)((long)U * j / nparts);
            q->u1 = (int)((long)U * (j + 1) / nparts);
            const int rows = 2 * (q->u1 - q->u0) + 1;
            q->list = malloc(sizeof(int) * rows);
            q->ulist = malloc(sizeof(int) * rows);
            q->plist = malloc(sizeof(int) * rows);
            q->split = nparts > 1 ? share(T->main + i, U, q->u0, q->u1) : NULL;
        }
    }
    (void)cB;
    int cur = first, n = 0;
    embed(T, cur, x);
    while (n < max_out) {
        for (int i = 0; i < T->nk; i++) {
            const int32_t* k = T->prog + 9 * i;
            if (k[7] > 0) resid(T, T->res + i, x, scr, k[4], k[7], k[8], work);
            norm(T, x, hn);
            if (k[0] == 0) { spmv_pre(T->main + i, hn, work, pre + i, 0); AttnJob J = {T, k, s, cA, mk, scr, work, keys + i}; attn_run(team, &J, &s0, &s1); }
            else { MlpJob J = {T, k, T->main + i, pre + i, parts + i * nparts, hn, scr, work}; mlp_run(team, &J, &s0, &s1); }
        }
        const int32_t* tk = T->tok;
        if (tk[3] > 0) resid(T, T->res + T->nk, x, scr, tk[0], tk[3], tk[4], work);
        const int32_t* I = T->I;
        int fid = 0, pay = 0, set = 0;
        for (int i = 0; i < T->NFAM + T->NPAY; i++) {
            float c = x[I[T->CODE + i]];
            set = set || (c != 0.0f);
            int bit = c > 0.0f ? 1 : 0;
            if (i < T->NFAM) fid |= bit << i; else pay |= bit << (i - T->NFAM);
        }
        int tok = -1;
        if (set) {
            if (fid < T->NFAMS) tok = pay < I[T->FSIZE + fid] ? T->V0 + I[T->FBASE + fid] + pay : -2;
            else if (fid < T->NFAMS + T->NCTRL) tok = T->V0 + T->CTRL + (fid - T->NFAMS);
            else if (fid == T->NFAMS + T->NCTRL && pay < T->NPRINT) tok = I[T->PRINT + pay];
            else tok = -2;
        }
        if (tok == -1 && cur >= T->V0) tok = -3;
        out[n++] = tok;
        if (tok < 0) break;
        cur = tok;
        embed(T, cur, x);
    }
    if (team != NULL) {
        team->phase = 0;
        bar_wait(&team->start, &s0);
        for (int i = 1; i < nthreads; i++) pthread_join(team->th[i], NULL);
        free(team);
    }
    free(keys);
    for (int i = 0; i < T->nk * nparts; i++) {
        free(parts[i].list); free(parts[i].ulist); free(parts[i].plist); free(parts[i].split);
    }
    free(parts);
    for (int i = 0; i < T->nk; i++) {
        free(pre[i].acc0); free(pre[i].mark); free(pre[i].list); free(pre[i].umark); free(pre[i].ulist); free(pre[i].plist);
        free(pre[i].cout);
    }
    free(pre);
    return n;  // (the convolutions' state after the run: cA)
}

void qp_init(void) { if (!silu_made) make_silu(); }
"""

P = ctypes.c_void_p


class _Mat(ctypes.Structure):
    _fields_ = [("cp", P), ("ri", P), ("v", P), ("ncols", ctypes.c_int), ("nrows", ctypes.c_int)]


_CONSTS = ("R H V0 CTRL NFAM NPAY NFAMS NCTRL NPRINT NSPEC EXPL_A EXPL_N META MSTRIDE SCR CSTRIDE SPEC_IDS BITOF FBASE FSIZE "
           "CODE PRINT SPEC_ROWS PLAIN EXPL_ROWS FAM_BASE FAM_DELTA").split()


class _Tab(ctypes.Structure):
    _fields_ = [("I", P), ("Z", P), ("Fl", P), ("bits", P), ("prog", P), ("tok", P), ("main", P), ("res", P),
                ("nk", ctypes.c_int)] + [(n, ctypes.c_int) for n in _CONSTS]


def build(folder: Path) -> Optional[Path]:
    """The library, built once for this source (None: no compiler, or it failed)."""
    tag = hashlib.sha256(SOURCE.encode()).hexdigest()[:12]
    lib = Path(folder) / f"computer-{tag}.dylib"
    if lib.exists():
        return lib
    src = Path(folder) / f"computer-{tag}.c"
    tmp = lib.with_suffix(f".{os.getpid()}.tmp")
    try:
        src.write_text(SOURCE)
        # (no contraction into fused multiply-adds: every rounding as the GPU's)
        subprocess.run(["clang", "-O3", "-ffp-contract=off", "-shared", "-fPIC", "-o", str(tmp), str(src)], check=True,
                       capture_output=True, timeout=120)
        os.replace(tmp, lib)
        return lib
    except (OSError, subprocess.SubprocessError):
        tmp.unlink(missing_ok=True)
        return None


class CPU:
    """The computer's runs on CPU cores, on a Device's tables (copied into host arrays once)."""

    @classmethod
    def make(cls, dev, folder: Path) -> Optional["CPU"]:
        lib = build(folder)
        return cls(dev, lib) if lib is not None else None

    def __init__(self, dev, lib: Path):
        self.lib = ctypes.CDLL(str(lib))
        self.lib.qp_init.restype = None
        self.lib.qp_init()  # (its SiLU tables)
        self.lib.qp_run.restype = ctypes.c_int
        self.lib.qp_run.argtypes = [ctypes.POINTER(_Tab), P, P, P, P, P, P, ctypes.c_int, P, ctypes.c_int, ctypes.c_int]
        st, W = dev.stepper, dev.w
        self._own: list = []
        I, Z = self._keep(st.I, np.int32), self._keep(W.Z, np.uint32)
        Fl, bits = self._keep(W.Fl, np.float32), self._keep(W.bits, np.uint8)
        prog = []
        for kind, L, n, kc in st.kernels:
            prog += [0 if kind == "attn" else 1, L, kc.get("OOFF", kc.get("POFF")), kc.get("KWRITE", 0), kc["PROWS"], kc["PPTR"],
                     kc["PNNZ"], kc["PN"], kc["PSRC"]]
        prog = self._keep(prog, np.int32)
        tok = self._keep([st.token[k] for k in ("PROWS", "PPTR", "PNNZ", "PN", "PSRC")], np.int32)
        M = st.consts["MSTRIDE"]
        meta = I[st.consts["META"]:st.consts["META"] + W.chip.n_layers * M].reshape(-1, M)
        mains = [self._by_cols(I, Z, *((m[5], m[4], m[6]) if kind == "attn" else (m[18], 2 * m[17], m[19])))
                 for kind, L, n, kc in st.kernels for m in [meta[L]]]
        ress = [self._by_cols(I, Z, kc["PPTR"], kc["PN"], kc["PNNZ"]) for _, _, _, kc in st.kernels]
        ress.append(self._by_cols(I, Z, st.token["PPTR"], st.token["PN"], st.token["PNNZ"]))
        self._mains, self._ress = (_Mat * len(mains))(*mains), (_Mat * len(ress))(*ress)
        self.work = max([m.nrows for m in mains + ress] + [1])
        self.scr = max(int(st.scr), 1)
        self.tab = _Tab(I.ctypes.data, Z.ctypes.data, Fl.ctypes.data, bits.ctypes.data, prog.ctypes.data, tok.ctypes.data,
                        ctypes.cast(self._mains, P), ctypes.cast(self._ress, P), len(st.kernels),
                        *[int(st.consts[n]) for n in _CONSTS])

    def _keep(self, a, dtype) -> np.ndarray:
        a = np.ascontiguousarray(np.array(a), dtype=dtype)
        self._own.append(a)
        return a

    def _by_cols(self, I, Z, ptr_off: int, n: int, z_off: int) -> _Mat:
        """A block of the tables' rows (CSR) by columns, each column's entries by row."""
        if n == 0:
            return _Mat(0, 0, 0, 0, 0)
        ptr = I[ptr_off:ptr_off + n + 1]
        z = Z[z_off + ptr[0]:z_off + ptr[-1]]
        rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(ptr))
        cols = (z >> 16).astype(np.int32)
        vals = ((z & 0xFFFF).astype(np.uint32) << 16).view(np.float32)
        order = np.lexsort((rows, cols))
        ncols = int(cols.max()) + 1 if len(cols) else 0
        cp = self._keep(np.r_[0, np.cumsum(np.bincount(cols, minlength=ncols))], np.int32)
        ri, v = self._keep(rows[order], np.int32), self._keep(vals[order], np.float32)
        return _Mat(cp.ctypes.data, ri.ctypes.data, v.ctypes.data, ncols, n)

    def run(self, s: np.ndarray, c: np.ndarray, mk: np.ndarray, first: int, max_tokens: int = 250_000,
            threads: int = 1) -> Tuple[List[int], np.ndarray]:
        """A run from ``first`` (the token the computer predicted), as ``Computer.step`` makes it: the run's tokens
        after ``first`` (a negative last one: it failed); the packed state ``s``, convolutions ``c`` and masks ``mk``
        (contiguous float32, float32 and uint32: the computer's own buffers, see ``shared``) updated in place;
        returns ``c`` too. ``threads``: cores the run's attention layers are shared out to (each key head and value
        head the same work on whichever does it). (The GIL is let go meanwhile.)"""
        for a, dt in ((s, np.float32), (c, np.float32), (mk, np.uint32)):
            if a.dtype != dt or not a.flags.c_contiguous or not a.flags.writeable:
                raise ValueError("the computer's buffers: contiguous and writable, float32 state and convolutions, uint32 masks")
        cA = c.reshape(-1)
        cB = np.zeros(1, np.float32)  # (unused: the convolutions' state moves on in place)
        out = np.zeros(max_tokens, np.int32)
        scr, work = np.zeros(self.scr, np.float32), np.zeros(self.work, np.float32)
        r = self.lib.qp_run(ctypes.byref(self.tab), s.ctypes.data, cA.ctypes.data, cB.ctypes.data, mk.ctypes.data, scr.ctypes.data,
                            work.ctypes.data, first, out.ctypes.data, max_tokens, threads)
        if r < 0:
            raise RuntimeError(f"the computer's CPU run failed ({r})")
        run, ended = [], False
        for t in out[:r].tolist():
            if t < 0:
                if t != -1:
                    run.append(t)
                ended = True
                break
            run.append(t)
        if not ended:
            run.append(-5)  # (too long: max_tokens without an end, as the GPU's Pool ends one)
        return run, c


def shared(arrays) -> Optional[List[np.ndarray]]:
    """NumPy views of evaluated MLX arrays' own buffers, for a run on a core to update in place (on Apple silicon
    the GPU's memory is the CPU's: no copy either way), from (array, dtype) pairs; None if one cannot be (another
    dtype, not contiguous, not writable)."""
    out = []
    for a, dt in arrays:
        try:
            v = np.array(a, copy=False)
        except (ValueError, TypeError):
            return None
        if v.dtype != dt or not v.flags.c_contiguous or not v.flags.writeable:
            return None
        out.append(v)
    return out
