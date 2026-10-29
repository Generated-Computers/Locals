"""One Metal kernel for a Gated DeltaNet decode step of S <= 16 tokens.

Stock mlx-lm runs about fourteen small kernels per linear-attention layer for
one decode token: conv-state concat and slice, depthwise conv, silu, two q/k
RMS norms and scalings, sigmoid(b), the decay g, the recurrence, and the gated
RMS norm of the output. At batch 1 these launches cost more than their
arithmetic. Here one threadgroup owns one (sequence, value head):

  1. conv + silu for the head's q, k and v channels (and write the new conv state)
  2. RMS-normalize q and k (scaled by Dk^-1/2 and Dk^-1 as in mlx-lm)
  3. the delta-rule recurrence, one simdgroup per 4 rows of the head's state
  4. RMS-normalize the head's output and gate it with silu(z)

Rounding to the activation dtype happens at the same points as in mlx-lm, so
results match the stock path up to summation order.
"""

from __future__ import annotations

import mlx.core as mx

_SOURCE = r"""
    const uint lane = thread_position_in_threadgroup.x;
    const uint sgi = thread_position_in_threadgroup.y;
    const uint tid = sgi * 32 + lane;
    const uint n = threadgroup_position_in_grid.z;
    const uint b = n / Hv;
    const uint hv = n % Hv;
    const uint hk = hv / (Hv / Hk);
    constexpr int C = 2 * Hk * Dk + Hv * Dv;
    constexpr int P = C + Hv * Dv + 2 * Hv;
    constexpr int RPS = Dv / NSG;  // state rows per simdgroup
    const device T* prb = proj + b * S * P;
    const device T* csb = conv_state + b * (KC - 1) * C;

    threadgroup float q_s[Dk];
    threadgroup float k_s[Dk];
    threadgroup float v_s[Dv];
    threadgroup float o_s[Dv];
    threadgroup float gate_s[2];

    // For S > 1 the head's state rows stay in registers across the steps; a
    // single step streams them (load, update, store), which keeps register
    // pressure and occupancy as in a plain decode kernel.
    const uint base = (b * Hv + hv) * Dv;
    float4 st[S > 1 ? RPS : 1];
    if (S > 1) {
        for (int r = 0; r < RPS; r++) {
            st[r] = *((const device float4*)(state_in + (base + sgi * RPS + r) * Dk) + lane);
        }
    }
    const bool qk_owner = (hv % (Hv / Hk)) == 0;

    for (int t = 0; t < S; t++) {
        const device T* pr = prb + t * P;
        // 1. conv + silu over the window [conv state | inputs 0..t], and the gates.
        if (tid == 0) {  // (mlx-lm: beta = sigmoid(b), g = exp(-exp(A_log) * softplus(a + dt_bias)), each op MLX's)
            const T bb = pr[C + Hv * Dv + hv];
            const float aa = float(pr[C + Hv * Dv + Hv + hv]);
            const float beta = float(mlx_sigmoid<T>(bb));
            const T x = static_cast<T>(aa + float(dt_bias[hv]));
            const float sp = float(mlx_softplus<T>(x));
            gate_s[0] = metal::precise::exp(-metal::precise::exp(float(A_log[hv])) * sp);
            gate_s[1] = beta;
        }
        for (uint i = tid; i < 2 * Dk + Dv; i += NSG * 32) {
            int ch;
            if (i < Dk) {
                ch = hk * Dk + i;
            } else if (i < 2 * Dk) {
                ch = Hk * Dk + hk * Dk + (i - Dk);
            } else {
                ch = 2 * Hk * Dk + hv * Dv + (i - 2 * Dk);
            }
            float acc = 0.0f;
            for (int j = 0; j < KC; j++) {
                const int pos = t + j;  // position in [conv state (KC-1) | inputs (S)]
                const float xin = pos < KC - 1 ? float(csb[pos * C + ch]) : float(prb[(pos - (KC - 1)) * P + ch]);
                acc += float(conv_w[ch * KC + j]) * xin;
            }
            const T cv = static_cast<T>(acc);
            const float sv = float(static_cast<T>(float(cv) * float(mlx_sigmoid<T>(cv))));  // nn.silu, in T
            if (i < Dk) { q_s[i] = sv; }
            else if (i < 2 * Dk) { k_s[i - Dk] = sv; }
            else { v_s[i - 2 * Dk] = sv; }
            if (t == S - 1 && (i >= 2 * Dk || qk_owner)) {
                // New conv state: the last KC-1 entries of [conv state | inputs].
                device T* co = conv_state_out + b * (KC - 1) * C + ch;
                for (int j = 0; j < KC - 1; j++) {
                    const int pos = S + j;
                    co[j * C] = pos < KC - 1 ? csb[pos * C + ch] : prb[(pos - (KC - 1)) * P + ch];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. q/k RMS norms: simdgroup 0 for q, 1 for k.
        if (sgi < 2) {
            threadgroup float* src = sgi == 0 ? q_s : k_s;
            float4 v4 = float4(src[lane * 4], src[lane * 4 + 1], src[lane * 4 + 2], src[lane * 4 + 3]);
            const float ss = simd_sum(dot(v4, v4));
            const float r = metal::precise::rsqrt(ss / float(Dk) + 1e-6f / float(Dk));  // l2norm: eps on sum(x^2), as mlx-lm's rms_norm
            const float scale = sgi == 0 ? (1.0f / float(Dk)) : metal::rsqrt(float(Dk));
            for (int i = 0; i < 4; i++) {
                const float nv = float(static_cast<T>(v4[i] * r));
                src[lane * 4 + i] = float(static_cast<T>(nv * static_cast<float>(static_cast<T>(scale))));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. recurrence on the register-resident rows.
        {
            const float g = gate_s[0];
            const float beta = gate_s[1];
            const float4 q4 = float4(q_s[lane * 4], q_s[lane * 4 + 1], q_s[lane * 4 + 2], q_s[lane * 4 + 3]);
            const float4 k4 = float4(k_s[lane * 4], k_s[lane * 4 + 1], k_s[lane * 4 + 2], k_s[lane * 4 + 3]);
            for (int r = 0; r < RPS; r++) {
                const uint dv = sgi * RPS + r;
                float4 sr = (S > 1 ? st[r] : *((const device float4*)(state_in + (base + dv) * Dk) + lane)) * g;
                const float kv = simd_sum(dot(sr, k4));
                const float delta = (v_s[dv] - kv) * beta;
                sr += k4 * delta;
                if (S > 1) {
                    st[r] = sr;
                } else {
                    *((device float4*)(state_out + (base + dv) * Dk) + lane) = sr;
                }
                //STATES// *((device float4*)(states_out + ((size_t)t * B * Hv * Dv + base + dv) * Dk) + lane) = sr;
                const float o = simd_sum(dot(sr, q4));
                if (lane == 0) {
                    o_s[dv] = float(static_cast<T>(o));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. gated RMS norm of the head's output.
        if (sgi == 0) {
            float4 o4 = float4(o_s[lane * 4], o_s[lane * 4 + 1], o_s[lane * 4 + 2], o_s[lane * 4 + 3]);
            const float ss = simd_sum(dot(o4, o4));
            const float r = metal::precise::rsqrt(ss / float(Dv) + 1e-6f);
            device T* yo = y + (b * S + t) * Hv * Dv + hv * Dv;
            const device T* zp = pr + C + hv * Dv;
            for (int i = 0; i < 4; i++) {  // (MLX's rms_norm: w * T(x * r); precise_swiglu: silu(z) in float32 times it)
                const int d = lane * 4 + i;
                const float xn = float(static_cast<T>(float(norm_w[d]) * float(static_cast<T>(o4[i] * r))));
                const float z = float(zp[d]);
                const float e = 1.0f / (1.0f + metal::precise::exp(metal::abs(z)));
                yo[d] = static_cast<T>(z * (z < 0.0f ? e : 1.0f - e) * xn);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (S > 1) {
        for (int r = 0; r < RPS; r++) {
            *((device float4*)(state_out + (base + sgi * RPS + r) * Dk) + lane) = st[r];
        }
    }
"""

# MLX's sigmoid and softplus (logaddexp(x, 0)) in the type T, each step rounded as its kernels round it: in bf16
# 1 / (1 + exp(-x)) and log(1 + exp(x)) rounded once differ from them in 17% and 15% of values.
_HEADER = r"""
template <typename T>
inline T mlx_sigmoid(T x) {
    const T e = T(metal::precise::exp(metal::abs(float(x))));
    const T y = T(1.0f / float(T(1.0f + float(e))));
    return float(x) < 0.0f ? y : T(1.0f - float(y));
}

template <typename T>
inline T mlx_softplus(T x) {
    const float xf = float(x);
    if (isnan(xf)) return x;
    const float hi = metal::max(xf, 0.0f), lo = metal::min(xf, 0.0f);
    if (lo == -INFINITY || hi == INFINITY) return T(hi);
    const T e = T(metal::precise::exp(float(T(lo - hi))));
    const float xp1 = 1.0f + float(e);
    const T l = xp1 == 1.0f ? e : T(float(e) * (metal::precise::log(xp1) / (xp1 - 1.0f)));
    return T(hi + float(l));
}
"""

_KERNEL = {}


def _kernel(states: bool = False):
    k = _KERNEL.get(states)
    if k is None:
        # the states variant also writes the state after every token (for
        # rewinding a partially accepted draft); the line is absent from the plain one
        src = _SOURCE.replace("//STATES//", "" if states else "//")
        k = _KERNEL[states] = mx.fast.metal_kernel(
            name="locals_gdn_step3" + ("_states" if states else ""),
            input_names=["proj", "conv_state", "conv_w", "A_log", "dt_bias", "state_in", "norm_w"],
            output_names=["y", "conv_state_out", "state_out"] + (["states_out"] if states else []),
            source=src, header=_HEADER,
        )
    return k


def supported(Hk: int, Hv: int, Dk: int, Dv: int, KC: int) -> bool:
    return Dk == 128 and Dv == 128 and Hv % Hk == 0 and KC >= 2


MAX_S = 16


def gdn_decode(proj, conv_state, conv_w, A_log, dt_bias, state, norm_w, Hk, Hv, Dk, Dv, NSG=8, states: bool = False):
    """proj (B, S, P) -> y (B, S, Hv*Dv), new conv state, new recurrent state
    [, the recurrent state after each token, (S, B, Hv, Dv, Dk)]."""
    B, S = proj.shape[0], proj.shape[1]
    KC = conv_w.shape[1]
    C = conv_state.shape[-1]
    shapes = [(B, S, Hv * Dv), (B, KC - 1, C), state.shape]
    dtypes = [proj.dtype, conv_state.dtype, state.dtype]
    if states:
        shapes.append((S,) + tuple(state.shape))
        dtypes.append(state.dtype)
    outs = _kernel(states)(
        inputs=[proj, conv_state, conv_w, A_log, dt_bias, state, norm_w],
        template=[("T", proj.dtype), ("Hk", Hk), ("Hv", Hv), ("Dk", Dk), ("Dv", Dv), ("KC", KC), ("NSG", NSG), ("S", S),
                  ("B", B)],
        grid=(32, NSG, B * Hv),
        threadgroup=(32, NSG, 1),
        output_shapes=shapes,
        output_dtypes=dtypes,
    )
    return tuple(outs)
