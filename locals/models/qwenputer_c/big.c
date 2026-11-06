/* Big whole numbers for the tools that need more than 64 bits (statistics, money, number theory, units): base 2^32
   limbs, least significant first, up to BIG_LIMBS (512 bits, 154 decimal digits); a decimal is a big and a scale. */
#include "tools.h"

void big_set(big *a, u64 v) {
    a->n = 0;
    while (v) a->d[a->n++] = (u32)v, v >>= 32;
}

static void trim_big(big *a) {
    while (a->n && !a->d[a->n - 1]) a->n--;
}

int big_cmp(const big *a, const big *b) {
    if (a->n != b->n) return a->n < b->n ? -1 : 1;
    for (int i = a->n - 1; i >= 0; i--)
        if (a->d[i] != b->d[i]) return a->d[i] < b->d[i] ? -1 : 1;
    return 0;
}

void big_add(big *r, const big *a, const big *b) {
    int n = a->n > b->n ? a->n : b->n;
    u64 c = 0;
    for (int i = 0; i < n; i++) {
        c += (u64)(i < a->n ? a->d[i] : 0) + (i < b->n ? b->d[i] : 0);
        r->d[i] = (u32)c;
        c >>= 32;
    }
    r->n = n;
    if (c) {
        if (n == BIG_LIMBS) fail("a number beyond the computer's 154 digits");
        r->d[r->n++] = (u32)c;
    }
}

void big_sub(big *r, const big *a, const big *b) { /* a - b, a >= b */
    i64 c = 0;
    for (int i = 0; i < a->n; i++) {
        c += (i64)a->d[i] - (i < b->n ? b->d[i] : 0);
        r->d[i] = (u32)c;
        c = c < 0 ? -1 : 0;
    }
    r->n = a->n;
    trim_big(r);
}

void big_mul_small(big *r, const big *a, u32 m, u32 add) {
    u64 c = add;
    int n = a->n;
    for (int i = 0; i < n; i++) {
        c += (u64)a->d[i] * m;
        r->d[i] = (u32)c;
        c >>= 32;
    }
    r->n = n;
    if (c) {
        if (n == BIG_LIMBS) fail("a number beyond the computer's 154 digits");
        r->d[r->n++] = (u32)c;
    }
    trim_big(r);
}

void big_mul(big *r, const big *a, const big *b) {
    big t;
    if (a->n + b->n > BIG_LIMBS + 1) fail("a number beyond the computer's 154 digits");
    memset(t.d, 0, sizeof t.d);
    for (int i = 0; i < a->n; i++) {
        u64 c = 0;
        for (int j = 0; j < b->n; j++) {
            if (i + j >= BIG_LIMBS) {
                if (a->d[i] && b->d[j]) fail("a number beyond the computer's 154 digits");
                continue;
            }
            c += (u64)a->d[i] * b->d[j] + t.d[i + j];
            t.d[i + j] = (u32)c;
            c >>= 32;
        }
        if (c) {
            if (i + b->n >= BIG_LIMBS) fail("a number beyond the computer's 154 digits");
            t.d[i + b->n] = (u32)c;
        }
    }
    t.n = a->n + b->n > BIG_LIMBS ? BIG_LIMBS : a->n + b->n;
    trim_big(&t);
    *r = t;
}

u32 big_div_small(big *q, const big *a, u32 m) { /* the remainder */
    u64 r = 0;
    for (int i = a->n - 1; i >= 0; i--) {
        r = r << 32 | a->d[i];
        q->d[i] = (u32)(r / m);
        r %= m;
    }
    q->n = a->n;
    trim_big(q);
    return (u32)r;
}

static int top_bit(const big *a) { return a->n ? (a->n - 1) * 32 + 31 - __builtin_clz(a->d[a->n - 1]) : -1; }

/* (u1:u0) / v and its remainder, u1 < v, v's top bit set: two 32-bit divisions of 16-bit digits (Hacker's Delight's
   divlu), not a 64-bit division's loop */
static u32 div_2by1(u32 u1, u32 u0, u32 v, u32 *rem) {
    u32 vh = v >> 16, vl = v & 0xffff, uh = u0 >> 16, ul = u0 & 0xffff;
    u32 q1 = u1 / vh, rh = u1 - q1 * vh;
    while (q1 >> 16 || q1 * vl > (rh << 16 | uh)) {
        q1--, rh += vh;
        if (rh >> 16) break;
    }
    u32 mid = (u1 << 16 | uh) - q1 * v;
    u32 q0 = mid / vh;
    rh = mid - q0 * vh;
    while (q0 >> 16 || q0 * vl > (rh << 16 | ul)) {
        q0--, rh += vh;
        if (rh >> 16) break;
    }
    *rem = (mid << 16 | ul) - q0 * v;
    return q1 << 16 | q0;
}

void big_divmod(big *q, big *r, const big *a, const big *b) { /* a / b and its remainder: Knuth's long division */
    if (!b->n) fail("division by zero");
    if (b->n == 1) {
        u32 m = big_div_small(q, a, b->d[0]);
        big_set(r, m);
        return;
    }
    if (big_cmp(a, b) < 0) {
        *r = *a;
        q->n = 0;
        return;
    }
    int n = b->n, m = a->n - n, sh = __builtin_clz(b->d[n - 1]);
    u32 v[BIG_LIMBS], u[BIG_LIMBS + 1];
    for (int i = n - 1; i > 0; i--) v[i] = sh ? b->d[i] << sh | b->d[i - 1] >> (32 - sh) : b->d[i];
    v[0] = b->d[0] << sh;
    u[a->n] = sh ? a->d[a->n - 1] >> (32 - sh) : 0;
    for (int i = a->n - 1; i > 0; i--) u[i] = sh ? a->d[i] << sh | a->d[i - 1] >> (32 - sh) : a->d[i];
    u[0] = a->d[0] << sh;
    big quo;
    quo.n = m + 1;
    for (int j = m; j >= 0; j--) {
        u32 qh, rh;
        int test = 1;
        if (u[j + n] >= v[n - 1]) { /* (equal: a quotient digit of 2^32 - 1 at most) */
            qh = 0xffffffffu;
            u64 t = (u64)u[j + n - 1] + v[n - 1];
            rh = (u32)t;
            test = !(t >> 32);
        } else
            qh = div_2by1(u[j + n], u[j + n - 1], v[n - 1], &rh);
        while (test && (u64)qh * v[n - 2] > ((u64)rh << 32 | u[j + n - 2])) {
            qh--;
            u64 t = (u64)rh + v[n - 1];
            rh = (u32)t;
            test = !(t >> 32);
        }
        i64 k = 0, t;
        for (int i = 0; i < n; i++) { /* u -= qh v */
            u64 p = (u64)qh * v[i];
            t = (i64)u[i + j] - k - (i64)(u32)p;
            u[i + j] = (u32)t;
            k = (i64)(p >> 32) - (t >> 32);
        }
        t = (i64)u[j + n] - k;
        u[j + n] = (u32)t;
        if (t < 0) { /* (qh one too big: v added back) */
            qh--;
            u64 c = 0;
            for (int i = 0; i < n; i++) {
                c += (u64)u[i + j] + v[i];
                u[i + j] = (u32)c;
                c >>= 32;
            }
            u[j + n] += (u32)c;
        }
        quo.d[j] = qh;
    }
    trim_big(&quo);
    for (int i = 0; i < n; i++) r->d[i] = sh ? u[i] >> sh | u[i + 1] << (32 - sh) : u[i];
    r->n = n;
    trim_big(r);
    *q = quo;
}

void big_pow10(big *a, int k) {
    big_set(a, 1);
    for (; k >= 9; k -= 9) big_mul_small(a, a, 1000000000u, 0);
    static const u32 P[9] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000};
    if (k) big_mul_small(a, a, P[k], 0);
}

void big_isqrt(big *r, const big *a) { /* Newton's: from above */
    if (!a->n) {
        r->n = 0;
        return;
    }
    big x, y, q, rem;
    big_set(&x, 1);
    for (int i = (top_bit(a) + 2) / 2; i > 0; i--) big_mul_small(&x, &x, 2, 0); /* (2^ceil(bits/2) >= sqrt) */
    for (;;) {
        big_divmod(&q, &rem, a, &x);
        big_add(&y, &x, &q);
        big_div_small(&y, &y, 2);
        if (big_cmp(&y, &x) >= 0) break;
        x = y;
    }
    *r = x;
}

int big_digits(const big *a, char *out) { /* its decimal digits, most significant first; their count */
    big t = *a;
    char tmp[162]; /* (18 pieces of 9 digits: 512 bits are 155 digits) */
    int n = 0;
    while (t.n) {
        u32 r = big_div_small(&t, &t, 1000000000u);
        for (int k = 0; k < 9; k++) tmp[n++] = '0' + r % 10, r /= 10;
    }
    while (n > 1 && tmp[n - 1] == '0') n--;
    if (!n) tmp[n++] = '0';
    for (int i = 0; i < n; i++) out[i] = tmp[n - 1 - i];
    return n;
}

void say_big(const big *a) {
    char d[160];
    int n = big_digits(a, d);
    for (int i = 0; i < n; i++) put(d[i]);
}

/* a decimal: neg, the digits as a big, and how many of them are after the point */
void say_decimal(const big *a, int neg, int scale, int max_frac) { /* (the fraction cut to max_frac, rounded half up) */
    big t = *a;
    if (scale > max_frac) {
        big p, q, r;
        big_pow10(&p, scale - max_frac);
        big_divmod(&q, &r, &t, &p);
        big_mul_small(&r, &r, 2, 0);
        if (big_cmp(&r, &p) >= 0) {
            big one;
            big_set(&one, 1);
            big_add(&q, &q, &one);
        }
        t = q, scale = max_frac;
    }
    char d[160];
    int n = big_digits(&t, d);
    while (scale > 0 && n > 0 && d[n - 1] == '0' && !(n == 1 && d[0] == '0')) n--, scale--; /* (trailing zeros) */
    if (neg && !(n == 1 && d[0] == '0')) put('-');
    if (n <= scale) {
        put('0');
        put('.');
        for (int i = 0; i < scale - n; i++) put('0');
        for (int i = 0; i < n; i++) put(d[i]);
        return;
    }
    for (int i = 0; i < n; i++) {
        if (i == n - scale) put('.');
        put(d[i]);
    }
}

/* a number's text (12, -3.5, 1,234.56, $12, 45%) as a big and a scale: 0 if it is not one */
int read_decimal(str s, big *a, int *neg, int *scale) {
    s = trim(s);
    const char *p = s.p, *e = s.e;
    *neg = 0, *scale = 0;
    big_set(a, 0);
    if (p < e && (*p == '-' || *p == '+')) *neg = *p == '-', p++;
    else if (e - p >= 3 && (u8)p[0] == 0xe2 && (u8)p[1] == 0x88 && (u8)p[2] == 0x92) *neg = 1, p += 3;
    if (p < e && *p == '$') p++;
    if (e > p && e[-1] == '%') e--;
    int digits = 0, point = 0;
    for (; p < e; p++) {
        if (digit(*p)) {
            big_mul_small(a, a, 10, *p - '0');
            digits++;
            if (point) ++*scale;
            if (digits > 150) return 0;
        } else if (*p == '.' && !point)
            point = 1;
        else if (*p == ',' && !point && p + 3 < e + 1 && digit(p[1]))
            continue;
        else if (*p == '_' && digits)
            continue;
        else
            return 0;
    }
    return digits > 0;
}
