/* Numbers: stats, percent_change, round_number, compound_interest, loan_payment, is_prime, factor, convert_units,
   roman_numeral, number_to_words. Exact where the answer is (big.c); a figure cut short says "(rounded)". */
#include "tools.h"

/* ---- signed bigs */
typedef struct {
    big m;
    int neg;
} sbig;

static void sb_add(sbig *r, const sbig *a, const sbig *b) {
    if (a->neg == b->neg) {
        big_add(&r->m, &a->m, &b->m);
        r->neg = a->neg;
        return;
    }
    int c = big_cmp(&a->m, &b->m);
    if (c >= 0) big_sub(&r->m, &a->m, &b->m), r->neg = a->neg;
    else big_sub(&r->m, &b->m, &a->m), r->neg = b->neg;
    if (!r->m.n) r->neg = 0;
}

static void say_sb(const sbig *a, int scale, int max_frac) { say_decimal(&a->m, a->neg, scale, max_frac); }

/* a / n (n > 0), to `frac` more decimals; rounded half up; 1 if that changed it */
static int sb_div_small(sbig *r, const sbig *a, u32 n, int frac) {
    big t, p, q, rem;
    big_pow10(&p, frac);
    big_mul(&t, &a->m, &p);
    big qq;
    u32 m = big_div_small(&qq, &t, n);
    q = qq;
    if (2ull * m >= n) {
        big one;
        big_set(&one, 1);
        big_add(&q, &q, &one);
    }
    (void)rem;
    r->m = q, r->neg = a->neg && q.n;
    return m != 0;
}

/* ---- a list of numbers: one a line, or separated by ", " or ";" (then a comma inside a number is a thousands
   separator), else by commas or spaces; each of up to 18 digits, 9 after the point */
static int read_numbers(str s, num *out, int most) {
    s = trim(s);
    char sep = 0;
    for (const char *p = s.p; p < s.e; p++) {
        if ((*p == '\n' || *p == ';') && !sep) sep = *p;
        if (*p == ',' && p + 1 < s.e && p[1] == ' ' && !sep) sep = ',';
    }
    int n = 0;
    const char *p = s.p;
    while (p < s.e) {
        const char *q = p;
        if (sep == ',') {
            while (q < s.e && !(*q == ',' && q + 1 < s.e && q[1] == ' ') && *q != '\n') q++;
        } else if (sep)
            while (q < s.e && *q != sep && *q != '\n') q++;
        else
            while (q < s.e && *q != ',' && !space(*q)) q++;
        str item = trim((str){p, q});
        p = q < s.e ? q + 1 : q;
        if (item.p == item.e) continue;
        if (n == most) fail("more than 256 numbers");
        if (!read_num(item, &out[n], 1)) {
            say("error (not a number of up to 18 digits: ");
            say_str(item);
            put(')');
            done();
        }
        n++;
    }
    if (!n) fail("no numbers");
    return n;
}

static void as_sbig(sbig *r, num v, int scale) { /* v * 10^scale as a big */
    u64 m = v.v < 0 ? -(u64)v.v : (u64)v.v;
    big_set(&r->m, m);
    r->neg = v.v < 0;
    if (scale > v.frac) {
        big p;
        big_pow10(&p, scale - v.frac);
        big_mul(&r->m, &r->m, &p);
    }
}

void stats(void) {
    str t = param("numbers");
    if (!have(t)) t = param("text");
    if (!have(t)) fail("missing argument: numbers");
    num *x = alloc(sizeof(num) * 256);
    int n = read_numbers(t, x, 256), S = 0;
    for (int i = 0; i < n; i++) S = x[i].frac > S ? x[i].frac : S;
    /* each number at the scale S in 64 bits where it is within 62 (most lists), else none (`w` 0) */
    i64 *w = alloc(sizeof(i64) * n);
    for (int i = 0; i < n && w; i++) {
        u64 m = x[i].v < 0 ? -(u64)x[i].v : (u64)x[i].v;
        int k = x[i].frac;
        for (; k < S && !(m >> 59); k++) m *= 10;
        if (k < S || m >> 62) w = 0;
        else w[i] = x[i].v < 0 ? -(i64)m : (i64)m;
    }
    int *o = alloc(sizeof(int) * n); /* (in order: Shell sort on their indices) */
    for (int i = 0; i < n; i++) o[i] = i;
    static const int gaps[] = {132, 57, 23, 10, 4, 1};
    for (int g = 0; g < 6; g++)
        for (int i = gaps[g]; i < n; i++) {
            int v = o[i], j = i;
            if (w)
                for (; j >= gaps[g] && w[o[j - gaps[g]]] > w[v]; j -= gaps[g]) o[j] = o[j - gaps[g]];
            else
                for (; j >= gaps[g] && num_cmp(x[o[j - gaps[g]]], x[v]) > 0; j -= gaps[g]) o[j] = o[j - gaps[g]];
            o[j] = v;
        }
    sbig sum = {{0}, 0}, xi;
    big sq = {0}, t2;
    if (w) { /* (sums in words: 256 numbers of 62 bits, their sum in 128 bits and their squares' in 192) */
        u64 sl = 0, a2 = 0, a1 = 0, a0 = 0;
        i64 sh = 0;
        for (int i = 0; i < n; i++) {
            u64 m = w[i] < 0 ? -(u64)w[i] : (u64)w[i], hi, lo, p0 = (u64)(u32)m * (u32)m, p1 = (u64)(u32)m * (u32)(m >> 32),
                p2 = (m >> 32) * (m >> 32), mid = (p0 >> 32) + 2 * (u64)(u32)p1;
            lo = mid << 32 | (u32)p0; /* (m^2 = hi:lo; p1 is m's cross product, twice) */
            hi = p2 + 2 * (p1 >> 32) + (mid >> 32);
            sl += (u64)w[i];
            sh += (w[i] < 0 ? -1 : 0) + (sl < (u64)w[i]);
            a0 += lo;
            hi += a0 < lo;
            a1 += hi;
            a2 += a1 < hi;
        }
        if (sh < 0) sl = -sl, sh = -sh - (sl != 0), sum.neg = 1;
        u32 d[6] = {(u32)sl, (u32)(sl >> 32), (u32)sh, (u32)((u64)sh >> 32)};
        sum.m.n = 4;
        memcpy(sum.m.d, d, 16);
        while (sum.m.n && !sum.m.d[sum.m.n - 1]) sum.m.n--;
        if (!sum.m.n) sum.neg = 0;
        u32 q[6] = {(u32)a0, (u32)(a0 >> 32), (u32)a1, (u32)(a1 >> 32), (u32)a2, (u32)(a2 >> 32)};
        sq.n = 6;
        memcpy(sq.d, q, 24);
        while (sq.n && !sq.d[sq.n - 1]) sq.n--;
    } else
        for (int i = 0; i < n; i++) {
            as_sbig(&xi, x[i], S);
            sb_add(&sum, &sum, &xi);
            big_mul(&t2, &xi.m, &xi.m);
            big_add(&sq, &sq, &t2);
        }
    say("count ");
    say_int(n);
    say(", sum ");
    say_sb(&sum, S, 9);
    sbig mean;
    int r = sb_div_small(&mean, &sum, n, 9);
    say(", mean ");
    say_sb(&mean, S + 9, 9 + S);
    if (r) say(" (rounded)");
    say(", median ");
    if (n & 1) say_fixed(x[o[n / 2]].v, x[o[n / 2]].frac);
    else {
        sbig a, b, m;
        as_sbig(&a, x[o[n / 2 - 1]], S);
        as_sbig(&b, x[o[n / 2]], S);
        sb_add(&m, &a, &b);
        big_mul_small(&m.m, &m.m, 5, 0);
        say_sb(&m, S + 1, 10);
    }
    int best = 1, at = 0, run = 1, ties = 0; /* the mode: the longest run of equal values in order */
    for (int i = 1; i <= n; i++) {
        if (i < n && !num_cmp(x[o[i]], x[o[i - 1]])) {
            run++;
            continue;
        }
        if (run > best) best = run, at = i - run, ties = 0;
        else if (run == best && best > 1) ties++;
        run = 1;
    }
    say(", mode ");
    if (best == 1) say("none (each value once)");
    else {
        say_fixed(x[o[at]].v, x[o[at]].frac);
        if (ties) say(" (and others as often)");
    }
    say(", min ");
    say_fixed(x[o[0]].v, x[o[0]].frac);
    say(", max ");
    say_fixed(x[o[n - 1]].v, x[o[n - 1]].frac);
    sbig lo, hi, range;
    as_sbig(&lo, x[o[0]], S);
    as_sbig(&hi, x[o[n - 1]], S);
    lo.neg = !lo.neg && lo.m.n;
    sb_add(&range, &hi, &lo);
    say(", range ");
    say_sb(&range, S, 9);
    if (n < 2) return;
    /* standard deviations: D = n sum(x^2) - (sum x)^2; population sqrt(D) / n, sample sqrt(D n / (n - 1)) / n */
    big s2, d, p12, a, q, rem, den;
    big_mul_small(&sq, &sq, (u32)n, 0);
    big_mul(&s2, &sum.m, &sum.m);
    big_sub(&d, &sq, &s2);
    big_pow10(&p12, 14); /* (7 decimals, then rounded to 6) */
    big_mul(&a, &d, &p12);
    big_set(&den, (u64)n);
    big_pow10(&p12, S);
    big_mul(&den, &den, &p12);
    for (int k = 0; k < 2; k++) {
        big v = a;
        if (k) {
            big_mul_small(&v, &v, (u32)n, 0);
            big tmp;
            big_div_small(&tmp, &v, (u32)(n - 1));
            v = tmp;
        }
        big root;
        big_isqrt(&root, &v);
        big_divmod(&q, &rem, &root, &den);
        say(k ? ", " : ", standard deviation ");
        say_decimal(&q, 0, 7, 6);
        say(k ? " (sample)" : " (population)");
    }
}

/* ---- money and percentages */
static void rescale(sbig *a, int from, int to) {
    if (to > from) {
        big p;
        big_pow10(&p, to - from);
        big_mul(&a->m, &a->m, &p);
    }
}

static void read_arg(const char *name, sbig *v, int *scale) {
    str s = param(name);
    if (!have(s)) {
        say("error (missing argument: ");
        say(name);
        put(')');
        done();
    }
    if (!read_decimal(s, &v->m, &v->neg, scale)) {
        say("error (");
        say(name);
        say(" is not a number: ");
        say_str(trim(s));
        put(')');
        done();
    }
    if (*scale > 18) fail("a number with more than 18 decimals");
    if (!v->m.n) v->neg = 0;
}

void percent_change(void) {
    sbig a, b;
    int sa, sb_;
    read_arg("old", &a, &sa);
    read_arg("new", &b, &sb_);
    int S = sa > sb_ ? sa : sb_;
    rescale(&a, sa, S), rescale(&b, sb_, S);
    if (!a.m.n) fail("a change from 0 is no percentage");
    sbig d, na = a;
    na.neg = !na.neg && na.m.n;
    sb_add(&d, &b, &na);
    /* (new - old) / |old| * 100, to 6 decimals */
    big t, p, q, rem;
    big_pow10(&p, 8);
    big_mul(&t, &d.m, &p);
    big_divmod(&q, &rem, &t, &a.m);
    big_mul_small(&rem, &rem, 2, 0);
    if (big_cmp(&rem, &a.m) >= 0) {
        big one;
        big_set(&one, 1);
        big_add(&q, &q, &one);
    }
    int up = d.m.n && !d.neg;
    say(d.m.n ? (up ? "+" : "-") : "");
    say_decimal(&q, 0, 6, 6);
    say("% (");
    say(d.m.n ? (up ? "up " : "down ") : "no change");
    if (d.m.n) say_decimal(&d.m, 0, S, 18);
    say(", from ");
    say_sb(&a, S, 18);
    say(" to ");
    say_sb(&b, S, 18);
    put(')');
}

void round_number(void) {
    sbig v;
    int s;
    read_arg("number", &v, &s);
    str pl = trim(param("places"));
    num places = {0, 0};
    if (have(pl) && pl.p < pl.e && (!read_num(pl, &places, 0) || places.frac || places.v > 18 || places.v < -18))
        fail("places: a whole number, e.g. 2 (or -2: to hundreds)");
    int k = (int)places.v;
    /* half away from zero (as money is rounded): to k decimals (k < 0: to 10^-k) */
    if (s > k) {
        big p, q, rem;
        big_pow10(&p, s - k);
        big_divmod(&q, &rem, &v.m, &p);
        big_mul_small(&rem, &rem, 2, 0);
        if (big_cmp(&rem, &p) >= 0) {
            big one;
            big_set(&one, 1);
            big_add(&q, &q, &one);
        }
        if (k < 0) {
            big_pow10(&p, -k);
            big_mul(&q, &q, &p);
            k = 0;
        }
        v.m = q;
        s = k;
    }
    if (!v.m.n) v.neg = 0;
    say_decimal(&v.m, v.neg, s, 18);
    if (s < k && k > 0) { /* (to as many decimals as asked: 2.5 -> 2.50) */
        int have_point = s > 0;
        if (!have_point) put('.');
        for (int i = s; i < k; i++) put('0');
    }
}

/* (1 + r / (100 n)) as a fixed-point number of 20 decimals, and its power */
#define FX 20

static void say_money(const big *a, int neg, int scale) { /* to the cent, with both its digits */
    big t = *a;
    if (scale > 2) {
        big p, q, r;
        big_pow10(&p, scale - 2);
        big_divmod(&q, &r, &t, &p);
        big_mul_small(&r, &r, 2, 0);
        if (big_cmp(&r, &p) >= 0) {
            big one;
            big_set(&one, 1);
            big_add(&q, &q, &one);
        }
        t = q;
    } else if (scale < 2) {
        big p;
        big_pow10(&p, 2 - scale);
        big_mul(&t, &t, &p);
    }
    char d[160];
    int n = big_digits(&t, d);
    if (neg && t.n) put('-');
    if (n < 3) {
        put('0');
        put('.');
        for (int i = n; i < 2; i++) put('0');
        for (int i = 0; i < n; i++) put(d[i]);
        return;
    }
    for (int i = 0; i < n; i++) {
        if (i == n - 2) put('.');
        put(d[i]);
    }
}
static void fx_mul(big *r, const big *a, const big *b) { /* a b / 10^20, rounded */
    big t;
    big_mul(&t, a, b);
    u32 m1 = big_div_small(&t, &t, 1000000000u);
    u32 m2 = big_div_small(&t, &t, 1000000000u);
    u32 m3 = big_div_small(&t, &t, 100u);
    (void)m1;
    if (m3 >= 50 || (m3 == 49 && m2 >= 500000000u)) {
        big one;
        big_set(&one, 1);
        big_add(&t, &t, &one);
    }
    *r = t;
}

static void fx_pow(big *r, const big *a, u64 e) {
    big x = *a;
    big_pow10(r, FX);
    while (e) {
        if (e & 1) fx_mul(r, r, &x);
        e >>= 1;
        if (e) fx_mul(&x, &x, &x);
    }
}

static void growth(big *f, const sbig *rate, int rs, u32 per) { /* 1 + rate / (100 per), 20 decimals */
    big t, p;
    big_pow10(&p, FX - rs);
    big_mul(&t, &rate->m, &p); /* rate * 10^20 / 10^rs */
    big q;
    u32 m = big_div_small(&q, &t, 100u * per);
    if (2ull * m >= 100u * per) {
        big one;
        big_set(&one, 1);
        big_add(&q, &q, &one);
    }
    big_pow10(f, FX);
    if (rate->neg) big_sub(f, f, &q);
    else big_add(f, f, &q);
}

static u64 periods(const char *name, u32 per) { /* years * per, which must be whole */
    sbig y;
    int s;
    read_arg(name, &y, &s);
    if (y.neg) fail("a negative number of years");
    big t, p, q, rem;
    big_mul_small(&t, &y.m, per, 0);
    big_pow10(&p, s);
    big_divmod(&q, &rem, &t, &p);
    if (rem.n) fail("the years make no whole number of periods");
    if (q.n > 1 || (q.n && q.d[0] > 100000)) fail("more than 100,000 periods");
    return q.n ? q.d[0] : 0;
}

static u32 per_year(void) {
    str c = trim(param("compounding"));
    if (!have(c) || c.p == c.e) c = trim(param("times_per_year"));
    if (!have(c) || c.p == c.e) return 12;
    static const struct {
        const char *z;
        u32 n;
    } K[] = {{"yearly", 1}, {"annually", 1}, {"annual", 1}, {"semiannually", 2}, {"quarterly", 4}, {"monthly", 12},
             {"weekly", 52}, {"daily", 365}};
    for (size_t i = 0; i < sizeof K / sizeof K[0]; i++)
        if (str_ieq(c, (str){K[i].z, K[i].z + strlen(K[i].z)})) return K[i].n;
    num v;
    if (read_num(c, &v, 0) && !v.frac && v.v >= 1 && v.v <= 366) return (u32)v.v;
    fail("compounding: yearly, quarterly, monthly, daily, or times a year (e.g. 12)");
}

void compound_interest(void) {
    sbig p, r;
    int ps, rs;
    read_arg("principal", &p, &ps);
    read_arg("rate", &r, &rs);
    u32 per = per_year();
    u64 n = periods("years", per);
    big f, fn, a;
    growth(&f, &r, rs, per);
    fx_pow(&fn, &f, n);
    big_mul(&a, &p.m, &fn); /* scale ps + 20 */
    say("amount ");
    say_money(&a, p.neg, ps + FX);
    big pr, pa, i;
    big_pow10(&pa, FX);
    big_mul(&pr, &p.m, &pa);
    int gain = big_cmp(&a, &pr) >= 0;
    if (gain) big_sub(&i, &a, &pr);
    else big_sub(&i, &pr, &a);
    say(gain ? ", interest " : ", loss ");
    say_money(&i, 0, ps + FX);
    say(" (");
    say_sb(&r, rs, 18);
    say("% a year compounded ");
    say(per == 1 ? "yearly" : per == 2 ? "twice a year" : per == 4 ? "quarterly" : per == 12 ? "monthly" : per == 52 ? "weekly" : per == 365 ? "daily" : "");
    if (per != 1 && per != 2 && per != 4 && per != 12 && per != 52 && per != 365) say_int(per), say(" times a year");
    say(", ");
    say_int((i64)n);
    say(n == 1 ? " period)" : " periods)");
}

void loan_payment(void) {
    sbig p, r;
    int ps, rs;
    read_arg("principal", &p, &ps);
    read_arg("rate", &r, &rs);
    u32 per = 12;
    str c = trim(param("payments_per_year"));
    if (have(c) && c.p < c.e) per = per_year();
    u64 n = periods("years", per);
    if (!n) fail("no payments");
    big pay; /* scale ps + 20 */
    if (!r.m.n) {
        big t, q;
        big_pow10(&t, FX);
        big_mul(&t, &p.m, &t);
        big_set(&q, n);
        big rem;
        big_divmod(&pay, &rem, &t, &q);
    } else { /* P i F / (F - 1), F = (1 + i)^n, i = r / (100 per) */
        big f, fn, i, one, num_, den, rem, t;
        growth(&f, &r, rs, per);
        big_pow10(&one, FX);
        big_sub(&i, &f, &one);
        fx_pow(&fn, &f, n);
        big_mul(&num_, &p.m, &i);
        big_mul(&num_, &num_, &fn); /* scale ps + 40 */
        big_sub(&den, &fn, &one);   /* scale 20 */
        big_divmod(&t, &rem, &num_, &den); /* scale ps + 20 */
        pay = t;
    }
    say(per == 12 ? "monthly payment " : "payment ");
    say_money(&pay, 0, ps + FX);
    say(" for ");
    say_int((i64)n);
    say(" payments: total ");
    /* (each payment rounded to cents, as paid) */
    big cents, p100, q, rem, tot, prin;
    big_pow10(&p100, ps + FX - 2);
    big_divmod(&cents, &rem, &pay, &p100);
    big_mul_small(&rem, &rem, 2, 0);
    if (big_cmp(&rem, &p100) >= 0) {
        big o;
        big_set(&o, 1);
        big_add(&cents, &cents, &o);
    }
    big nn;
    big_set(&nn, n);
    big_mul(&tot, &cents, &nn);
    say_money(&tot, 0, 2);
    say(", interest ");
    big pp;
    big_pow10(&pp, 2 > ps ? 2 - ps : 0);
    big_mul(&prin, &p.m, &pp);
    if (ps > 2) {
        big_pow10(&pp, ps - 2);
        big_divmod(&prin, &q, &p.m, &pp);
    }
    if (big_cmp(&tot, &prin) >= 0) big_sub(&tot, &tot, &prin);
    else big_set(&tot, 0);
    say_money(&tot, 0, 2);
}

/* ---- primes: 64-bit Montgomery arithmetic, Miller-Rabin (deterministic below 2^64), Pollard-Brent */
static u64 mul_hi(u64 a, u64 b) {
    u64 a0 = (u32)a, a1 = a >> 32, b0 = (u32)b, b1 = b >> 32;
    u64 p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    u64 mid = (p00 >> 32) + (u32)p01 + (u32)p10;
    return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

typedef struct {
    u64 n, ninv, r2; /* n odd; -n^-1 mod 2^64; 2^128 mod n */
} mont;

static u64 redc(const mont *m, u64 hi, u64 lo) { /* (hi 2^64 + lo) / 2^64 mod n */
    u64 q = lo * m->ninv, h = mul_hi(q, m->n), l = q * m->n;
    u64 carry = (lo + l) < lo;
    u64 t = hi + h + carry;
    if (t < hi || t >= m->n) t -= m->n;
    return t;
}

static u64 mmul(const mont *m, u64 a, u64 b) { return redc(m, mul_hi(a, b), a * b); }

static u64 mod64(u64 a, u64 n) { return a % n; }

static void mont_make(mont *m, u64 n) {
    m->n = n;
    u64 inv = n; /* (Newton: n^-1 mod 2^64) */
    for (int i = 0; i < 5; i++) inv *= 2 - n * inv;
    m->ninv = -inv;
    u64 r = mod64(-n, n); /* 2^64 mod n */
    u64 r2 = r;
    for (int i = 0; i < 64; i++) { /* (r2 = 2^128 mod n, by doubling) */
        r2 = r2 >= n - r2 ? r2 - (n - r2) : r2 + r2;
    }
    m->r2 = r2;
}

static u64 to_m(const mont *m, u64 a) { return mmul(m, a % m->n, m->r2); }

static int mr_composite(const mont *m, u64 a, u64 d, int s) {
    u64 one = to_m(m, 1), minus = to_m(m, m->n - 1), x = to_m(m, a), r = one;
    for (u64 e = d; e; e >>= 1) {
        if (e & 1) r = mmul(m, r, x);
        x = mmul(m, x, x);
    }
    if (r == one || r == minus) return 0;
    for (int i = 1; i < s; i++) {
        r = mmul(m, r, r);
        if (r == minus) return 0;
    }
    return 1;
}

static int prime64(u64 n) {
    if (n < 2) return 0;
    static const u32 SMALL[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (int i = 0; i < 12; i++) {
        if (n == SMALL[i]) return 1;
        if (n % SMALL[i] == 0) return 0;
    }
    mont m;
    mont_make(&m, n);
    u64 d = n - 1;
    int s = 0;
    while (!(d & 1)) d >>= 1, s++;
    /* deterministic witnesses: 3 below 4,759,123,141; 4 below 3,474,749,660,383 (2, 3, 5, 7, 11, 13); else 7 */
    static const u64 B32[] = {2, 7, 61}, B42[] = {2, 3, 5, 7, 11, 13}, B64[] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
    const u64 *b = n < 4759123141ULL ? B32 : n < 3474749660383ULL ? B42 : B64;
    int k = n < 4759123141ULL ? 3 : n < 3474749660383ULL ? 6 : 7;
    for (int i = 0; i < k; i++) {
        u64 a = b[i] % n;
        if (a && mr_composite(&m, a, d, s)) return 0;
    }
    return 1;
}

static u64 gcd64(u64 a, u64 b) { /* (binary: shifts, no 64-bit division, which the computer does a bit at a time) */
    if (!a || !b) return a | b;
    int k = __builtin_ctzll(a | b);
    a >>= __builtin_ctzll(a);
    while (b) {
        b >>= __builtin_ctzll(b);
        if (a > b) {
            u64 t = a;
            a = b, b = t;
        }
        b -= a;
    }
    return a << k;
}

static u64 rho(u64 n, u64 c, int *budget) { /* a factor of the odd composite n, or 0 (each round counted) */
    mont m;
    mont_make(&m, n);
    u64 y = to_m(&m, 2), cc = to_m(&m, c), g = 1, q = to_m(&m, 1), x = y, ys = y;
    for (u64 r = 1; g == 1; r <<= 1) {
        x = y;
        for (u64 i = 0; i < r; i++) y = mmul(&m, y, y) + cc, y = y >= n ? y - n : y;
        if ((*budget -= (int)r) < 0) return 0;
        for (u64 k = 0; k < r && g == 1; k += 64) {
            ys = y;
            u64 steps = r - k < 64 ? r - k : 64;
            for (u64 i = 0; i < steps; i++) {
                y = mmul(&m, y, y) + cc, y = y >= n ? y - n : y;
                q = mmul(&m, q, x > y ? x - y : y - x);
            }
            g = gcd64(q, n);
            if ((*budget -= (int)steps) < 0) return 0;
        }
    }
    if (g == n)
        do {
            ys = mmul(&m, ys, ys) + cc, ys = ys >= n ? ys - n : ys;
            g = gcd64(x > ys ? x - ys : ys - x, n);
            if (--*budget < 0) return 0;
        } while (g == 1);
    return g == n ? 0 : g;
}

static u64 whole_arg(const char *name) {
    str s = trim(param(name));
    if (!have(s)) {
        say("error (missing argument: ");
        say(name);
        put(')');
        done();
    }
    big b;
    int neg, scale;
    if (!read_decimal(s, &b, &neg, &scale) || scale || neg) fail("a whole number, 0 or more");
    if (b.n > 2) fail("a number of more than 19 digits (the computer's 64 bits)");
    return b.n == 2 ? (u64)b.d[1] << 32 | b.d[0] : b.n ? b.d[0] : 0;
}

void is_prime(void) {
    u64 n = whole_arg("number");
    say_u(n);
    if (prime64(n)) {
        say(" is prime");
        return;
    }
    say(n < 2 ? " is not prime" : " is not prime");
    if (n >= 4) {
        u64 d = 0;
        static const u32 P[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
        for (int i = 0; i < 15 && !d; i++)
            if (n % P[i] == 0) d = P[i];
        if (d) say(": divisible by "), say_u(d);
    }
}

/* n's prime factors into out (k so far); a composite it could not split within the budget goes in as it is,
   marked in its bit 63 (none is above 2^63) */
static int factor_rec(u64 n, u64 *out, int k, int *budget) {
    if (n == 1) return k;
    if (prime64(n)) {
        out[k++] = n;
        return k;
    }
    for (u64 c = 1; c < 20 && *budget > 0; c++) {
        u64 d = rho(n, c, budget);
        if (d) {
            k = factor_rec(d, out, k, budget);
            return factor_rec(n / d, out, k, budget);
        }
    }
    out[k++] = n | (1ULL << 63);
    return k;
}

void factor(void) {
    u64 n = whole_arg("number"), f[70];
    if (n < 2) {
        say_u(n);
        say(" has no prime factors");
        return;
    }
    if (n >> 63) fail("a number below 2^63");
    int k = 0;
    u64 m = n;
    for (u64 p = 2; p < 1000 && p * p <= m; p += p == 2 ? 1 : 2)
        while (m % p == 0) f[k++] = p, m /= p;
    /* rounds of Pollard-Brent, about 280 trace tokens each, after a primality test of 45,000 (below 2^32) to 100,000
       (above 3.4 * 10^12): the run inside its limit; most factors to 100,000 below 2^32, to about 20,000 above */
    int budget = m < 4294967296ULL ? 900 : m < 3474749660383ULL ? 420 : 350;
    if (m > 1) k = factor_rec(m, f, k, &budget);
    for (int i = 1; i < k; i++) /* (in order) */
        for (int j = i; j > 0 && (f[j - 1] & ~(1ULL << 63)) > (f[j] & ~(1ULL << 63)); j--) {
            u64 t = f[j];
            f[j] = f[j - 1], f[j - 1] = t;
        }
    int left = 0;
    say_u(n);
    say(" = ");
    for (int i = 0; i < k;) {
        int e = 1;
        while (i + e < k && f[i + e] == f[i]) e++;
        if (i) say(" * ");
        say_u(f[i] & ~(1ULL << 63));
        left |= f[i] >> 63;
        if (e > 1) put('^'), say_int(e);
        i += e;
    }
    if (k == 1 && !left) say(" (prime)");
    if (left) say(" (the last not prime, but with no factor the computer found in its run: its factors are most likely above 20,000)");
}

/* ---- units: each a factor to its dimension's base (as num / den), or a temperature's scale */
enum { D_LEN, D_MASS, D_VOL, D_AREA, D_TIME, D_SPEED, D_DATA, D_ENERGY, D_PRESS, D_TEMP };
static const char *const DIMS[] = {"length", "mass", "volume", "area", "time", "speed", "data", "energy", "pressure", "temperature"};
static const struct {
    const char *names; /* its names, separated by | */
    u8 dim;
    u64 num, den;
} UNITS[] = {
    {"mm|millimeter|millimeters|millimetre|millimetres", D_LEN, 1, 1000},
    {"cm|centimeter|centimeters|centimetre|centimetres", D_LEN, 1, 100},
    {"m|meter|meters|metre|metres", D_LEN, 1, 1},
    {"km|kilometer|kilometers|kilometre|kilometres", D_LEN, 1000, 1},
    {"um|micrometer|micrometers|micron|microns", D_LEN, 1, 1000000},
    {"nm|nanometer|nanometers", D_LEN, 1, 1000000000},
    {"in|inch|inches|\"", D_LEN, 254, 10000},
    {"ft|foot|feet|'", D_LEN, 3048, 10000},
    {"yd|yard|yards", D_LEN, 9144, 10000},
    {"mi|mile|miles", D_LEN, 1609344, 1000},
    {"nmi|nautical mile|nautical miles", D_LEN, 1852, 1},
    {"mg|milligram|milligrams", D_MASS, 1, 1000},
    {"g|gram|grams", D_MASS, 1, 1},
    {"kg|kilogram|kilograms|kilo|kilos", D_MASS, 1000, 1},
    {"t|tonne|tonnes|metric ton|metric tons", D_MASS, 1000000, 1},
    {"oz|ounce|ounces", D_MASS, 28349523125ULL, 1000000000ULL},
    {"lb|lbs|pound|pounds", D_MASS, 45359237, 100000},
    {"st|stone|stones", D_MASS, 635029318, 100000},
    {"mL|ml|milliliter|milliliters|millilitre|millilitres", D_VOL, 1, 1},
    {"cL|cl|centiliter|centiliters", D_VOL, 10, 1},
    {"dL|dl|deciliter|deciliters", D_VOL, 100, 1},
    {"L|l|liter|liters|litre|litres", D_VOL, 1000, 1},
    {"m³|m3|cubic meter|cubic meters", D_VOL, 1000000, 1},
    {"tsp|teaspoon|teaspoons", D_VOL, 492892159375ULL, 100000000000ULL},
    {"tbsp|tablespoon|tablespoons", D_VOL, 1478676478125ULL, 100000000000ULL},
    {"fl oz|floz|fluid ounce|fluid ounces", D_VOL, 295735295625ULL, 10000000000ULL},
    {"cups|cup", D_VOL, 2365882365ULL, 10000000},
    {"pt|pint|pints", D_VOL, 473176473, 1000000},
    {"qt|quart|quarts", D_VOL, 946352946, 1000000},
    {"gal|gallon|gallons|us gallon|us gallons", D_VOL, 3785411784ULL, 1000000},
    {"imperial gallon|imperial gallons|uk gallon|uk gallons", D_VOL, 454609, 100},
    {"mm²|mm2|square millimeter|square millimeters", D_AREA, 1, 1000000},
    {"cm²|cm2|square centimeter|square centimeters", D_AREA, 1, 10000},
    {"m²|m2|square meter|square meters|square metre|square metres", D_AREA, 1, 1},
    {"km²|km2|square kilometer|square kilometers", D_AREA, 1000000, 1},
    {"ha|hectare|hectares", D_AREA, 10000, 1},
    {"acres|acre", D_AREA, 40468564224ULL, 10000000},
    {"ft²|ft2|sq ft|square foot|square feet", D_AREA, 9290304, 100000000},
    {"in²|in2|sq in|square inch|square inches", D_AREA, 64516, 100000000},
    {"yd²|yd2|square yard|square yards", D_AREA, 83612736, 100000000},
    {"mi²|mi2|square mile|square miles", D_AREA, 2589988110336ULL, 1000000},
    {"ms|millisecond|milliseconds", D_TIME, 1, 1000},
    {"s|sec|second|seconds", D_TIME, 1, 1},
    {"min|minute|minutes", D_TIME, 60, 1},
    {"h|hr|hour|hours", D_TIME, 3600, 1},
    {"days|day", D_TIME, 86400, 1},
    {"weeks|week", D_TIME, 604800, 1},
    {"m/s|meters per second|metres per second", D_SPEED, 1, 1},
    {"km/h|kph|kmh|kilometers per hour|kilometres per hour", D_SPEED, 5, 18},
    {"mph|miles per hour", D_SPEED, 44704, 100000},
    {"knots|knot|kn", D_SPEED, 463, 900},
    {"ft/s|feet per second", D_SPEED, 3048, 10000},
    {"bits|bit", D_DATA, 1, 8},
    {"B|b|byte|bytes", D_DATA, 1, 1},
    {"KB|kb|kilobyte|kilobytes", D_DATA, 1000, 1},
    {"MB|mb|megabyte|megabytes", D_DATA, 1000000, 1},
    {"GB|gb|gigabyte|gigabytes", D_DATA, 1000000000, 1},
    {"TB|tb|terabyte|terabytes", D_DATA, 1000000000000ULL, 1},
    {"KiB|kib|kibibyte|kibibytes", D_DATA, 1024, 1},
    {"MiB|mib|mebibyte|mebibytes", D_DATA, 1048576, 1},
    {"GiB|gib|gibibyte|gibibytes", D_DATA, 1073741824, 1},
    {"TiB|tib|tebibyte|tebibytes", D_DATA, 1099511627776ULL, 1},
    {"kbit|kilobit|kilobits", D_DATA, 125, 1},
    {"mbit|megabit|megabits", D_DATA, 125000, 1},
    {"gbit|gigabit|gigabits", D_DATA, 125000000, 1},
    {"J|j|joule|joules", D_ENERGY, 1, 1},
    {"kJ|kj|kilojoule|kilojoules", D_ENERGY, 1000, 1},
    {"cal|calorie|calories", D_ENERGY, 4184, 1000},
    {"kcal|kilocalorie|kilocalories|food calorie|food calories", D_ENERGY, 4184, 1},
    {"Wh|wh|watt hour|watt hours", D_ENERGY, 3600, 1},
    {"kWh|kwh|kilowatt hour|kilowatt hours", D_ENERGY, 3600000, 1},
    {"BTU|btu|btus", D_ENERGY, 105505585262ULL, 100000000},
    {"Pa|pa|pascal|pascals", D_PRESS, 1, 1},
    {"kPa|kpa|kilopascal|kilopascals", D_PRESS, 1000, 1},
    {"MPa|mpa|megapascal|megapascals", D_PRESS, 1000000, 1},
    {"bar|bars", D_PRESS, 100000, 1},
    {"mbar|millibar|millibars", D_PRESS, 100, 1},
    {"atm|atmosphere|atmospheres", D_PRESS, 101325, 1},
    {"psi", D_PRESS, 6894757293168361ULL, 1000000000000ULL},
    {"mmHg|mmhg|torr", D_PRESS, 101325, 760},
    {"°C|c|celsius|degrees celsius|°c", D_TEMP, 0, 0},
    {"°F|f|fahrenheit|degrees fahrenheit|°f", D_TEMP, 1, 0},
    {"K|k|kelvin|kelvins", D_TEMP, 2, 0},
};
#define N_UNITS (int)(sizeof UNITS / sizeof UNITS[0])

static int unit_of(str u) {
    u = trim(u);
    if (u.e - u.p > 4 && lower((u8)u.e[-1]) == '.' ) u.e--;
    for (int i = 0; i < N_UNITS; i++) {
        const char *z = UNITS[i].names;
        while (*z) {
            const char *q = z;
            while (*q && *q != '|') q++;
            str name = {z, q};
            if (str_ieq(name, u)) return i;
            z = *q ? q + 1 : q;
        }
    }
    return -1;
}

static void say_unit_name(int i) {
    const char *z = UNITS[i].names, *q = z;
    while (*q && *q != '|') q++;
    say_str((str){z, q});
}

void convert_units(void) {
    sbig v;
    int s;
    read_arg("value", &v, &s);
    int a = unit_of(param("from")), b = unit_of(param("to"));
    if (a < 0 || b < 0) {
        say("error (a unit the computer does not know: ");
        say_str(trim(param(a < 0 ? "from" : "to")));
        say("; it knows lengths, masses, volumes, areas, times, speeds, data sizes, energies, pressures and temperatures)");
        done();
    }
    if (UNITS[a].dim != UNITS[b].dim) {
        say("error (");
        say_unit_name(a);
        say(" is a ");
        say(DIMS[UNITS[a].dim]);
        say(" and ");
        say_unit_name(b);
        say(" a ");
        say(DIMS[UNITS[b].dim]);
        put(')');
        done();
    }
    /* the result as num / den, then as a decimal of 12 significant digits */
    big num_ = v.m, den;
    int neg = v.neg;
    big_pow10(&den, s);
    if (UNITS[a].dim == D_TEMP) { /* kelvin as k / kd, from C (x + 273.15), F ((x - 32) 5/9 + 273.15) or K */
        int ta = (int)UNITS[a].num, tb = (int)UNITS[b].num;
        static const u32 MUL[3] = {100, 500, 1}, ADD[3] = {27315, 229835, 0}, DEN[3] = {100, 900, 1};
        sbig k = v, off;
        big p, kd;
        big_pow10(&p, s);
        big_mul_small(&k.m, &k.m, MUL[ta], 0);
        big_set(&off.m, ADD[ta]);
        big_mul(&off.m, &off.m, &p);
        off.neg = 0;
        sb_add(&k, &k, &off);
        big_mul_small(&kd, &p, DEN[ta], 0);
        /* to C: (100 k - 27315 kd) / (100 kd); F: (900 k - 229835 kd) / (500 kd); K: k / kd */
        static const u32 TM[3] = {100, 900, 1}, TS[3] = {27315, 229835, 0}, TD[3] = {100, 500, 1};
        big_mul_small(&k.m, &k.m, TM[tb], 0);
        big_mul_small(&off.m, &kd, TS[tb], 0);
        off.neg = 1;
        if (off.m.n) sb_add(&k, &k, &off);
        big_mul_small(&den, &kd, TD[tb], 0);
        num_ = k.m, neg = k.neg;
    } else {
        big t;
        big_set(&t, UNITS[a].num);
        big_mul(&num_, &num_, &t);
        big_set(&t, UNITS[b].den);
        big_mul(&num_, &num_, &t);
        big_set(&t, UNITS[a].den);
        big_mul(&den, &den, &t);
        big_set(&t, UNITS[b].num);
        big_mul(&den, &den, &t);
    }
    /* num / den to as many decimals as make 12 significant digits (at least 2, at most 15) */
    big q, rem, p;
    big_divmod(&q, &rem, &num_, &den);
    char d[160];
    int whole = q.n ? big_digits(&q, d) : 0, frac = 12 - whole;
    if (frac < 2) frac = 2;
    if (frac > 15) frac = 15;
    big_pow10(&p, frac);
    big_mul(&num_, &num_, &p);
    big_divmod(&q, &rem, &num_, &den);
    int exact = !rem.n;
    big_mul_small(&rem, &rem, 2, 0);
    if (big_cmp(&rem, &den) >= 0) {
        big one;
        big_set(&one, 1);
        big_add(&q, &q, &one);
    }
    say_sb(&v, s, 18);
    put(' ');
    say_unit_name(a);
    say(" = ");
    say_decimal(&q, neg, frac, frac);
    put(' ');
    say_unit_name(b);
    if (!exact) say(" (rounded)");
}

/* ---- roman numerals, both ways; numbers in words */
void roman_numeral(void) {
    str s = trim(param("value"));
    if (!have(s) || s.p == s.e) fail("missing argument: value, a number (1 to 3999) or a roman numeral");
    static const struct {
        const char *z;
        int v;
    } R[] = {{"M", 1000}, {"CM", 900}, {"D", 500}, {"CD", 400}, {"C", 100}, {"XC", 90}, {"L", 50}, {"XL", 40},
             {"X", 10}, {"IX", 9}, {"V", 5}, {"IV", 4}, {"I", 1}};
    num n;
    if (read_num(s, &n, 1) && !n.frac) {
        if (n.v < 1 || n.v > 3999) fail("roman numerals go from 1 to 3999");
        i64 x = n.v;
        for (int i = 0; i < 13; i++)
            while (x >= R[i].v) say(R[i].z), x -= R[i].v;
        return;
    }
    /* a numeral: its value, if it is written the standard way (the same as the value's numeral) */
    int v = 0;
    const char *p = s.p;
    for (int i = 0; i < 13 && p < s.e; i++) {
        int k = (int)strlen(R[i].z), reps = 0;
        while (p + k <= s.e && lower((u8)p[0]) == lower((u8)R[i].z[0]) && (k == 1 || lower((u8)p[1]) == lower((u8)R[i].z[1]))) {
            v += R[i].v, p += k, reps++;
            if (k == 2 || reps == 3) break;
        }
    }
    /* (and the value's own numeral is the one written: IXI, VIV or XCX are not) */
    int ok = p == s.e && v > 0 && v <= 3999;
    const char *q = s.p;
    for (int i = 0, x = v; i < 13 && ok; i++)
        for (; x >= R[i].v && ok; x -= R[i].v)
            for (const char *z = R[i].z; *z && ok; z++, q++) ok = q < s.e && lower((u8)*q) == lower((u8)*z);
    if (!ok || q != s.e) fail("not a roman numeral written the standard way (I V X L C D M)");
    say_int(v);
}

static const char *const ONES[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                   "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
                                   "nineteen"};
static const char *const TENS[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

static void say_hundreds(int v) { /* 1..999 */
    if (v >= 100) {
        say(ONES[v / 100]);
        say(" hundred");
        v %= 100;
        if (v) put(' ');
    }
    if (v >= 20) {
        say(TENS[v / 10]);
        if (v % 10) put('-'), say(ONES[v % 10]);
    } else if (v)
        say(ONES[v]);
}

void number_to_words(void) {
    big b;
    int neg, scale;
    str s = trim(param("number"));
    if (!have(s) || !read_decimal(s, &b, &neg, &scale)) fail("number: a number such as 1234 or -5.25");
    big whole, frac, p;
    big_pow10(&p, scale);
    big_divmod(&whole, &frac, &b, &p);
    u64 w = whole.n == 2 ? (u64)whole.d[1] << 32 | whole.d[0] : whole.n ? whole.d[0] : 0;
    if (whole.n > 2 || w >= 1000000000000000000ULL) fail("numbers below a quintillion (10^18)");
    if (neg && (w || frac.n)) say("minus ");
    if (!w) say("zero");
    static const char *const BIG[] = {"", " thousand", " million", " billion", " trillion", " quadrillion"};
    int groups[6], n = 0;
    for (u64 t = w; t; t /= 1000) groups[n++] = (int)(t % 1000);
    int first = 1;
    for (int i = n - 1; i >= 0; i--) {
        if (!groups[i]) continue;
        if (!first) put(' ');
        say_hundreds(groups[i]);
        say(BIG[i]);
        first = 0;
    }
    if (scale) { /* point, then each digit */
        say(" point");
        char d[160];
        int k = big_digits(&frac, d);
        for (int i = 0; i < scale - k; i++) say(" zero");
        for (int i = 0; i < k && frac.n; i++) put(' '), say(ONES[d[i] - '0']);
    }
}
