/* Versions and lists: compare_versions, sort_versions, compare_lists. */
#include "tools.h"

/* ---- versions: v1.2.3-rc.1+build. Dot-separated parts compared as numbers (a missing one is 0), a part's letters
   after its number before it alone (1.0rc1 < 1.0); a pre-release (after -) before the release, its parts as
   semver compares them; build metadata (after +) left out. */
static str clean(str v) {
    v = trim(v);
    if (v.p < v.e && (*v.p == 'v' || *v.p == 'V')) v.p++;
    for (const char *q = v.p; q < v.e; q++)
        if (*q == '+') v.e = q;
    return v;
}

static int is_num(str s) {
    if (s.p == s.e) return 0;
    for (const char *q = s.p; q < s.e; q++)
        if (!digit(*q)) return 0;
    return 1;
}

static int num_part_cmp(str a, str b) { /* digits, as numbers of any length */
    while (a.p < a.e - 1 && *a.p == '0') a.p++;
    while (b.p < b.e - 1 && *b.p == '0') b.p++;
    if (a.e - a.p != b.e - b.p) return a.e - a.p < b.e - b.p ? -1 : 1;
    for (; a.p < a.e; a.p++, b.p++)
        if (*a.p != *b.p) return *a.p < *b.p ? -1 : 1;
    return 0;
}

static int ascii_cmp(str a, str b) {
    for (; a.p < a.e && b.p < b.e; a.p++, b.p++)
        if (*a.p != *b.p) return (u8)*a.p < (u8)*b.p ? -1 : 1;
    return (a.p < a.e) - (b.p < b.e);
}

static str next_part(str *s, char sep) { /* the part of *s up to sep, and *s after it */
    const char *q = s->p;
    while (q < s->e && *q != sep) q++;
    str part = {s->p, q};
    s->p = q < s->e ? q + 1 : q;
    return part;
}

static int version_cmp(str a, str b) {
    a = clean(a), b = clean(b);
    str ra = a, rb = b, pa = {0, 0}, pb = {0, 0};
    for (const char *q = a.p; q < a.e; q++)
        if (*q == '-') {
            ra.e = q, pa = (str){q + 1, a.e};
            break;
        }
    for (const char *q = b.p; q < b.e; q++)
        if (*q == '-') {
            rb.e = q, pb = (str){q + 1, b.e};
            break;
        }
    while (ra.p < ra.e || rb.p < rb.e) {
        str x = next_part(&ra, '.'), y = next_part(&rb, '.');
        str xn = x, yn = y; /* each part's number, then its letters */
        xn.e = xn.p, yn.e = yn.p;
        while (xn.e < x.e && digit(*xn.e)) xn.e++;
        while (yn.e < y.e && digit(*yn.e)) yn.e++;
        static const char z[] = "0";
        str zero = {z, z + 1};
        int d = num_part_cmp(xn.p == xn.e ? zero : xn, yn.p == yn.e ? zero : yn);
        if (d) return d;
        str xs = {xn.e, x.e}, ys = {yn.e, y.e};
        if ((xs.p == xs.e) != (ys.p == ys.e)) return xs.p == xs.e ? 1 : -1;
        if ((d = ascii_cmp(xs, ys))) return d;
    }
    if (!pa.p || !pb.p) return pa.p ? -1 : pb.p ? 1 : 0;
    while (pa.p < pa.e && pb.p < pb.e) {
        str x = next_part(&pa, '.'), y = next_part(&pb, '.');
        int nx = is_num(x), ny = is_num(y), d;
        if (nx != ny) return nx ? -1 : 1;
        if ((d = nx ? num_part_cmp(x, y) : ascii_cmp(x, y))) return d;
    }
    return (pa.p < pa.e) - (pb.p < pb.e);
}

void compare_versions(void) {
    str a = trim(param("first")), b = trim(param("second"));
    if (!have(a) || !have(b) || a.p == a.e || b.p == b.e) fail("missing argument: first and second, the versions");
    int d = version_cmp(a, b);
    say_str(a);
    if (!d) {
        say(" and ");
        say_str(b);
        say(" are the same version");
        return;
    }
    say(d > 0 ? " is newer than " : " is older than ");
    say_str(b);
}

/* items of a list: one a line if it has more than one line, else separated by commas (or semicolons) */
static int items(str s, str **list) {
    s = trim(s);
    int lines = find_byte(s.p, s.e, '\n') != 0, semis = 0, commas = 0, n = 0, most = 1;
    for (const char *q = s.p; q < s.e; q++) semis += *q == ';', commas += *q == ',';
    char sep = lines ? '\n' : semis > commas ? ';' : ',';
    if (!lines && !semis && !commas) sep = ' ';
    for (const char *q = s.p; q < s.e; q++) most += *q == sep;
    str *out = *list = alloc(sizeof(str) * most);
    while (s.p < s.e) {
        str x = trim(next_part(&s, sep));
        if (x.p < x.e && (x.e[-1] == ',' || x.e[-1] == ';') && lines) x.e--; /* ("a,\nb,") */
        while (x.p < x.e && (*x.p == '-' || *x.p == '*') && x.p + 1 < x.e && x.p[1] == ' ') x = trim((str){x.p + 1, x.e});
        if (x.p == x.e) continue;
        out[n++] = x;
    }
    return n;
}

/* a version's first 4 numbers, read once (the sort compares them first, and the whole only where they tie) */
typedef struct {
    u32 k[4];
    u8 plain; /* (no letters or pre-release: the numbers are the whole order) */
} vkey;

static void version_key(str v, vkey *out) {
    v = clean(v);
    out->plain = 1;
    int i = 0;
    for (const char *p = v.p; p < v.e && i < 4;) {
        u32 x = 0;
        int digits = 0, sig = 0; /* (sig: digits after the leading zeros; more than 9: above any of 9, ties the whole) */
        while (p < v.e && digit(*p)) sig += sig || *p != '0', x = x < 400000000u ? x * 10 + (*p - '0') : x, p++, digits++;
        out->k[i++] = sig > 9 ? 0xffffffffu : x;
        if (!digits || sig > 9) out->plain = 0;
        if (p < v.e && *p != '.') { /* (letters or a pre-release: the numbers so far, then the whole compared) */
            out->plain = 0;
            break;
        }
        if (p < v.e) p++;
    }
    while (i < 4) out->k[i++] = 0;
    int dots = 0;
    for (const char *p = v.p; p < v.e; p++) dots += *p == '.';
    if (dots > 3) out->plain = 0;
}

static int key_cmp(const vkey *a, const vkey *b, str x, str y) {
    for (int i = 0; i < 4; i++)
        if (a->k[i] != b->k[i]) return a->k[i] < b->k[i] ? -1 : 1;
    return a->plain && b->plain ? 0 : version_cmp(x, y);
}

void sort_versions(void) {
    str s = param("versions");
    if (!have(s)) fail("missing argument: versions");
    str *v;
    int n = items(s, &v);
    str o = trim(param("order"));
    int desc = have(o) && o.p < o.e && (lower((u8)*o.p) == 'd' || lower((u8)*o.p) == 'n');
    vkey *k = alloc(sizeof(vkey) * (n + 1));
    for (int i = 0; i < n; i++) version_key(v[i], &k[i]);
    int *a = alloc(sizeof(int) * (n + 1)), *b = alloc(sizeof(int) * (n + 1)); /* (a stable merge sort of indices) */
    for (int i = 0; i < n; i++) a[i] = i;
    for (int w = 1; w < n; w *= 2) {
        for (int lo = 0; lo < n; lo += 2 * w) {
            int mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n, i = lo, j = mid, m = lo;
            while (i < mid && j < hi) {
                int d = key_cmp(&k[a[j]], &k[a[i]], v[a[j]], v[a[i]]);
                b[m++] = (desc ? d > 0 : d < 0) ? a[j++] : a[i++];
            }
            while (i < mid) b[m++] = a[i++];
            while (j < hi) b[m++] = a[j++];
        }
        int *t = a;
        a = b, b = t;
    }
    str *sorted = alloc(sizeof(str) * (n + 1));
    for (int i = 0; i < n; i++) sorted[i] = v[a[i]];
    v = sorted;
    for (int i = 0; i < n; i++) {
        if (i) say(", ");
        say_str(v[i]);
    }
}

/* ---- lists: which items are in both, and which in one only (ignoring letter case and spaces around them) */
static u32 hash(str s) {
    u32 h = 2166136261u;
    for (const char *q = s.p; q < s.e; q++) h = (h ^ (u32)lower((u8)*q)) * 16777619u;
    return h;
}

typedef struct {
    str *v;
    int n;
    int *table; /* indices + 1, open addressing; size a power of 2 */
    u32 mask;
} set;

static void set_make(set *s, str text) {
    s->n = items(text, &s->v);
    u32 size = 16;
    while (size < 2u * (u32)s->n + 2) size *= 2;
    s->mask = size - 1;
    s->table = alloc(sizeof(int) * size);
    memset(s->table, 0, sizeof(int) * size);
    int k = 0;
    for (int i = 0; i < s->n; i++) { /* (a repeated item kept once) */
        u32 h = hash(s->v[i]) & s->mask;
        int dup = 0;
        for (; s->table[h]; h = (h + 1) & s->mask)
            if (str_ieq(s->v[s->table[h] - 1], s->v[i])) dup = 1;
        if (dup) continue;
        s->v[k] = s->v[i];
        s->table[h] = ++k;
    }
    s->n = k;
}

static int set_has(set *s, str x) {
    for (u32 h = hash(x) & s->mask; s->table[h]; h = (h + 1) & s->mask)
        if (str_ieq(s->v[s->table[h] - 1], x)) return 1;
    return 0;
}

static void say_group(const char *title, str *v, const u8 *in, int n, int want) {
    int count = 0;
    for (int i = 0; i < n; i++) count += in[i] == want;
    say(title);
    say(" (");
    say_int(count);
    say(")");
    if (!count) return;
    say(": ");
    int j = 0;
    for (int i = 0; i < n; i++)
        if (in[i] == want) {
            if (j++) say(", ");
            say_str(v[i]);
        }
}

void compare_lists(void) {
    str a = param("first"), b = param("second");
    if (!have(a) || !have(b)) fail("missing argument: first and second, the lists");
    set A, B;
    set_make(&A, a);
    set_make(&B, b);
    u8 *ina = alloc(A.n + 1), *inb = alloc(B.n + 1);
    for (int i = 0; i < A.n; i++) ina[i] = set_has(&B, A.v[i]);
    for (int i = 0; i < B.n; i++) inb[i] = set_has(&A, B.v[i]);
    say_group("in both", A.v, ina, A.n, 1);
    put('\n');
    say_group("only in the first", A.v, ina, A.n, 0);
    put('\n');
    say_group("only in the second", B.v, inb, B.n, 0);
}
