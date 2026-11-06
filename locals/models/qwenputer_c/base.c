/* The toolbox's C side: c_main (the dispatcher's call for a name it does not have), a call's parameters, output,
   memory and numbers. */
#include "tools.h"

#ifdef HOST
#include <setjmp.h>
#include <stdio.h>
extern jmp_buf host_end;
extern char host_heap[], *host_heap_end, host_low[];
#endif

const char *call_name;
const char *text, *text_end;

static struct {
    str name, value;
} params[16];
static int n_params;
static void (*close_run)(void) __attribute__((unused));
static char *heap, *heap_end;
static char *low, *low_end; /* (memory for tables: RAM's 0x1c00..0x3000, which the assembly tools' sort uses) */

static const struct {
    const char *name;
    void (*run)(void);
} TOOLS[] = {
    {"json_check", json_check},       {"json_format", json_format},
    {"json_get", json_get},           {"json_count", json_count},
    {"column_stats", column_stats},   {"table_sort", table_sort},
    {"table_filter", table_filter},   {"table_totals", table_totals},
    {"compare_versions", compare_versions}, {"sort_versions", sort_versions},
    {"compare_lists", compare_lists}, {"truth_table", truth_table},
    {"logic_equivalent", logic_equivalent}, {"regex_find", regex_find},
    {"regex_replace", regex_replace}, {"remember", remember},
    {"recall", recall},               {"forget", forget},
    {"check_call", check_call},       {"copy_check", copy_check},
    {"run_python", run_python},
    {"stats", stats},
    {"percent_change", percent_change},
    {"round_number", round_number},
    {"compound_interest", compound_interest},
    {"loan_payment", loan_payment},
    {"is_prime", is_prime},
    {"factor", factor},
    {"convert_units", convert_units},
    {"roman_numeral", roman_numeral},
    {"number_to_words", number_to_words},
    {"replace_text", replace_text},
    {"change_case", change_case},
    {"text_part", text_part},
    {"sort_lines", sort_lines},
    {"unique_items", unique_items},
    {"word_frequency", word_frequency},
    {"is_palindrome", is_palindrome},
    {"is_anagram", is_anagram},
    {"validate", validate},
    {"encode", encode},
    {"decode", decode},
    {"edit_distance", edit_distance},
    {"time_add", time_add},
    {"time_between", time_between},
    {"age", age},
    {"date_info", date_info},
    {"unix_time", unix_time},
    {"convert_timezone", convert_timezone},
    {"check_digit", check_digit},
    {"hash_text", hash_text},
};

static int starts(const char *p, const char *e, const char *z) {
    for (; *z; p++, z++)
        if (p >= e || *p != *z) return 0;
    return 1;
}

const char *find_byte(const char *p, const char *e, int c) {
    for (; ((size_t)p & 3) && p < e; p++)
        if (*p == c) return p;
    for (; p + 4 <= e; p += 4)
        if (HAS_BYTE(*(const u32 *)p, (u32)c)) break;
    for (; p < e; p++)
        if (*p == c) return p;
    return 0;
}

const char *find_n(const char *p, const char *e, const char *z, size_t n) {
    if ((size_t)(e - p) < n) return 0;
    for (e -= n - 1; p < e && (p = find_byte(p, e, *z)); p++)
        if (p[1] == z[1] && !memcmp(p + 2, z + 2, n - 2)) return p;
    return 0;
}

/* the parameters of the call's first function: <parameter=NAME>\nVALUE\n</parameter>, one after another */
static void read_params(void) {
    const char *p = find(text, text_end, "<function="), *e = text_end;
    n_params = 0;
    if (!p) return;
    while (p < e && *p != '>') p++;
    p++;
    while (n_params < 16) {
        p = json_space(p, e);
        if (!starts(p, e, "<parameter=")) return;
        const char *n = p + 11, *ne = n;
        while (ne < e && *ne != '>') ne++;
        if (ne >= e) return;
        const char *v = ne + 1;
        if (v < e && *v == '\n') v++;
        const char *ve = find(v, e, "</parameter>");
        if (!ve) return;
        p = ve + 12;
        if (ve > v && ve[-1] == '\n') ve--;
        params[n_params].name = (str){n, ne};
        params[n_params].value = (str){v, ve};
        n_params++;
    }
}

str param(const char *name) {
    for (int i = 0; i < n_params; i++)
        if (str_is(params[i].name, name)) return params[i].value;
    return (str){0, 0};
}

str trim(str s) {
    if (!s.p) return s;
    while (s.p < s.e && space(*s.p)) s.p++;
    while (s.e > s.p && space(s.e[-1])) s.e--;
    return s;
}

int str_is(str s, const char *z) {
    const char *p = s.p;
    for (; p < s.e; p++, z++)
        if (*p != *z) return 0;
    return *z == 0;
}

int str_ieq(str a, str b) {
    if (a.e - a.p != b.e - b.p) return 0;
    for (; a.p < a.e; a.p++, b.p++)
        if (lower((u8)*a.p) != lower((u8)*b.p)) return 0;
    return 1;
}

/* ---- output */
char *capture_at, *capture_end;

#ifdef HOST
void put(int c) {
    if (capture_at) {
        if (capture_at < capture_end) *capture_at++ = (char)c;
        return;
    }
    if (c < 256) putchar(c);
    else printf("<%d>", c);
}
#endif

void say(const char *z) {
    for (; *z; z++) put((u8)*z);
}

void say_str(str s) {
    for (const char *p = s.p; p < s.e; p++) put((u8)*p);
}

static void say_u32(u32 v, int width) { /* (width: at least that many digits, zeros in front) */
    char b[10];
    int n = 0;
    do {
        b[n++] = '0' + v % 10;
        v /= 10;
    } while (v || n < width);
    while (n) put(b[--n]);
}

void say_u(u64 v) { /* (in 9-digit pieces: the computer divides 32 bits at a time) */
    if (v >> 32 == 0) return say_u32((u32)v, 1);
    u64 hi = v / 1000000000;
    u32 lo = (u32)(v - hi * 1000000000);
    say_u(hi);
    say_u32(lo, 9);
}

void say_int(i64 v) {
    if (v < 0) {
        put('-');
        say_u(-(u64)v);
    } else
        say_u(v);
}

static const i64 TEN[19] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000,
                            10000000000LL, 100000000000LL, 1000000000000LL, 10000000000000LL,
                            100000000000000LL, 1000000000000000LL, 10000000000000000LL,
                            100000000000000000LL, 1000000000000000000LL};

void say_fixed(i64 v, int frac) {
    u64 m = v < 0 ? -(u64)v : (u64)v;
    if (v < 0) put('-');
    say_u(m / TEN[frac]);
    u64 f = m % TEN[frac];
    if (!f) return;
    put('.');
    while (f % 10 == 0) {
        f /= 10;
        frac--;
    }
    for (int i = frac - 1; i >= 0; i--) put('0' + (f / TEN[i]) % 10);
}

void fail(const char *msg) {
    capture_at = 0;
    say("error (");
    say(msg);
    put(')');
    done();
}

void done(void) {
#ifdef HOST
    longjmp(host_end, 1);
#else
    close_run();
    __builtin_unreachable();
#endif
}

/* ---- 64-bit division (the compiler's calls for it): 32 bits at a time where the divisor fits 32 bits */
#ifndef HOST
static u64 udivmod(u64 n, u64 d, u64 *rem) {
    u32 hi = (u32)(n >> 32), lo = (u32)n;
    if (d >> 32 == 0) {
        u32 dd = (u32)d;
        if (hi == 0) {
            *rem = lo % dd;
            return lo / dd;
        }
        u32 qh = hi / dd, r = hi % dd, ql;
        if (dd >> 16 == 0) { /* two 16-bit digits: r < dd < 2^16 */
            u32 a = r << 16 | lo >> 16, qa = a / dd;
            r = a % dd;
            u32 b = r << 16 | (lo & 0xffff), qb = b / dd;
            r = b % dd;
            ql = qa << 16 | qb;
        } else { /* a bit at a time */
            ql = 0;
            for (int i = 31; i >= 0; i--) {
                u32 top = r >> 31;
                r = r << 1 | (lo >> i & 1);
                ql <<= 1;
                if (top || r >= dd) r -= dd, ql |= 1;
            }
        }
        *rem = r;
        return (u64)qh << 32 | ql;
    }
    u64 q = 0, r = 0;
    for (int i = 63; i >= 0; i--) {
        r = r << 1 | (n >> i & 1);
        q <<= 1;
        if (r >= d) r -= d, q |= 1;
    }
    *rem = r;
    return q;
}

u64 __udivdi3(u64 n, u64 d) {
    u64 r;
    return udivmod(n, d, &r);
}

u64 __umoddi3(u64 n, u64 d) {
    u64 r;
    udivmod(n, d, &r);
    return r;
}

i64 __divdi3(i64 n, i64 d) {
    u64 r, q = udivmod(n < 0 ? -(u64)n : (u64)n, d < 0 ? -(u64)d : (u64)d, &r);
    return (n < 0) != (d < 0) ? -(i64)q : (i64)q;
}

i64 __moddi3(i64 n, i64 d) {
    u64 r;
    udivmod(n < 0 ? -(u64)n : (u64)n, d < 0 ? -(u64)d : (u64)d, &r);
    return n < 0 ? -(i64)r : (i64)r;
}
#endif

/* ---- memory */
const char *oom_why;

static __attribute__((noreturn, cold)) void oom(void) {
    fail(oom_why ? oom_why : "out of memory: the computer has about 15 KB for a call");
}

void *alloc(size_t n) {
    n = (n + 7) & ~(size_t)7;
    if ((size_t)(heap_end - heap) < n) oom();
    void *p = heap;
    heap += n;
    return p;
}

size_t heap_left(void) { return heap_end - heap; }

char *heap_mark(void) { return heap; }

static char *top_was;

void *alloc_top(size_t n) { /* from the heap's end, given back with free_top (run_python's tokens while it parses) */
    if ((size_t)(heap_end - heap) < n + 8) oom();
    top_was = heap_end;
    heap_end = (char *)(((size_t)heap_end - n) & ~(size_t)7); /* (8-aligned: the end is the stack's floor, and the
                                                                    computer has no unaligned loads) */
    return heap_end;
}

void free_top(void) { heap_end = top_was; }

void heap_back(char *at) { /* the heap from `at` on free again (run_python: a loop's round given back) */
    if (at >= heap) return;
#ifdef HOST
    memset(at, 0xa5, (size_t)(heap - at)); /* (poisoned: a use of what was given back shows in the tests) */
#endif
    heap = at;
}

size_t low_left(void) { return low_end - low; }

void *alloc_low(size_t n) {
    n = (n + 7) & ~(size_t)7;
    if ((size_t)(low_end - low) < n) return alloc(n);
    void *p = low;
    low += n;
    return p;
}

char *heap_reserve(char *here, size_t stack) {
#ifdef HOST
    return here - stack;
#else
    char *floor = here - stack;
    if (floor < heap_end) heap_end = floor > heap ? floor : heap;
    return heap_end;
#endif
}

void *memcpy(void *d, const void *s, size_t n) {
    u8 *a = d;
    const u8 *b = s;
    if ((((size_t)a | (size_t)b) & 3) == 0) {
        for (; n >= 4; n -= 4, a += 4, b += 4) *(u32 *)a = *(const u32 *)b;
    }
    while (n--) *a++ = *b++;
    return d;
}

void *memmove(void *d, const void *s, size_t n) {
    u8 *a = d;
    const u8 *b = s;
    if (a <= b || a >= b + n) return memcpy(d, s, n);
    while (n--) a[n] = b[n];
    return d;
}

void *memset(void *d, int c, size_t n) {
    u8 *a = d;
    if (((size_t)a & 3) == 0) {
        u32 w = (u8)c * 0x01010101u;
        for (; n >= 4; n -= 4, a += 4) *(u32 *)a = w;
    }
    while (n--) *a++ = (u8)c;
    return d;
}

int memcmp(const void *a, const void *b, size_t n) {
    const u8 *x = a, *y = b;
    for (; n; n--, x++, y++)
        if (*x != *y) return *x - *y;
    return 0;
}

size_t strlen(const char *z) {
    const char *p = z;
    while (*p) p++;
    return p - z;
}

/* ---- numbers */
int read_num(str s, num *out, int loose) {
    s = trim(s);
    const char *p = s.p, *e = s.e;
    int neg = 0, cur = 0, digits = 0, frac = -1;
    i64 v = 0;
    if (!p || p == e) return 0;
    { /* (most numbers at once: [-]digits[.digits], 9 digits at most, in 32 bits) */
        const char *q = p + (*p == '-');
        if (e - q >= 1 && e - q <= 9 && digit(*q)) {
            u32 x = 0;
            int f = -1;
            for (const char *r = q; r < e; r++)
                if (digit(*r)) x = x * 10 + (u32)(*r - '0'), f += f >= 0;
                else if (*r == '.' && f < 0 && r + 1 < e) f = 0;
                else goto slow;
            out->v = q > p ? -(i64)x : (i64)x;
            out->frac = f < 0 ? 0 : f;
            return 1;
        }
    }
slow:
    for (;;) { /* a sign and, loosely, a currency sign, in either order */
        if (p < e && (*p == '-' || *p == '+') && !neg) {
            neg = *p == '-' ? 1 : 2;
            p++;
        } else if (e - p >= 3 && (u8)p[0] == 0xe2 && (u8)p[1] == 0x88 && (u8)p[2] == 0x92 && !neg) {
            neg = 1; /* U+2212 */
            p += 3;
        } else if (loose && !cur && p < e && *p == '$') {
            cur = 1;
            p++;
        } else if (loose && !cur && e - p >= 3 && (u8)p[0] == 0xe2 && (u8)p[1] == 0x82 && (u8)p[2] == 0xac) {
            cur = 1;
            p += 3;
        } else if (loose && !cur && e - p >= 2 && (u8)p[0] == 0xc2 && (u8)p[1] == 0xa3) {
            cur = 1;
            p += 2;
        } else
            break;
    }
    if (loose && e > p && e[-1] == '%') e--;
    int group = -1, whole = 0; /* digits since the last thousands separator (-1: none yet), before the point */
    for (; p < e; p++) {
        if (digit(*p)) {
            if (digits == 18) return 0;
            v = v * 10 + (*p - '0');
            if (v || frac >= 0) digits++;
            if (frac >= 0) {
                if (++frac > 9) return 0;
            } else {
                whole++;
                if (group >= 0) group++;
            }
        } else if (*p == '.' && frac < 0) {
            if (group >= 0 && group != 3) return 0;
            frac = 0;
        } else if (*p == ',' && loose && frac < 0 && p + 1 < e && digit(p[1])) {
            if (group >= 0 ? group != 3 : whole == 0 || whole > 3) return 0;
            group = 0;
        } else
            return 0;
    }
    if (group >= 0 && frac < 0 && group != 3) return 0;
    if (frac == 0) { /* "12." */
        if (p[-1] == '.') return 0;
    }
    int any = 0;
    for (const char *q = s.p; q < e; q++) any |= digit(*q);
    if (!any) return 0;
    out->v = neg == 1 ? -v : v;
    out->frac = frac < 0 ? 0 : frac;
    return 1;
}

int num_cmp(num a, num b) {
    if (a.frac == b.frac) return a.v < b.v ? -1 : a.v > b.v; /* (the same decimals: as they are, no division) */
    { /* the one with fewer decimals scaled to the other's, where that fits 64 bits */
        static const i64 LIM[10] = {0x7fffffffffffffffLL, 922337203685477580LL, 92233720368547758LL, 9223372036854775LL,
                                    922337203685477LL, 92233720368547LL, 9223372036854LL, 922337203685LL, 92233720368LL,
                                    9223372036LL};
        int d = a.frac > b.frac ? a.frac - b.frac : b.frac - a.frac;
        i64 *lo = a.frac < b.frac ? &a.v : &b.v;
        if (*lo <= LIM[d] && *lo >= -LIM[d]) {
            *lo *= TEN[d];
            return a.v < b.v ? -1 : a.v > b.v;
        }
    }
    int sa = a.v < 0 ? -1 : a.v > 0, sb = b.v < 0 ? -1 : b.v > 0;
    if (sa != sb) return sa < sb ? -1 : 1;
    u64 ma = a.v < 0 ? -(u64)a.v : (u64)a.v, mb = b.v < 0 ? -(u64)b.v : (u64)b.v;
    u64 wa = ma / TEN[a.frac], wb = mb / TEN[b.frac];
    u64 fa = (ma % TEN[a.frac]) * TEN[9 - a.frac], fb = (mb % TEN[b.frac]) * TEN[9 - b.frac];
    int c = wa != wb ? (wa < wb ? -1 : 1) : fa != fb ? (fa < fb ? -1 : 1) : 0;
    return sa < 0 ? -c : c;
}

/* ---- the dispatcher's call: the tool named at 0x6c0 on the call's text at 0x3000 (its end at 0x6f0); 0 if there
   is none of that name. A tool's response ends at `close`, the dispatcher's end of a response and park. */
#ifndef HOST
__attribute__((section(".text.entry"))) int c_main(void (*close)(void)) {
    close_run = close;
    capture_at = 0;
    call_name = (const char *)0x6c0;
    text = (const char *)0x3000;
    text_end = *(const char *const *)0x6f0;
    heap = (char *)(((size_t)text_end + 15) & ~(size_t)7);
    heap_end = (char *)0x6c00;
    low = (char *)0x1c00, low_end = (char *)0x3000;
    oom_why = 0;
#else
int c_main(void) {
    capture_at = 0;
    heap = host_heap;
    heap_end = host_heap_end;
    low = host_low, low_end = host_low + 5120;
    oom_why = 0;
#endif
    read_params();
    for (size_t i = 0; i < sizeof TOOLS / sizeof TOOLS[0]; i++) {
        const char *a = TOOLS[i].name, *b = call_name;
        while (*a && *a == *b) a++, b++;
        if (*a == 0 && *b == 0) {
            TOOLS[i].run();
            done();
        }
    }
    return 0;
}
