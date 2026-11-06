/* Tables: column_stats, table_sort, table_filter, table_totals. A table is CSV (commas or semicolons, "quoted"
   cells), tab-separated, or a Markdown table (cells between |, its --- row skipped); its first row is its header
   unless every cell of it is a number. A column is named by its header, its number (from 1) or its letter (A). */
#include "tools.h"

static const char *tp, *te; /* the table's text */
static char sep;            /* '|', '\t', ',' or ';' */

/* one row's cells (up to `most`) and its line; the next row's start in *p. 0 at the table's end. */
static int is_rule(str c) { /* a Markdown table's ---, :--, --: or :-: */
    c = trim(c);
    int dash = 0;
    for (const char *q = c.p; q < c.e; q++) {
        if (*q == '-') dash = 1;
        else if (*q != ':') return 0;
    }
    return dash;
}

static int row(const char **pp, str *cells, int most, str *line) {
    const char *p = *pp;
    for (;;) { /* (blank lines, and a Markdown table's rule) */
        while (p < te && (*p == '\n' || *p == '\r')) p++;
        if (p >= te) return 0;
        const char *q = p;
        while (q < te && (*q == ' ' || *q == '\t')) q++;
        if (q < te && *q != '\n' && *q != '\r') break;
        p = q;
    }
    *line = (str){p, p};
    int n = 0;
    if (sep == '|') {
        const char *e = find_byte(p, te, '\n');
        if (!e) e = te;
        str l = trim((str){p, e});
        *line = l;
        if (l.p < l.e && *l.p == '|') l.p++;
        if (l.e > l.p && l.e[-1] == '|') l.e--;
        for (const char *c = l.p;;) {
            const char *d = c;
            while (d < l.e && *d != '|') d++;
            if (n < most) cells[n] = trim((str){c, d});
            n++;
            if (d >= l.e) break;
            c = d + 1;
        }
        *pp = e < te ? e + 1 : te;
        int rule = 1;
        for (int i = 0; i < n && i < most; i++) rule &= is_rule(cells[i]);
        if (rule) return row(pp, cells, most, line);
        return n;
    }
    for (;;) { /* CSV: a cell, quoted or not */
        while (p < te && *p == ' ') p++;
        const char *c = p, *d;
        if (p < te && *p == '"') {
            for (p++; p < te; p++)
                if (*p == '"') {
                    if (p + 1 < te && p[1] == '"') p++;
                    else break;
                }
            d = p; /* (its closing quote, or the table's end without one) */
            if (p < te) p++;
            c++; /* (the cell without its quotes; "" stays as written) */
            while (p < te && *p != sep && *p != '\n' && *p != '\r') p++;
        } else {
            while (p < te && *p != sep && *p != '\n' && *p != '\r') p++;
            d = p;
        }
        if (n < most) cells[n] = trim((str){c, d});
        n++;
        if (p < te && *p == sep) {
            p++;
            continue;
        }
        break;
    }
    line->e = p;
    if (p < te && *p == '\r') p++;
    if (p < te && *p == '\n') p++;
    *pp = p;
    return n;
}

#define COLS 64
static str head[COLS], head_line, rule_line; /* (rule_line: a Markdown table's, after its header, or none) */
static int n_head, has_head;
static const char *body; /* the first row after the header */

static void read_table(void) {
    str t = param("table");
    if (!have(t)) t = param("text");
    if (!have(t)) fail("missing argument: table");
    tp = t.p, te = t.e;
    const char *nl = find_byte(tp, te, '\n'), *e1 = nl ? nl : te;
    int commas = 0, semis = 0, q = 0, bars = 0, tabs = 0;
    for (const char *p = tp; p < e1; p++) {
        if (*p == '"') q ^= 1;
        else if (!q) commas += *p == ',', semis += *p == ';', bars += *p == '|', tabs += *p == '\t';
    }
    sep = bars ? '|' : tabs ? '\t' : semis > commas ? ';' : ',';
    const char *p = tp;
    n_head = row(&p, head, COLS, &head_line);
    if (!n_head) fail("an empty table");
    rule_line = (str){0, 0};
    if (sep == '|') { /* (row() skips the rule: the line after the header, if it is one) */
        const char *q = p;
        while (q < te && (*q == '\n' || *q == '\r' || *q == ' ')) q++;
        const char *e = find_byte(q, te, '\n');
        str l = trim((str){q, e ? e : te});
        int rule = l.p < l.e;
        for (const char *r = l.p; r < l.e; r++) rule &= *r == '|' || *r == '-' || *r == ':' || *r == ' ';
        if (rule) rule_line = l;
    }
    if (n_head > COLS) n_head = COLS;
    has_head = 0;
    for (int i = 0; i < n_head; i++) {
        num v;
        if (head[i].p < head[i].e && !read_num(head[i], &v, 1)) has_head = 1;
    }
    body = has_head ? p : tp;
}

static void say_columns(void) {
    for (int i = 0; i < n_head; i++) {
        if (i) say(", ");
        say_str(head[i]);
    }
}

static int column(const char *name) { /* the column's index; a column it cannot find ends the run */
    str c = trim(param(name));
    if (!have(c) || c.p == c.e) {
        say("error (missing argument: ");
        say(name);
        say(has_head ? "; the columns: " : ")");
        if (has_head) say_columns(), put(')');
        done();
    }
    if (has_head)
        for (int i = 0; i < n_head; i++)
            if (str_ieq(head[i], c)) return i;
    num v;
    if (read_num(c, &v, 0) && v.frac == 0 && v.v >= 1 && v.v <= n_head) return (int)v.v - 1;
    if (c.e - c.p == 1 && (unsigned)(lower((u8)*c.p) - 'a') < (unsigned)n_head) return lower((u8)*c.p) - 'a';
    say("error (no column \"");
    say_str(c);
    say("\"; ");
    if (has_head) say("the columns: "), say_columns();
    else say("the table has no header: name a column by its number, 1 to "), say_int(n_head);
    put(')');
    done();
}

/* ---- exact sums: whole part in w1 * 10^18 + w0, fraction in billionths f */
typedef struct {
    u64 w1, w0;
    u32 f;
} dec;

static void dec_add(dec *a, num x) { /* |x| */
    u64 m = x.v < 0 ? -(u64)x.v : (u64)x.v;
    if (!x.frac) { /* (a whole number: no division) */
        a->w0 += m;
        if (a->w0 >= 1000000000000000000ULL) a->w0 -= 1000000000000000000ULL, a->w1++;
        return;
    }
    static const u32 P[10] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};
    u64 w = m / P[x.frac];
    u32 f = (u32)(m % P[x.frac]) * P[9 - x.frac];
    a->f += f;
    if (a->f >= 1000000000u) a->f -= 1000000000u, w++;
    a->w0 += w;
    if (a->w0 >= 1000000000000000000ULL) a->w0 -= 1000000000000000000ULL, a->w1++;
}

static int dec_cmp(dec a, dec b) {
    if (a.w1 != b.w1) return a.w1 < b.w1 ? -1 : 1;
    if (a.w0 != b.w0) return a.w0 < b.w0 ? -1 : 1;
    return a.f != b.f ? (a.f < b.f ? -1 : 1) : 0;
}

static dec dec_sub(dec a, dec b) { /* a - b, a >= b */
    dec r;
    int borrow = a.f < b.f;
    r.f = a.f + (borrow ? 1000000000u : 0) - b.f;
    u64 w0 = a.w0, w1 = a.w1;
    if (borrow) {
        if (w0 == 0) w0 = 1000000000000000000ULL, w1--;
        w0--;
    }
    if (w0 < b.w0) w0 += 1000000000000000000ULL, w1--;
    r.w0 = w0 - b.w0;
    r.w1 = w1 - b.w1;
    return r;
}

static void say_frac9(u32 f) {
    if (!f) return;
    char b[9];
    int n = 9;
    for (int i = 8; i >= 0; i--) b[i] = '0' + f % 10, f /= 10;
    while (b[n - 1] == '0') n--;
    put('.');
    for (int i = 0; i < n; i++) put(b[i]);
}

static void say_dec(dec a, int neg) {
    if (neg && (a.w1 || a.w0 || a.f)) put('-');
    if (a.w1) {
        say_u(a.w1);
        char b[18];
        u64 w = a.w0;
        for (int i = 17; i >= 0; i--) b[i] = '0' + w % 10, w /= 10;
        for (int i = 0; i < 18; i++) put(b[i]);
    } else
        say_u(a.w0);
    say_frac9(a.f);
}

/* a sum of signed values: its positive and negative parts */
typedef struct {
    dec pos, neg;
} sum;

static void sum_add(sum *s, num x) { dec_add(x.v < 0 ? &s->neg : &s->pos, x); }

static void say_sum(sum s) {
    if (dec_cmp(s.pos, s.neg) >= 0) say_dec(dec_sub(s.pos, s.neg), 0);
    else say_dec(dec_sub(s.neg, s.pos), 1);
}

/* the sum over n, to 9 decimals (rounded half away from zero; " (rounded)" if that changed it) */
static void say_mean(sum s, u32 n) {
    int neg = dec_cmp(s.pos, s.neg) < 0;
    dec a = neg ? dec_sub(s.neg, s.pos) : dec_sub(s.pos, s.neg);
    /* as base-10^9 limbs: w1, w0's two halves, f */
    u32 l[4] = {(u32)a.w1, (u32)(a.w0 / 1000000000u), (u32)(a.w0 % 1000000000u), a.f}, q[4];
    u64 r = 0;
    for (int i = 0; i < 4; i++) {
        u64 x = r * 1000000000u + l[i];
        q[i] = (u32)(x / n);
        r = x % n;
    }
    int rounded = r != 0;
    if (2 * r >= n) /* (half away from zero) */
        for (int i = 3; i >= 0 && ++q[i] == 1000000000u; i--) q[i] = 0;
    dec m = {q[0], (u64)q[1] * 1000000000u + q[2], q[3]};
    say_dec(m, neg);
    if (rounded) say(" (rounded)");
}

/* ---- the tools */
#define ROWS 1000
#define GROUPS 64

typedef struct { /* a row: its line, and its cells of the key and label columns, as offsets from tp */
    i64 v;           /* (the key's value, if numeric: v / 10^frac) */
    u16 line[2], key[2], label[2];
    u8 frac, numeric;
} item;
#define NV(it) ((num){(it)->v, (it)->frac})
/* row x's number before row y's (at once where they have as many decimals) */
#define LESS(x, y) (items[x].frac == items[y].frac ? items[x].v < items[y].v : num_cmp(NV(&items[x]), NV(&items[y])) < 0)

static item *items;
static int n_items;

static str at(const u16 *o) { return o[0] == 0xffff ? (str){0, 0} : (str){tp + o[0], tp + o[1]}; }
#define LINE(it) at((it)->line)
#define KEY(it) at((it)->key)
#define LABEL(it) at((it)->label)

static void set(u16 *o, str s) {
    if (!s.p) o[0] = o[1] = 0xffff;
    else o[0] = (u16)(s.p - tp), o[1] = (u16)(s.e - tp);
}

/* the rows after the header, with the cells of columns a (key) and b (label, or -1) */
static void read_rows(int a, int b, size_t keep) { /* (keep: memory the tool needs besides) */
    /* the rows in the heap, and the tools' arrays of row numbers (2 ints a row) in the memory kept for tables while
       they fit there (else in the heap as well: 3 ints a row more) */
    size_t left = heap_left() > keep + 512 ? heap_left() - keep - 512 : 0, low = low_left();
    int both = (int)(left / sizeof(item)), alone = (int)(left / (sizeof(item) + 3 * sizeof(int)));
    if (both > (int)(low / (2 * sizeof(int))) - 2) both = (int)(low / (2 * sizeof(int))) - 2;
    int cap = both > alone ? both : alone;
    if (cap > ROWS) cap = ROWS;
    items = alloc(sizeof(item) * (cap + 1));
    n_items = 0;
    str cells[COLS];
    const char *p = body;
    str line;
    int n;
    while ((n = row(&p, cells, COLS, &line))) {
        if (n_items == cap) fail(cap == ROWS ? "more than 1,000 rows" : "a table too big for the computer's memory");
        item *it = &items[n_items++];
        set(it->line, line);
        set(it->key, a < n ? cells[a] : (str){line.e, line.e});
        set(it->label, b >= 0 && b < n ? cells[b] : (str){0, 0});
        num x;
        it->numeric = (u8)read_num(KEY(it), &x, 1);
        it->v = x.v, it->frac = (u8)x.frac;
    }
}

static int first_label(int col) { /* a row's name for its min and max: the first column that is not col */
    return col == 0 ? (n_head > 1 ? 1 : -1) : 0;
}

void column_stats(void) {
    read_table();
    int c = column("column");
    read_rows(c, first_label(c), 0);
    sum s = {0};
    int n = 0, lo = -1, hi = -1, empty = 0, bad = 0, bad_rows[5];
    for (int i = 0; i < n_items; i++) {
        item *it = &items[i];
        if (!it->numeric) {
            if (KEY(it).p == KEY(it).e) empty++;
            else if (bad < 5) bad_rows[bad++] = i;
            else bad++;
            continue;
        }
        sum_add(&s, NV(it));
        if (lo < 0 || LESS(i, lo)) lo = i;
        if (hi < 0 || LESS(hi, i)) hi = i;
        n++;
    }
    if (!n) fail("the column has no numbers");
    /* the median: the (n / 2)-th number (from 0) found by selection, quickselect's with a 3-way partition (equal
       numbers in the middle), and the one before it the largest of those left of it */
    int *ord = alloc_low(sizeof(int) * n), k = 0;
    for (int i = 0; i < n_items; i++)
        if (items[i].numeric) ord[k++] = i;
    int want = n / 2;
    for (int lo2 = 0, hi2 = n - 1; lo2 < hi2;) {
        int x = ord[lo2], y = ord[lo2 + (hi2 - lo2) / 2], z = ord[hi2]; /* (the pivot: the median of three) */
        int pv = LESS(x, y) ? (LESS(y, z) ? y : LESS(x, z) ? z : x) : (LESS(x, z) ? x : LESS(y, z) ? z : y);
        int lt = lo2, gt = hi2, i = lo2;
        while (i <= gt) {
            int t = ord[i];
            if (LESS(t, pv)) ord[i++] = ord[lt], ord[lt++] = t;
            else if (LESS(pv, t)) ord[i] = ord[gt], ord[gt--] = t;
            else i++;
        }
        if (want < lt) hi2 = lt - 1;
        else if (want > gt) lo2 = gt + 1;
        else break;
    }
    int below = -1; /* (for an even count: the largest left of the (n / 2)-th) */
    if (!(n & 1))
        for (int i = 0; i < want; i++)
            if (below < 0 || LESS(below, ord[i])) below = ord[i];
    say("count ");
    say_int(n);
    say(", sum ");
    say_sum(s);
    say(", average ");
    say_mean(s, n);
    say(", median ");
    if (n & 1) say_fixed(items[ord[want]].v, items[ord[want]].frac);
    else {
        sum m = {0};
        sum_add(&m, NV(&items[below]));
        sum_add(&m, NV(&items[ord[want]]));
        say_mean(m, 2);
    }
    for (int t = 0; t < 2; t++) {
        item *it = &items[t ? hi : lo];
        say(t ? ", max " : ", min ");
        say_fixed(it->v, it->frac);
        say(" (row ");
        say_int((t ? hi : lo) + 1);
        if (have(LABEL(it)) && LABEL(it).p < LABEL(it).e) say(": "), say_str(LABEL(it));
        put(')');
    }
    if (empty) say("; empty cells: "), say_int(empty);
    if (bad) {
        say("; not numbers, left out: row");
        say(bad > 1 ? "s " : " ");
        for (int i = 0; i < bad && i < 5; i++) {
            if (i) say(", ");
            say_int(bad_rows[i] + 1);
            say(" (");
            say_str(KEY(&items[bad_rows[i]]));
            put(')');
        }
        if (bad > 5) say(", ...");
    }
}

static void say_head(void) { /* the header's line as written (and a Markdown table's rule after it) */
    if (!has_head) return;
    say_str(head_line);
    put('\n');
    if (have(rule_line)) say_str(rule_line), put('\n');
}

static int text_cmp(str a, str b) { /* ASCII letters in either case, then as written */
    const char *p = a.p, *q = b.p;
    for (; p < a.e && q < b.e; p++, q++) {
        int x = lower((u8)*p), y = lower((u8)*q);
        if (x != y) return x < y ? -1 : 1;
    }
    if (p < a.e || q < b.e) return p < a.e ? 1 : -1;
    for (p = a.p, q = b.p; p < a.e; p++, q++)
        if (*p != *q) return (u8)*p < (u8)*q ? -1 : 1;
    return 0;
}

static int istarts(str s, const char *z) {
    const char *p = s.p;
    for (; *z; p++, z++)
        if (p >= s.e || lower((u8)*p) != *z) return 0;
    return 1;
}

static int numeric_col, desc; /* every non-empty cell of the sort's column a number; largest first */

static int order_cmp(const item *a, const item *b) { /* (empty cells last either way) */
    int ea = numeric_col ? !a->numeric : KEY(a).p == KEY(a).e, eb = numeric_col ? !b->numeric : KEY(b).p == KEY(b).e;
    if (ea || eb) return ea - eb;
    int d = numeric_col ? (a->frac == b->frac ? (a->v > b->v) - (a->v < b->v) : num_cmp(NV(a), NV(b))) : text_cmp(KEY(a), KEY(b));
    return desc ? -d : d;
}

void table_sort(void) {
    read_table();
    int c = column("column");
    read_rows(c, -1, 0);
    str o = trim(param("order"));
    desc = have(o) && o.p < o.e && (lower((u8)*o.p) == 'd' || istarts(o, "largest") || istarts(o, "high") ||
                                     istarts(o, "newest") || istarts(o, "z"));
    int nums = 0, texts = 0; /* (numbers, if most cells are: the others after them, as empty cells) */
    for (int i = 0; i < n_items; i++)
        if (items[i].numeric) nums++;
        else if (KEY(&items[i]).p != KEY(&items[i]).e) texts++;
    numeric_col = nums > texts;
    /* a stable merge sort of the rows' indices */
    int n = n_items, *a = alloc_low(sizeof(int) * (n + 1)), *b = alloc_low(sizeof(int) * (n + 1));
    for (int i = 0; i < n; i++) a[i] = i;
    for (int w = 1; w < n; w *= 2) {
        for (int lo = 0; lo < n; lo += 2 * w) {
            int mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n, i = lo, j = mid, k = lo;
            while (i < mid && j < hi) {
                b[k++] = order_cmp(&items[a[j]], &items[a[i]]) < 0 ? a[j++] : a[i++];
            }
            while (i < mid) b[k++] = a[i++];
            while (j < hi) b[k++] = a[j++];
        }
        int *t = a;
        a = b, b = t;
    }
    say_head();
    for (int i = 0; i < n; i++) {
        say_str(LINE(&items[a[i]]));
        if (i + 1 < n) put('\n');
    }
}

/* a condition: [op] value, op one of = == != <> > >= < <= contains, "starts with", "ends with" */
enum { EQ, NE, GT, GE, LT, LE, HAS, STARTS, ENDS };

static int contains(str hay, str needle, int where) { /* 0 anywhere, 1 at the start, 2 at the end */
    long n = needle.e - needle.p, h = hay.e - hay.p;
    if (n > h) return 0;
    for (long i = 0; i + n <= h; i++) {
        if (where == 1 && i) return 0;
        if (where == 2 && i + n != h) continue;
        str s = {hay.p + i, hay.p + i + n};
        if (str_ieq(s, needle)) return 1;
    }
    return 0;
}

void table_filter(void) {
    read_table();
    int c = column("column");
    str cond = trim(param("condition"));
    if (!have(cond) || cond.p == cond.e) fail("missing argument: condition, e.g. > 100, = Paris or contains York");
    int op = EQ;
    static const struct {
        const char *z;
        int op;
    } OPS[] = {{">=", GE}, {"<=", LE}, {"!=", NE}, {"<>", NE}, {"==", EQ}, {"=", EQ}, {">", GT}, {"<", LT},
               {"contains ", HAS}, {"starts with ", STARTS}, {"ends with ", ENDS}, {"is not ", NE}, {"is ", EQ}};
    for (size_t i = 0; i < sizeof OPS / sizeof OPS[0]; i++)
        if (istarts(cond, OPS[i].z)) {
            op = OPS[i].op;
            cond.p += strlen(OPS[i].z);
            break;
        }
    cond = trim(cond);
    if (cond.e - cond.p >= 2 && (*cond.p == '"' || *cond.p == '\'') && cond.e[-1] == *cond.p) cond.p++, cond.e--;
    num want;
    int want_num = read_num(cond, &want, 1);
    read_rows(c, -1, 0);
    int *hit = alloc_low(sizeof(int) * (n_items + 1)), n = 0;
    for (int i = 0; i < n_items; i++) {
        item *it = &items[i];
        int ok, d;
        str key = KEY(it);
        if (op >= HAS) ok = contains(key, cond, op - HAS);
        else {
            if (want_num && it->numeric) d = num_cmp(NV(it), want);
            else if (want_num && op != EQ && op != NE) continue; /* (a text cell is not > 100) */
            else d = str_ieq(key, cond) ? 0 : text_cmp(key, cond);
            ok = op == EQ ? d == 0 : op == NE ? d != 0 : op == GT ? d > 0 : op == GE ? d >= 0 : op == LT ? d < 0 : d <= 0;
        }
        if (ok) hit[n++] = i;
    }
    say_int(n);
    say(" of ");
    say_int(n_items);
    say(n_items == 1 ? " row" : " rows");
    if (!n) return;
    say(":\n");
    say_head();
    for (int i = 0; i < n; i++) {
        say_str(LINE(&items[hit[i]]));
        if (i + 1 < n) put('\n');
    }
}

void table_totals(void) {
    read_table();
    int g = column("group_by"), c = -1;
    str cv = trim(param("column"));
    if (have(cv) && cv.p < cv.e) c = column("column");
    typedef struct {
        str name;
        sum s;
        int rows, nums;
    } group;
    read_rows(g, c, sizeof(group) * GROUPS);
    group *gs = alloc(sizeof(group) * GROUPS);
    int n = 0, bad = 0;
    for (int i = 0; i < n_items; i++) {
        item *it = &items[i];
        int j = 0;
        while (j < n && !str_ieq(gs[j].name, KEY(it))) j++;
        if (j == n) {
            if (n == GROUPS) fail("more than 64 groups");
            gs[n] = (group){KEY(it), {{0}, {0}}, 0, 0};
            n++;
        }
        gs[j].rows++;
        if (c >= 0) {
            num v;
            str l = LABEL(it);
            if (have(l) && read_num(l, &v, 1)) sum_add(&gs[j].s, v), gs[j].nums++;
            else if (have(l) && l.p != l.e) bad++;
        }
    }
    for (int j = 0; j < n; j++) {
        if (j) put('\n');
        if (gs[j].name.p == gs[j].name.e) say("(empty)");
        else say_str(gs[j].name);
        say(": ");
        if (c >= 0) {
            say_sum(gs[j].s);
            say(" (");
        }
        say_int(gs[j].rows);
        say(gs[j].rows == 1 ? " row" : " rows");
        if (c >= 0) put(')');
    }
    if (bad) say("\n(cells that are not numbers, left out: "), say_int(bad), put(')');
}
