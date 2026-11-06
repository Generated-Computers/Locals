/* Checks the server asks of the computer, in calls it writes itself:

   check_call: a call of one of the client's tools, before the client gets it. Its schema, then the call as Qwen
   wrote it (the schema in the check's own function, not a parameter of it: the computer's text buffer holds 80
   tokens of a call, and a check that fits is one run):
     \n<function=check_call>\nSCHEMA\n</function>\n<function=NAME>\n<parameter=...
   SCHEMA's lines: the called tool's parameters, a line each: its name, the JSON types it allows as the server reads
   them (oai.coerce's; a letter each: s string, i integer, n number, b boolean, o object, a array, z null; none: any),
   r or o (required or not) and its enum's values; or, for a tool the client has not got, "tools" and the client's
   tools' names. Fields separated by tabs, a list's items by 0x1f. The response: ok, or what is wrong.

   copy_check: a number Qwen wrote in its answer after the computer's results (one a line): ok, or "fix" and the
   result it miscopied, written as Qwen wrote numbers (thousands separators, a minus sign). */
#include "tools.h"

#define US 0x1f

static str field(str *line) { /* the next tab-separated field of *line */
    const char *q = line->p;
    while (q < line->e && *q != '\t') q++;
    str f = {line->p, q};
    line->p = q < line->e ? q + 1 : q;
    return f;
}

static str item(str *list) { /* the next 0x1f-separated item */
    const char *q = list->p;
    while (q < list->e && *q != US) q++;
    str f = {list->p, q};
    list->p = q < list->e ? q + 1 : q;
    return f;
}

static int problems;

static void problem(void) { say(problems++ ? "; " : "error ("); }

static int is_int(str s) { /* as Python's int() reads it (1_000 too), or a number with no fraction (5.0) */
    s = trim(s);
    const char *p = s.p;
    if (p < s.e && (*p == '-' || *p == '+')) p++;
    int digits = 0, under = 0;
    for (; p < s.e; p++) {
        if (digit(*p)) digits++, under = 0;
        else if (*p == '_' && digits && !under) under = 1;
        else break;
    }
    if (p == s.e && digits && !under) return 1;
    num v;
    if (!read_num(s, &v, 0)) return 0;
    i64 t = 1;
    for (int i = 0; i < v.frac; i++) t *= 10;
    return v.v % t == 0;
}

static int is_number(str s) { /* as Python's float() reads it, finite */
    s = trim(s);
    const char *p = s.p;
    if (p < s.e && (*p == '-' || *p == '+')) p++;
    int digits = 0;
    while (p < s.e && digit(*p)) p++, digits++;
    if (p < s.e && *p == '.') {
        p++;
        while (p < s.e && digit(*p)) p++, digits++;
    }
    if (!digits) return 0;
    if (p < s.e && (*p == 'e' || *p == 'E')) {
        p++;
        if (p < s.e && (*p == '-' || *p == '+')) p++;
        if (p >= s.e || !digit(*p)) return 0;
        while (p < s.e && digit(*p)) p++;
    }
    return p == s.e;
}

static int word_in(str s, const char *const *words) {
    for (; *words; words++)
        if (str_ieq(s, (str){*words, *words + strlen(*words)})) return 1;
    return 0;
}

/* the value's JSON, if it is an object (want '{') or array ('['): 0, or what is wrong */
static const char *json_kind(str s, char want) {
    s = trim(s);
    jread r = {0};
    const char *v = json_value(&r, s.p, s.e);
    if (!v) return r.err;
    if (json_space(v, s.e) < s.e) return "more text after the JSON value";
    return *json_space(s.p, s.e) == want ? 0 : want == '{' ? "not a JSON object" : "not a JSON array";
}

static int type_ok(str value, str types, const char **why) {
    static const char *const BOOL[] = {"true", "false", "yes", "no", "1", "0", 0};
    *why = 0;
    if (types.p == types.e) return 1;
    for (const char *t = types.p; t < types.e; t++) {
        if (*t == 's') return 1;
        if (*t == 'z' && str_is(trim(value), "null")) return 1;
        if (*t == 'i' && is_int(value)) return 1;
        if (*t == 'n' && (is_number(value) || is_int(value))) return 1;
        if (*t == 'b' && word_in(trim(value), BOOL)) return 1;
        if (*t == 'o' || *t == 'a') {
            const char *e = json_kind(value, *t == 'o' ? '{' : '[');
            if (!e) return 1;
            *why = e;
        }
    }
    return 0;
}

void check_call(void) {
    const char *at = find(text, text_end, "<function=check_call>"), *end = at ? find(at, text_end, "</function>") : 0;
    if (!end) fail("no schema");
    str schema = {at + 21, end};
    if (schema.p < schema.e && *schema.p == '\n') schema.p++;
    if (schema.e > schema.p && schema.e[-1] == '\n') schema.e--;
    /* the call after the check's own </function> */
    const char *c = find(schema.e, text_end, "<function=");
    if (!c) fail("no call to check");
    const char *ne = find_byte(c, text_end, '>');
    if (!ne) fail("no call to check");
    str name = {c + 10, ne};
    str tools = {0, 0};
    problems = 0;
    /* the call's parameters */
    enum { MOST = 32 };
    str pn[MOST], pv[MOST];
    int np = 0;
    const char *p = ne + 1;
    for (;;) {
        p = json_space(p, text_end);
        if (text_end - p < 11 || memcmp(p, "<parameter=", 11)) break;
        const char *n = p + 11, *n_e = find_byte(n, text_end, '>');
        if (!n_e) break;
        const char *v = n_e + 1;
        if (v < text_end && *v == '\n') v++;
        const char *v_e = find(v, text_end, "</parameter>");
        if (!v_e) break;
        p = v_e + 12;
        if (v_e > v && v_e[-1] == '\n') v_e--;
        str k = trim((str){n, n_e});
        for (int i = 0; i < np; i++)
            if (str_ieq(pn[i], k) && pn[i].e - pn[i].p == k.e - k.p && !memcmp(pn[i].p, k.p, k.e - k.p)) {
                problem();
                say("the parameter ");
                say_str(k);
                say(" is given twice");
            }
        if (np < MOST) pn[np] = k, pv[np] = (str){v, v_e}, np++;
    }
    /* the schema's lines: the tools' names first (a tool of another name is all that is wrong with its call) */
    int known = 0, n_params = 0;
    u8 seen[MOST] = {0};
    for (str s = schema; s.p < s.e;) {
        const char *nl = find_byte(s.p, s.e, '\n');
        str line = {s.p, nl ? nl : s.e};
        s.p = nl ? nl + 1 : s.e;
        if (!str_is(field(&line), "tools")) continue;
        tools = line;
        for (str t = line; t.p < t.e;) {
            str x = item(&t);
            if (x.e - x.p == name.e - name.p && !memcmp(x.p, name.p, x.e - x.p)) known = 1;
        }
    }
    for (str s = schema; s.p < s.e && !have(tools);) {
        const char *nl = find_byte(s.p, s.e, '\n');
        str line = {s.p, nl ? nl : s.e};
        s.p = nl ? nl + 1 : s.e;
        if (line.p == line.e) continue;
        n_params++;
        str pname = field(&line), types = field(&line), req = field(&line), values = field(&line);
        int at = -1;
        for (int i = 0; i < np; i++)
            if (pn[i].e - pn[i].p == pname.e - pname.p && !memcmp(pn[i].p, pname.p, pname.e - pname.p)) at = i;
        if (at < 0) {
            if (str_is(req, "r")) {
                problem();
                say("the required parameter ");
                say_str(pname);
                say(" is missing");
            }
            continue;
        }
        seen[at] = 1;
        str v = pv[at];
        const char *why;
        if (values.p < values.e) {
            int ok = 0;
            str tv = trim(v);
            for (str t = values; t.p < t.e && !ok;) { /* (as written, or without the spaces around it) */
                str x = item(&t);
                ok = (x.e - x.p == v.e - v.p && !memcmp(x.p, v.p, x.e - x.p)) ||
                     (x.e - x.p == tv.e - tv.p && !memcmp(x.p, tv.p, x.e - x.p));
            }
            if (!ok) {
                problem();
                say_str(pname);
                say(" must be one of: ");
                int i = 0;
                for (str t = values; t.p < t.e;) {
                    if (i++) say(", ");
                    say_str(item(&t));
                }
                say(" (it is ");
                say_str(v);
                put(')');
            }
        } else if (!type_ok(v, types, &why)) {
            problem();
            say_str(pname);
            say(" must be ");
            for (const char *t = types.p; t < types.e; t++) {
                if (t > types.p) say(" or ");
                say(*t == 'i' ? "an integer" : *t == 'o' ? "a JSON object" : *t == 'a' ? "a JSON array" : *t == 'n' ? "a number"
                    : *t == 'b' ? "true or false" : *t == 'z' ? "null" : "a string");
            }
            if (why) say(" ("), say(why), put(')');
            else {
                say(" (it is ");
                say_str(v);
                put(')');
            }
        }
    }
    if (have(tools) && !known) {
        problem();
        say("no tool named ");
        say_str(name);
        say("; the tools: ");
        int i = 0;
        for (str t = tools; t.p < t.e;) {
            if (i++) say(", ");
            say_str(item(&t));
        }
    } else if (n_params)
        for (int i = 0; i < np && i < MOST; i++)
            if (!seen[i]) {
                problem();
                say("no parameter named ");
                say_str(pn[i]);
                say(" (the parameters: ");
                int j = 0;
                for (str s = schema; s.p < s.e;) {
                    const char *nl = find_byte(s.p, s.e, '\n');
                    str line = {s.p, nl ? nl : s.e};
                    s.p = nl ? nl + 1 : s.e;
                    if (line.p == line.e) continue;
                    if (j++) say(", ");
                    say_str(field(&line));
                }
                put(')');
            }
    if (problems) put(')');
    else say("ok");
}

/* ---- copy_check */
typedef struct {
    int neg, frac; /* frac: digits after the point */
    char sep;      /* the thousands separator written (0: none; ' ' for a space or a narrow space) */
    str sep_text;  /* as written */
    int minus2212; /* the minus written as U+2212 */
    char d[64];    /* the digits, without leading zeros before the point */
    int n;
} written;

static int read_written(str s, written *w) {
    s = trim(s);
    const char *p = s.p, *e = s.e;
    w->neg = w->frac = w->n = w->minus2212 = 0;
    w->sep = 0;
    w->sep_text = (str){0, 0};
    if (p < e && *p == '-') w->neg = 1, p++;
    else if (e - p >= 3 && (u8)p[0] == 0xe2 && (u8)p[1] == 0x88 && (u8)p[2] == 0x92) w->neg = w->minus2212 = 1, p += 3;
    else if (p < e && *p == '+') p++;
    int point = 0, lead = 1;
    for (; p < e; p++) {
        if (digit(*p)) {
            if (lead && *p == '0' && !point && p + 1 < e && digit(p[1])) continue;
            lead = 0;
            if (w->n == 64) return 0;
            w->d[w->n++] = *p;
            if (point) w->frac++;
        } else if (*p == '.' && !point && p + 1 < e && digit(p[1]))
            point = 1;
        else if (!point && (*p == ',' || *p == '_' || *p == '\'' || *p == ' ') && p + 1 < e && digit(p[1]) && p > s.p) {
            if (!w->sep) w->sep = *p, w->sep_text = (str){p, p + 1};
        } else if (!point && e - p >= 4 && (u8)p[0] == 0xe2 && (u8)p[1] == 0x80 && (u8)p[2] == 0xaf && digit(p[3])) {
            if (!w->sep) w->sep = ' ', w->sep_text = (str){p, p + 3}; /* (a narrow no-break space) */
            p += 2;
        } else
            return 0;
    }
    while (w->frac && w->d[w->n - 1] == '0') w->n--, w->frac--; /* (2.50 is 2.5) */
    return w->n > 0;
}

static int same(const written *a, const written *b) {
    return a->neg == b->neg && a->n == b->n && a->frac == b->frac && !memcmp(a->d, b->d, a->n);
}

/* r rounded to f decimals (half up) is w */
static int rounds_to(const written *r, const written *w) {
    if (w->frac >= r->frac || r->neg != w->neg) return 0;
    int keep = r->n - (r->frac - w->frac);
    char d[65];
    memcpy(d + 1, r->d, keep);
    d[0] = '0';
    if (r->d[keep] >= '5') {
        int i = keep;
        while (d[i] == '9') d[i--] = '0';
        d[i]++;
    }
    const char *q = d;
    int n = keep + 1;
    if (*q == '0' && n > 1) q++, n--;
    written x = *w;
    int f = w->frac;
    while (f && q[n - 1] == '0') n--, f--; /* (trailing zeros of the rounded fraction) */
    x.n = n, x.frac = f;
    memcpy(x.d, q, n);
    return same(&x, w);
}

void copy_check(void) {
    str res = param("results"), wr = param("written");
    if (!have(res) || !have(wr)) fail("missing argument: results and written");
    written w, best_r;
    if (!read_written(wr, &w)) {
        say("ok");
        return;
    }
    int best = -1;
    for (str s = res; s.p < s.e;) {
        const char *nl = find_byte(s.p, s.e, '\n');
        str line = trim((str){s.p, nl ? nl : s.e});
        s.p = nl ? nl + 1 : s.e;
        written r;
        if (!read_written(line, &r)) continue;
        if (same(&r, &w) || rounds_to(&r, &w)) {
            say("ok");
            return;
        }
        if (r.neg != w.neg || r.n < 6 || r.frac != w.frac) continue;
        int dn = r.n - w.n;
        if (dn < -6 || dn > 6) continue;
        int m = r.n < w.n ? r.n : w.n, lcp = 0, lcs = 0;
        while (lcp < m && r.d[lcp] == w.d[lcp]) lcp++;
        while (lcs < m - lcp && r.d[r.n - 1 - lcs] == w.d[w.n - 1 - lcs]) lcs++;
        if (lcp < 3 || lcp + lcs < m - 2) continue; /* (a different number) */
        if (lcp + lcs > best) best = lcp + lcs, best_r = r;
    }
    if (best < 0) {
        say("ok");
        return;
    }
    /* the result, written as Qwen wrote the number */
    say("fix\n");
    if (best_r.neg) say(w.minus2212 ? "\xe2\x88\x92" : "-");
    int whole = best_r.n - best_r.frac;
    if (whole == 0) put('0');
    for (int i = 0; i < whole; i++) {
        if (i && w.sep && (whole - i) % 3 == 0) say_str(w.sep_text);
        put(best_r.d[i]);
    }
    if (best_r.frac) {
        put('.');
        for (int i = whole; i < best_r.n; i++) put(best_r.d[i]);
    }
}
