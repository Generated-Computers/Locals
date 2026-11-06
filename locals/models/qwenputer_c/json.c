/* JSON (RFC 8259): json_check, json_format, json_get (a path, * for each item) and json_count. */
#include "tools.h"

const char *json_space(const char *p, const char *e) {
    while (p < e && space(*p))
        p += !((size_t)p & 3) && p + 4 <= e && *(const u32 *)p == 0x20202020u ? 4 : 1; /* (indentation) */
    return p;
}

static void note(jread *r, int slot, const char *s, const char *end) { /* an index entry: s and end, from base */
    if (slot < 0) {
        if (!r->ix || r->nix == r->cap) return;
        slot = r->nix++;
    }
    r->ix[slot] = (u32)(s - r->base) << 16 | (u32)(end - r->base);
}

static const char *bad(jread *r, const char *at, const char *msg) {
    r->at = at;
    r->err = msg;
    return 0;
}

static int hex(int c) {
    return digit(c) ? c - '0' : (c | 32) >= 'a' && (c | 32) <= 'f' ? (c | 32) - 'a' + 10 : -1;
}

static int word(int c) { return digit(c) || ((c | 32) >= 'a' && (c | 32) <= 'z') || c == '_'; }

static const char *jstring(jread *r, const char *p, const char *e) {
    const char *s = p++;
    while (p < e) {
        while (((size_t)p & 3) && p < e && (u8)*p >= 0x20 && *p != '"' && *p != '\\') p++;
        if (!((size_t)p & 3)) /* 4 bytes at a time while none is a quote, a backslash or a control character */
            for (; p + 4 <= e; p += 4) {
                u32 w = *(const u32 *)p;
                if ((HAS_BYTE(w, 0x22u) | HAS_BYTE(w, 0x5cu) | ((w - 0x20202020u) & ~w & 0x80808080u))) break;
            }
        if (p >= e) break;
        u8 c = *p;
        if (c == '"') {
            if (p - s >= 12) note(r, -1, s, p + 1);
            return p + 1;
        }
        if (c == '\\') {
            if (p + 1 >= e) break;
            c = p[1];
            if (c == 'u') {
                for (int i = 2; i < 6; i++)
                    if (p + i >= e || hex((u8)p[i]) < 0) return bad(r, p, "a \\u escape without 4 hex digits");
                p += 6;
            } else if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't')
                p += 2;
            else
                return bad(r, p, "an unknown escape (\\\\ is a backslash)");
        } else if (c < 0x20)
            return bad(r, p, c == '\n' ? "a line break in a string (write \\n)" : "a control character in a string");
        else
            p++;
    }
    return bad(r, s, "a string without its closing quote");
}

static const char *jnumber(jread *r, const char *p, const char *e) {
    const char *s = p;
    if (*p == '-') p++;
    if (p >= e || !digit(*p)) return bad(r, s, "a minus sign without a number");
    if (*p == '0') {
        if (++p < e && digit(*p)) return bad(r, s, "a number with a leading zero");
    } else
        while (p < e && digit(*p)) p++;
    if (p < e && *p == '.') {
        if (++p >= e || !digit(*p)) return bad(r, p, "a decimal point without digits after it");
        while (p < e && digit(*p)) p++;
    }
    if (p < e && (*p == 'e' || *p == 'E')) {
        if (++p < e && (*p == '+' || *p == '-')) p++;
        if (p >= e || !digit(*p)) return bad(r, p, "an exponent without digits");
        while (p < e && digit(*p)) p++;
    }
    return p;
}

static const char *jword(jread *r, const char *p, const char *e) {
    const char *s = p;
    while (p < e && word((u8)*p)) p++;
    size_t n = p - s;
    if ((n == 4 && !memcmp(s, "true", 4)) || (n == 5 && !memcmp(s, "false", 5)) || (n == 4 && !memcmp(s, "null", 4)))
        return p;
    if ((n == 3 && !memcmp(s, "NaN", 3)) || (n == 8 && !memcmp(s, "Infinity", 8)))
        return bad(r, s, "NaN and Infinity are not JSON numbers");
    return bad(r, s, "an unknown word (JSON has true, false and null, in lower case)");
}

static const char *jlist(jread *r, const char *p, const char *e, int obj) {
    const char *s = p;
    char close = obj ? '}' : ']';
    int slot = r->ix && r->nix < r->cap ? r->nix++ : -1;
    if (++r->depth > 40) return bad(r, p, "nested more than 40 deep");
    p = json_space(p + 1, e);
    if (p < e && *p == close) {
        r->depth--;
        if (slot >= 0) note(r, slot, s, p + 1);
        return p + 1;
    }
    for (;;) {
        p = json_space(p, e);
        if (p >= e) return bad(r, s, obj ? "an object without its closing }" : "an array without its closing ]");
        if (*p == close) return bad(r, p, obj ? "a comma before }" : "a comma before ]");
        if (obj) {
            if (*p != '"') {
                if (*p == '\'') return bad(r, p, "single quotes (JSON strings take double quotes)");
                if (word((u8)*p)) return bad(r, p, "a key without quotes (keys are strings in double quotes)");
                return bad(r, p, "expected a key (a string in double quotes)");
            }
            if (!(p = jstring(r, p, e))) return 0;
            p = json_space(p, e);
            if (p >= e || *p != ':') return bad(r, p, "expected : after a key");
            p++;
        }
        if (!(p = json_value(r, p, e))) return 0;
        p = json_space(p, e);
        if (p >= e) return bad(r, s, obj ? "an object without its closing }" : "an array without its closing ]");
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p == close) {
            r->depth--;
            if (slot >= 0) note(r, slot, s, p + 1);
            return p + 1;
        }
        return bad(r, p, obj ? "expected , or } after a value in an object" : "expected , or ] after an item of an array");
    }
}

const char *json_value(jread *r, const char *p, const char *e) {
    p = json_space(p, e);
    if (p >= e) return bad(r, p, "expected a value, found the end");
    u8 c = *p;
    if (c == '{' || c == '[') return jlist(r, p, e, c == '{');
    if (c == '"') return jstring(r, p, e);
    if (c == '-' || digit(c)) return jnumber(r, p, e);
    if (word(c)) return jword(r, p, e);
    if (c == '\'') return bad(r, p, "single quotes (JSON strings take double quotes)");
    if (c == '/') return bad(r, p, "a comment (JSON has none)");
    if (c == '+' || c == '.') return bad(r, p, "a number that does not start with a digit or a minus sign");
    return bad(r, p, "expected a value (an object, array, string, number, true, false or null)");
}

/* ---- reading the call's JSON: the document [doc, doc_end), checked (an error ends the run) */
static const char *doc, *doc_end;
static u32 *ix;
static int ix_n;

static void where(const char *at) {
    int line = 1, col = 1;
    for (const char *q = doc; q < at; q++) {
        if (*q == '\n') line++, col = 1;
        else if (((u8)*q & 0xc0) != 0x80) col++;
    }
    say(" at line ");
    say_int(line);
    say(", column ");
    say_int(col);
}

static void read_doc(const char *name) {
    str s = param(name);
    if (!have(s)) fail("missing argument: text, the JSON");
    doc = s.p, doc_end = s.e;
    if (doc_end - doc >= 3 && (u8)doc[0] == 0xef && (u8)doc[1] == 0xbb && (u8)doc[2] == 0xbf) doc += 3;
    jread r = {0};
    ix_n = 0;
    if (doc_end - doc < 0xffff) { /* an index of the long values, to skip them at once */
        r.cap = (int)((doc_end - doc) / 4) + 8;
        r.ix = ix = alloc(4 * (size_t)r.cap);
        r.base = doc;
    }
    const char *v = json_value(&r, doc, doc_end);
    ix_n = r.nix;
    if (v && json_space(v, doc_end) < doc_end) bad(&r, json_space(v, doc_end), "more text after the JSON value");
    if (r.err) {
        say("error (not valid JSON: ");
        say(r.err);
        where(r.at);
        put(')');
        done();
    }
}

/* the members of an object or array (at its { or [, checked): each value's start, and an object's key (at its ") */
typedef struct {
    const char *p, *e;
    int obj;
} jiter;

static void jit(jiter *it, const char *v, const char *e) {
    it->p = json_space(v + 1, e);
    it->e = e;
    it->obj = *v == '{';
}

static const char *jskip(const char *p, const char *e) { /* (the end of a checked value) */
    if (*p == '"') { /* (a short string, not in the index: its end at once) */
        const char *q = p + 1, *lim = e - p > 12 ? p + 12 : e;
        while (q < lim && *q != '"' && *q != '\\') q++;
        if (q < lim && *q == '"') return q + 1;
    }
    if (ix_n && (*p == '{' || *p == '[' || *p == '"')) {
        u32 off = (u32)(p - doc);
        int lo = 0, hi = ix_n - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            u32 s = ix[mid] >> 16;
            if (s == off) return doc + (ix[mid] & 0xffff);
            if (s < off) lo = mid + 1;
            else hi = mid - 1;
        }
    }
    jread r = {0};
    return json_value(&r, p, e);
}

static int jnext(jiter *it, const char **key, const char **val) {
    const char *p = json_space(it->p, it->e);
    if (*p == ',') p = json_space(p + 1, it->e);
    if (*p == '}' || *p == ']') return 0;
    if (it->obj) {
        *key = p;
        p = json_space(jskip(p, it->e), it->e) + 1;
    }
    *val = json_space(p, it->e);
    it->p = jskip(*val, it->e);
    return 1;
}

/* the next character of a JSON string (at q, after its opening quote) as UTF-8 bytes in b: their count, 0 at the end */
static int jchar(const char **q, u8 *b) {
    const char *p = *q;
    if (*p == '"') return 0;
    if (*p != '\\') {
        b[0] = *p;
        *q = p + 1;
        return 1;
    }
    char c = p[1];
    *q = p + 2;
    if (c != 'u') {
        b[0] = c == 'b' ? 8 : c == 'f' ? 12 : c == 'n' ? 10 : c == 'r' ? 13 : c == 't' ? 9 : c;
        return 1;
    }
    u32 u = 0;
    for (int i = 2; i < 6; i++) u = u << 4 | hex((u8)p[i]);
    *q = p + 6;
    if (u >= 0xd800 && u < 0xdc00 && p[6] == '\\' && p[7] == 'u') {
        u32 lo = 0;
        for (int i = 8; i < 12; i++) lo = lo << 4 | hex((u8)p[i]);
        if (lo >= 0xdc00 && lo < 0xe000) {
            u = 0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00);
            *q = p + 12;
        }
    }
    if (u < 0x80) return b[0] = u, 1;
    if (u < 0x800) return b[0] = 0xc0 | u >> 6, b[1] = 0x80 | (u & 63), 2;
    if (u < 0x10000) return b[0] = 0xe0 | u >> 12, b[1] = 0x80 | (u >> 6 & 63), b[2] = 0x80 | (u & 63), 3;
    b[0] = 0xf0 | u >> 18, b[1] = 0x80 | (u >> 12 & 63), b[2] = 0x80 | (u >> 6 & 63), b[3] = 0x80 | (u & 63);
    return 4;
}

static void say_jstring(const char *q) { /* the string at q (its opening quote), decoded */
    u8 b[4];
    q++;
    for (int n; (n = jchar(&q, b));)
        for (int i = 0; i < n; i++) put(b[i]);
}

static int key_is(const char *q, str k) {
    u8 b[4];
    const char *p = k.p;
    q++;
    for (int n; (n = jchar(&q, b));)
        for (int i = 0; i < n; i++, p++)
            if (p >= k.e || (u8)*p != b[i]) return 0;
    return p == k.e;
}

/* a value's JSON with no spaces outside strings (pretty: indented 2 spaces a level, one member a line) */
static void say_json(const char *v, const char *e, int pretty) {
    const char *end = jskip(v, e);
    int level = 0;
    for (const char *p = v; p < end; p++) {
        u8 c = *p;
        if (c == '"') {
            const char *s = p;
            jread r = {0};
            p = jstring(&r, p, end) - 1;
            say_str((str){s, p + 1});
        } else if (space(c))
            continue;
        else if (pretty && (c == '{' || c == '[')) {
            const char *n = json_space(p + 1, end);
            if (*n == '}' || *n == ']') {
                put(c);
                put(*n);
                p = n;
                continue;
            }
            put(c);
            put('\n');
            level++;
            for (int i = 0; i < 2 * level; i++) put(' ');
        } else if (pretty && (c == '}' || c == ']')) {
            put('\n');
            level--;
            for (int i = 0; i < 2 * level; i++) put(' ');
            put(c);
        } else if (pretty && c == ',') {
            put(',');
            put('\n');
            for (int i = 0; i < 2 * level; i++) put(' ');
        } else if (pretty && c == ':') {
            put(':');
            put(' ');
        } else
            put(c);
    }
}

static void say_value(const char *v, const char *e) {
    if (*v == '"') say_jstring(v);
    else say_json(v, e, 0);
}

static const char *kind(const char *v) {
    return *v == '{' ? "an object" : *v == '[' ? "an array" : *v == '"' ? "a string" : *v == 't' || *v == 'f' ? "true or false"
           : *v == 'n' ? "null" : "a number";
}

static int members(const char *v, const char *e) {
    jiter it;
    const char *k, *x;
    int n = 0;
    jit(&it, v, e);
    while (jnext(&it, &k, &x)) n++;
    return n;
}

void json_check(void) {
    read_doc("text");
    const char *v = json_space(doc, doc_end);
    say("valid JSON: ");
    say(kind(v));
    if (*v == '{' || *v == '[') {
        int n = members(v, doc_end);
        say(*v == '{' ? " with " : " of ");
        say_int(n);
        say(*v == '{' ? (n == 1 ? " key" : " keys") : (n == 1 ? " item" : " items"));
    }
}

void json_format(void) {
    read_doc("text");
    str style = trim(param("style"));
    int pretty = !(have(style) && (str_is(style, "compact") || str_is(style, "minified") || str_is(style, "minify")));
    say_json(json_space(doc, doc_end), doc_end, pretty);
}

/* ---- paths: a.b[2].c, $.a.b, a["x y"], items[*].price, a/b/2 (a JSON pointer) */
enum { KEY, INDEX, ALL };
typedef struct {
    int type;
    str key;
    int index;
    const char *end; /* where the segment ends in the path's text */
} seg;

static int read_path(str p, seg *s, int max) {
    const char *q = p.p, *e = p.e;
    int n = 0;
    if (q < e && *q == '$') q++;
    int pointer = q < e && *q == '/';
    while (q < e) {
        if (n == max) fail("a path of more than 32 steps");
        seg *g = &s[n];
        if (pointer) {
            const char *k = ++q;
            while (q < e && *q != '/') q++;
            g->type = KEY, g->key = (str){k, q};
        } else if (*q == '.') {
            q++;
            continue;
        } else if (*q == '[') {
            q++;
            if (q < e && (*q == '"' || *q == '\'')) {
                char quote = *q;
                const char *k = ++q;
                while (q < e && *q != quote) q++;
                g->type = KEY, g->key = (str){k, q};
                if (q < e) q++;
            } else if (q < e && *q == '*') {
                g->type = ALL;
                q++;
            } else {
                const char *k = q;
                while (q < e && *q != ']') q++;
                g->type = KEY, g->key = trim((str){k, q});
            }
            if (q >= e || *q != ']') fail("a path's [ without its ]");
            q++;
        } else {
            const char *k = q;
            while (q < e && *q != '.' && *q != '[') q++;
            g->type = KEY, g->key = (str){k, q};
            if (str_is(g->key, "*")) g->type = ALL;
        }
        if (g->type == KEY) { /* digits (or -1, the last) index an array */
            const char *d = g->key.p;
            int neg = d < g->key.e && *d == '-', v = 0, ok = g->key.e > d + neg;
            for (d += neg; d < g->key.e && ok; d++)
                ok = digit(*d) && v < 100000000, v = v * 10 + (*d - '0');
            g->index = ok ? (neg ? -v : v) : 1 << 30;
        }
        g->end = q;
        n++;
    }
    return n;
}

enum { FOUND, NO_KEY, PAST_END, NOT_LIST };
static int found, miss;
static const char *miss_at; /* the value where a path without * stopped, or the first value found */
static const seg *miss_seg;

static void walk(const char *v, const char *e, seg *s, int i, int n, int show) {
    if (i == n) {
        if (show) {
            if (found) put('\n');
            say_value(v, e);
        } else if (!found)
            miss_at = v;
        found++;
        return;
    }
    const seg *g = &s[i];
    jiter it;
    const char *k, *x;
    int why = NOT_LIST;
    if (*v == '{' && (g->type == KEY || g->type == ALL)) {
        jit(&it, v, e);
        while (jnext(&it, &k, &x))
            if (g->type == ALL || key_is(k, g->key)) walk(x, e, s, i + 1, n, show);
        why = NO_KEY;
    } else if (*v == '[' && (g->type == ALL || g->index != 1 << 30)) {
        int want = g->index < 0 ? members(v, e) + g->index : g->index, j = 0; /* (counted only from the end) */
        jit(&it, v, e);
        while (jnext(&it, &k, &x)) {
            if (g->type == ALL || j == want) walk(x, e, s, i + 1, n, show);
            j++;
        }
        why = PAST_END;
    }
    if (!found && !miss) miss = why, miss_at = v, miss_seg = g;
}

static void say_keys(const char *v, int most) {
    jiter it;
    const char *k, *x;
    int j = 0;
    jit(&it, v, doc_end);
    while (jnext(&it, &k, &x)) {
        if (j == most) {
            say(", ...");
            break;
        }
        if (j++) say(", ");
        say_jstring(k);
    }
}

/* the values at the call's path (printed if `show`); the first of them. Nothing there ends the run with why. */
static const char *at_path(int show) {
    seg s[32];
    str path = trim(param("path"));
    int n = have(path) ? read_path(path, s, 32) : 0, all = 0, i = 0;
    found = 0, miss = FOUND, miss_at = 0, miss_seg = 0;
    walk(json_space(doc, doc_end), doc_end, s, 0, n, show);
    if (found) return miss_at;
    for (int j = 0; j < n; j++) {
        all |= s[j].type == ALL;
        if (&s[j] == miss_seg) i = j;
    }
    if (all || miss == FOUND) fail("no value at that path");
    str before = {path.p, i ? s[i - 1].end : path.p};
    say("error (");
    if (miss == NO_KEY) {
        say("no key \"");
        say_str(miss_seg->key);
        say("\" in the object");
        if (i) say(" at "), say_str(before);
        say("; its keys: ");
        if (members(miss_at, doc_end)) say_keys(miss_at, 40);
        else say("none");
    } else if (miss == PAST_END) {
        if (i) say_str(before);
        else say("the array");
        say(" has ");
        int c = members(miss_at, doc_end);
        say_int(c);
        say(c == 1 ? " item (they count from 0)" : " items (they count from 0)");
    } else {
        if (i) say_str(before);
        else say("the JSON");
        say(" is ");
        say(kind(miss_at));
        if (*miss_at == '{') say(": use a key, not a number");
        else if (*miss_at == '[') say(": use a number (from 0) or *, not a key");
        else say(", which has no keys or items");
    }
    put(')');
    done();
}

void json_get(void) {
    read_doc("text");
    at_path(1);
}

void json_count(void) {
    read_doc("text");
    const char *v = at_path(0);
    if (found > 1) fail("the path names more than one value: count one of them");
    if (*v != '{' && *v != '[') {
        say("error (that is ");
        say(kind(v));
        say(", not an object or array)");
        done();
    }
    int n = members(v, doc_end);
    say_int(n);
    if (*v == '[') {
        say(n == 1 ? " item" : " items");
        return;
    }
    say(n == 1 ? " key" : " keys");
    if (!n) return;
    say(": ");
    say_keys(v, 100);
}
