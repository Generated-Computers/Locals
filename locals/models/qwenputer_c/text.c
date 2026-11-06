/* Text: replace_text, change_case, text_part, sort_lines, unique_items, word_frequency, is_palindrome, is_anagram,
   validate, encode, decode, edit_distance. Letters' case is ASCII's; a character is a UTF-8 one. */
#include "tools.h"

static str need(const char *name) {
    str s = param(name);
    if (!have(s)) {
        say("error (missing argument: ");
        say(name);
        put(')');
        done();
    }
    return s;
}

static int utf8_len(u8 c) { return c < 0x80 ? 1 : c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : c >= 0xc0 ? 2 : 1; }
static int alpha(int c) { return (unsigned)((c | 32) - 'a') < 26; }
static int alnum(int c) { return alpha(c) || digit(c); }
static str lit(const char *z) { return (str){z, z + strlen(z)}; }

static u32 code_point(const char *p, const char *e, int *n) { /* a UTF-8 character's */
    u8 c = *p;
    int k = utf8_len(c);
    if (p + k > e) k = 1;
    u32 u = k == 1 ? c : c & (0x3f >> (k - 1));
    for (int i = 1; i < k; i++) u = u << 6 | ((u8)p[i] & 0x3f);
    *n = k;
    return u;
}

void replace_text(void) {
    str t = need("text"), f = need("find"), w = param("replace");
    if (!have(w)) w = param("with");
    if (f.p == f.e) fail("find: the text to replace, not empty");
    int n = 0;
    for (const char *p = t.p; p + (f.e - f.p) <= t.e;)
        if (*p == *f.p && !memcmp(p, f.p, f.e - f.p)) n++, p += f.e - f.p;
        else p++;
    say_int(n);
    say(n == 1 ? " replacement:\n" : " replacements:\n");
    for (const char *p = t.p; p < t.e;) {
        if (*p == *f.p && p + (f.e - f.p) <= t.e && !memcmp(p, f.p, f.e - f.p)) {
            if (have(w)) say_str(w);
            p += f.e - f.p;
        } else
            put((u8)*p++);
    }
}

void change_case(void) {
    str t = need("text"), c = trim(need("case"));
    static const char *const CASES[] = {"upper", "lower", "title", "sentence", "snake", "camel", "kebab", "pascal", "constant", 0};
    int k = -1;
    for (int i = 0; CASES[i]; i++) {
        size_t n = strlen(CASES[i]);
        if ((size_t)(c.e - c.p) >= n) {
            str head = {c.p, c.p + n};
            if (str_ieq(head, (str){CASES[i], CASES[i] + n})) k = i;
        }
    }
    if (k < 0) fail("case: upper, lower, title, sentence, snake, camel, kebab, pascal or constant");
    if (k <= 3) {
        int start = 1;
        for (const char *p = t.p; p < t.e; p++) {
            int ch = (u8)*p;
            int up = k == 0 || ((k == 2 || k == 3) && start && alpha(ch));
            put(up ? ((unsigned)(ch - 'a') < 26 ? ch - 32 : ch) : (k == 1 || alpha(ch)) ? lower(ch) : ch);
            if (k == 2) start = !alnum(ch) && ch != '\'';
            else if (k == 3) start = (start && !alpha(ch)) || ch == '.' || ch == '!' || ch == '?';
        }
        return;
    }
    /* words (runs of letters and digits; a lower letter then an upper one starts a new word), joined */
    int first = 1;
    const char *p = t.p;
    while (p < t.e) {
        while (p < t.e && !alnum((u8)*p)) p++;
        if (p >= t.e) break;
        const char *s = p;
        p++;
        while (p < t.e && alnum((u8)*p) && !((unsigned)(*p - 'A') < 26 && (unsigned)(p[-1] - 'a') < 26)) p++;
        if (!first && (k == 4 || k == 8)) put('_');
        if (!first && k == 6) put('-');
        for (const char *q = s; q < p; q++) {
            int ch = (u8)*q, up = k == 8 || ((k == 5 && !first) || k == 7) * (q == s);
            put(up ? ((unsigned)(ch - 'a') < 26 ? ch - 32 : ch) : lower(ch));
        }
        first = 0;
    }
}

/* the pieces of a text: words, lines, characters or sentences */
enum { P_WORD, P_LINE, P_CHAR, P_SENTENCE };

static int pieces(str t, int kind, str *out, int most) {
    int n = 0;
    const char *p = t.p;
    while (p < t.e && n < most) {
        const char *s = p;
        if (kind == P_CHAR) {
            p += utf8_len((u8)*p);
            if (p > t.e) p = t.e;
        } else if (kind == P_LINE) {
            while (p < t.e && *p != '\n') p++;
            const char *e = p > s && p[-1] == '\r' ? p - 1 : p;
            out[n++] = (str){s, e};
            if (p < t.e) p++;
            continue;
        } else if (kind == P_WORD) {
            while (p < t.e && space(*p)) p++;
            s = p;
            while (p < t.e && !space(*p)) p++;
            if (s == p) break;
        } else {
            while (p < t.e && space(*p)) p++;
            s = p;
            while (p < t.e && !((*p == '.' || *p == '!' || *p == '?') && (p + 1 >= t.e || space(p[1])))) p++;
            if (p < t.e) p++;
            if (s == p) break;
        }
        out[n++] = (str){s, p};
    }
    return n;
}

void text_part(void) {
    str t = need("text"), k = trim(need("part")), num_ = trim(need("number"));
    int kind = -1;
    static const char *const K[] = {"word", "line", "char", "sentence"};
    for (int i = 0; i < 4; i++)
        if (k.e - k.p >= 4 && str_ieq((str){k.p, k.p + 4}, (str){K[i], K[i] + 4})) kind = i;
    if (kind < 0) fail("part: word, line, character or sentence");
    /* number: n, -n (from the end) or a-b */
    const char *dash = 0;
    for (const char *q = num_.p + 1; q < num_.e; q++)
        if (*q == '-' || *q == ':') dash = q;
    num a, b;
    if (!read_num(dash ? (str){num_.p, dash} : num_, &a, 0) || a.frac ||
        (dash && (!read_num((str){dash + 1, num_.e}, &b, 0) || b.frac)))
        fail("number: which one, e.g. 3, -1 for the last, or 2-5");
    if (!dash) b = a;
    int most = 1; /* (as many as the text can have of that kind) */
    for (const char *q = t.p; q < t.e; q++)
        most += kind == P_CHAR ? ((u8)*q & 0xc0) != 0x80 : kind == P_LINE ? *q == '\n' : kind == P_WORD ? space(*q)
                : (*q == '.' || *q == '!' || *q == '?');
    str *ps = alloc(sizeof(str) * most);
    int n = pieces(t, kind, ps, most);
    i64 lo = a.v < 0 ? n + a.v + 1 : a.v, hi = b.v < 0 ? n + b.v + 1 : b.v;
    if (lo < 1 || hi > n || lo > hi) {
        say("error (the text has ");
        say_int(n);
        say(n == 1 ? " " : " ");
        say(K[kind]);
        say(kind == P_CHAR ? (n == 1 ? "acter" : "acters") : n == 1 ? "" : "s");
        say(": they count from 1)");
        done();
    }
    if (kind == P_CHAR) {
        say_str((str){ps[lo - 1].p, ps[hi - 1].e});
        return;
    }
    for (i64 i = lo; i <= hi; i++) {
        if (i > lo) put(kind == P_LINE ? '\n' : ' ');
        say_str(ps[i - 1]);
    }
}

/* items: one a line, or comma-separated on one line */
static int items_of(str t, str **list) {
    t = trim(t);
    int lines = find_byte(t.p, t.e, '\n') != 0, most = 1;
    char sep = lines ? '\n' : ',';
    for (const char *q = t.p; q < t.e; q++) most += *q == sep;
    str *out = *list = alloc(sizeof(str) * most);
    int n = 0;
    for (const char *p = t.p; p <= t.e;) {
        const char *q = p;
        while (q < t.e && *q != sep) q++;
        str x = trim((str){p, q});
        if (x.p < x.e) out[n++] = x;
        p = q + 1;
    }
    return n;
}

static u32 ihash(str s) { /* (letter case ignored) */
    u32 h = 2166136261u;
    for (const char *p = s.p; p < s.e; p++) h = (h ^ (u32)lower((u8)*p)) * 16777619u;
    return h;
}

static u32 prefix_key(str s) { /* the first 4 bytes, letters in lower case, as a number that sorts as they do */
    u32 k = 0;
    for (int i = 0; i < 4; i++) k = k << 8 | (s.p + i < s.e ? (u32)lower((u8)s.p[i]) : 0);
    return k;
}

static int text_cmp(str a, str b) { /* ASCII letters in either case, then as written */
    const char *p = a.p, *q = b.p;
    for (; p < a.e && q < b.e; p++, q++) {
        int x = lower((u8)*p), y = lower((u8)*q);
        if (x != y) return x < y ? -1 : 1;
    }
    if (p < a.e || q < b.e) return p < a.e ? 1 : -1;
    return memcmp(a.p, b.p, a.e - a.p);
}

void sort_lines(void) {
    str *v;
    int n = items_of(need("text"), &v), lines = find_byte(param("text").p, param("text").e, '\n') != 0;
    str o = trim(param("order"));
    int desc = have(o) && o.p < o.e && (lower((u8)*o.p) == 'd' || lower((u8)*o.p) == 'z');
    u32 *k = alloc(sizeof(u32) * (n + 1));
    for (int i = 0; i < n; i++) k[i] = prefix_key(v[i]);
    int *a = alloc(sizeof(int) * (n + 1)), *b = alloc(sizeof(int) * (n + 1)); /* (a stable merge sort of indices) */
    for (int i = 0; i < n; i++) a[i] = i;
    for (int w = 1; w < n; w *= 2) {
        for (int lo = 0; lo < n; lo += 2 * w) {
            int mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n, i = lo, j = mid, m = lo;
            while (i < mid && j < hi) {
                int x = a[j], y = a[i];
                int d = k[x] != k[y] ? (k[x] < k[y] ? -1 : 1) : text_cmp(v[x], v[y]);
                b[m++] = (desc ? d > 0 : d < 0) ? a[j++] : a[i++];
            }
            while (i < mid) b[m++] = a[i++];
            while (j < hi) b[m++] = a[j++];
        }
        int *tmp = a;
        a = b, b = tmp;
    }
    for (int i = 0; i < n; i++) {
        if (i) say(lines ? "\n" : ", ");
        say_str(v[a[i]]);
    }
}

void unique_items(void) {
    str *v;
    str t = need("text");
    int n = items_of(t, &v), lines = find_byte(t.p, t.e, '\n') != 0, k = 0;
    u32 size = 16;
    while (size < 2u * (u32)n + 2) size *= 2;
    int *table = alloc(sizeof(int) * size); /* (the items kept, by hash: open addressing, index + 1) */
    memset(table, 0, sizeof(int) * size);
    for (int i = 0; i < n; i++) {
        u32 h = ihash(v[i]) & (size - 1);
        int seen = 0;
        for (; table[h] && !seen; h = (h + 1) & (size - 1)) seen = str_ieq(v[table[h] - 1], v[i]);
        if (seen) continue;
        v[k] = v[i];
        table[h] = ++k;
    }
    say_int(k);
    say(" unique of ");
    say_int(n);
    say(":\n");
    for (int i = 0; i < k; i++) {
        if (i) say(lines ? "\n" : ", ");
        say_str(v[i]);
    }
}

void word_frequency(void) {
    str t = need("text"), tp = trim(param("top"));
    num top = {10, 0};
    if (have(tp) && tp.p < tp.e && (!read_num(tp, &top, 0) || top.frac || top.v < 1)) fail("top: how many words, e.g. 5");
    typedef struct {
        u16 at, len, n; /* (a word: its place in the text, its length, its count) */
    } wc;
#define WORDCH(c) (alnum((u8)(c)) || (u8)(c) >= 0x80)
    if (t.e - t.p > 65535) fail("a text too long for the computer");
    enum { MOST = 600, SIZE = 1024 }; /* (different words at most; the table's places, a power of 2 above 1.5 MOST) */
    int k = 0, total = 0;
    wc *ws = alloc(sizeof(wc) * (MOST + 1));
    u16 *table = alloc_low(sizeof(u16) * SIZE); /* (the words by hash: open addressing, index + 1) */
    memset(table, 0, sizeof(u16) * SIZE);
    for (const char *p = t.p; p < t.e;) {
        while (p < t.e && !WORDCH(*p)) p++;
        const char *s = p;
        u32 h = 2166136261u; /* (the word's hash as it is read: ihash's) */
        for (; p < t.e && (WORDCH(*p) || (*p == '\'' && p + 1 < t.e && alpha((u8)p[1]))); p++)
            h = (h ^ (u32)lower((u8)*p)) * 16777619u;
        if (s == p) break;
        str w = {s, p};
        total++;
        int j = -1;
        for (h &= SIZE - 1; table[h]; h = (h + 1) & (SIZE - 1)) {
            const wc *x = &ws[table[h] - 1];
            if (x->len == p - s && str_ieq((str){t.p + x->at, t.p + x->at + x->len}, w)) {
                j = table[h] - 1;
                break;
            }
        }
        if (j < 0) {
            if (k == MOST) fail("more than 600 different words");
            ws[k] = (wc){(u16)(s - t.p), (u16)(p - s), 0};
            j = k++;
            table[h] = (u16)k;
        }
        ws[j].n++;
    }
#undef WORDCH
    /* by count, most first, and in the text's order where counts tie (a counting sort: counts are at most `total`) */
    int most = 0;
    for (int i = 0; i < k; i++) most = ws[i].n > most ? ws[i].n : most;
    u16 *at = alloc_low(sizeof(u16) * (most + 2)), *a = alloc_low(sizeof(u16) * (k + 1));
    memset(at, 0, sizeof(u16) * (most + 2));
    for (int i = 0; i < k; i++) at[ws[i].n]++;
    for (int c = most, sum = 0; c >= 1; c--) {
        int here = at[c];
        at[c] = (u16)sum, sum += here;
    }
    for (int i = 0; i < k; i++) a[at[ws[i].n]++] = (u16)i;
    say_int(total);
    say(" words, ");
    say_int(k);
    say(" different");
    for (int i = 0; i < k && i < top.v; i++) {
        const wc *x = &ws[a[i]];
        put('\n');
        for (const char *q = t.p + x->at; q < t.p + x->at + x->len; q++) put(lower((u8)*q));
        say(": ");
        say_int(x->n);
    }
}

void is_palindrome(void) {
    str t = need("text");
    char *k = alloc(t.e - t.p + 1);
    int n = 0;
    for (const char *p = t.p; p < t.e; p++)
        if (alnum((u8)*p)) k[n++] = (char)lower((u8)*p);
    int yes = n > 0;
    for (int i = 0; i < n / 2 && yes; i++) yes = k[i] == k[n - 1 - i];
    say(yes ? "yes" : "no");
    say(" (its letters and digits, case ignored: ");
    for (int i = 0; i < n && i < 200; i++) put(k[i]);
    say(yes ? ", the same backwards)" : ", not the same backwards)");
}

void is_anagram(void) {
    str a = need("first"), b = need("second");
    int ca[36] = {0}, cb[36] = {0};
    for (const char *p = a.p; p < a.e; p++)
        if (alnum((u8)*p)) ca[digit(*p) ? 26 + *p - '0' : lower((u8)*p) - 'a']++;
    for (const char *p = b.p; p < b.e; p++)
        if (alnum((u8)*p)) cb[digit(*p) ? 26 + *p - '0' : lower((u8)*p) - 'a']++;
    int same = 1;
    for (int i = 0; i < 36; i++) same &= ca[i] == cb[i];
    say(same ? "yes: the same letters, as often (case, spaces and punctuation ignored)" : "no: ");
    if (same) return;
    int first = 1;
    for (int i = 0; i < 36; i++)
        if (ca[i] != cb[i]) {
            if (!first) say(", ");
            first = 0;
            put(i < 26 ? 'a' + i : '0' + i - 26);
            put(' ');
            say_int(ca[i]);
            say(" and ");
            say_int(cb[i]);
        }
    say(" times");
}

/* ---- validate: email, ipv4, url, uuid, date, time, hex color */
static int hexd(int c) { return digit(c) || (unsigned)((c | 32) - 'a') < 6; }

void validate(void) {
    str t = trim(need("text")), k = trim(need("kind"));
    const char *why = 0;
    const char *p = t.p, *e = t.e;
#define IS(z) str_ieq(k, lit(z))
    if (IS("email") || IS("e-mail")) {
        const char *at = 0;
        for (const char *q = p; q < e; q++)
            if (*q == '@') why = at ? "more than one @" : why, at = q;
        if (!at) why = "no @";
        else if (at == p) why = "nothing before the @";
        else {
            for (const char *q = p; q < at && !why; q++)
                if (!(alnum((u8)*q) || *q == '.' || *q == '_' || *q == '-' || *q == '+' || *q == '%')) why = "a character an address does not have before the @";
            if (!why && (*p == '.' || at[-1] == '.')) why = "a dot at the start or end of the name";
            const char *dot = 0;
            for (const char *q = at + 1; q < e && !why; q++) {
                if (*q == '.') {
                    if (q == at + 1 || q[-1] == '.') why = "an empty part of the domain";
                    dot = q;
                } else if (!(alnum((u8)*q) || *q == '-'))
                    why = "a character a domain does not have";
            }
            if (!why && !dot) why = "a domain without a dot";
            if (!why && (e - dot - 1 < 2 || e[-1] == '.')) why = "a domain's last part shorter than 2 letters";
        }
    } else if (IS("ipv4") || IS("ip")) {
        int parts = 0;
        for (const char *q = p; q <= e && !why;) {
            const char *s = q;
            int v = 0;
            while (q < e && digit(*q)) v = v * 10 + (*q - '0'), q++;
            if (q == s || q - s > 3) why = "a part that is not 1 to 3 digits";
            else if (v > 255) why = "a part above 255";
            else if (q - s > 1 && *s == '0') why = "a part with a leading zero";
            parts++;
            if (q < e && *q != '.') why = "a character that is not a digit or a dot";
            q++;
        }
        if (!why && parts != 4) why = "not 4 parts";
    } else if (IS("url")) {
        const char *s = p;
        while (s < e && *s != ':') s++;
        if (s == e || e - s < 3 || s[1] != '/' || s[2] != '/') why = "no scheme:// (e.g. https://)";
        else {
            str scheme = {p, s};
            if (!(str_ieq(scheme, lit("http")) || str_ieq(scheme, lit("https")) || str_ieq(scheme, lit("ftp"))))
                why = "a scheme other than http, https or ftp";
            const char *h = s + 3, *q = h;
            while (q < e && *q != '/' && *q != '?' && *q != '#') q++;
            if (!why && q == h) why = "no host";
            for (const char *c = h; c < q && !why; c++)
                if (!(alnum((u8)*c) || *c == '.' || *c == '-' || *c == ':' || *c == '@' || *c == '_')) why = "a character a host does not have";
            for (const char *c = p; c < e && !why; c++)
                if ((u8)*c <= ' ') why = "a space in it";
        }
    } else if (IS("uuid")) {
        if (e - p != 36) why = "not 36 characters";
        for (int i = 0; i < 36 && !why && p + i < e; i++) {
            int dash = i == 8 || i == 13 || i == 18 || i == 23;
            if (dash ? p[i] != '-' : !hexd((u8)p[i])) why = "not the form 8-4-4-4-12 hex digits";
        }
    } else if (IS("date")) {
        int y = 0, m = 0, d = 0;
        if (e - p != 10 || p[4] != '-' || p[7] != '-') why = "not YYYY-MM-DD";
        for (int i = 0; i < 10 && !why; i++)
            if (i != 4 && i != 7 && !digit(p[i])) why = "not YYYY-MM-DD";
        if (!why) {
            y = (p[0] - '0') * 1000 + (p[1] - '0') * 100 + (p[2] - '0') * 10 + (p[3] - '0');
            m = (p[5] - '0') * 10 + (p[6] - '0'), d = (p[8] - '0') * 10 + (p[9] - '0');
            static const int DAYS[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
            int leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
            if (m < 1 || m > 12) why = "no month 1 to 12";
            else if (d < 1 || d > DAYS[m - 1] + (m == 2 && leap)) why = m == 2 && d == 29 ? "February 29 in a year that is not a leap year" : "no such day in that month";
        }
    } else if (IS("time")) {
        int h = -1, mi = -1, se = 0;
        if (e - p == 5 && p[2] == ':') h = (p[0] - '0') * 10 + (p[1] - '0'), mi = (p[3] - '0') * 10 + (p[4] - '0');
        else if (e - p == 8 && p[2] == ':' && p[5] == ':')
            h = (p[0] - '0') * 10 + (p[1] - '0'), mi = (p[3] - '0') * 10 + (p[4] - '0'), se = (p[6] - '0') * 10 + (p[7] - '0');
        for (const char *q = p; q < e; q++)
            if (!digit(*q) && *q != ':') h = -1;
        if (h < 0) why = "not HH:MM or HH:MM:SS";
        else if (h > 23 || mi > 59 || se > 59) why = "an hour above 23 or minutes or seconds above 59";
    } else if (IS("hex color") || IS("color") || IS("colour") || IS("hex colour")) {
        int n = (int)(e - p) - (p < e && *p == '#');
        if (p >= e || *p != '#') why = "no # in front";
        else if (n != 3 && n != 6 && n != 8) why = "not 3, 6 or 8 hex digits";
        for (const char *q = p + 1; q < e && !why; q++)
            if (!hexd((u8)*q)) why = "a character that is not a hex digit";
    } else
        fail("kind: email, ipv4, url, uuid, date, time or hex color");
#undef IS
    if (why) say("not valid: "), say(why);
    else say("valid");
}

/* ---- encodings */
static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static const char *const MORSE[] = {".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..",
                                    "--", "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-",
                                    "-.--", "--..", "-----", ".----", "..---", "...--", "....-", ".....", "-....", "--...",
                                    "---..", "----."};

static int encoding(str k, int *shift) {
    static const char *const E[] = {"base64", "hex", "url", "rot13", "caesar", "morse", "binary", 0};
    k = trim(k);
    for (int i = 0; E[i]; i++)
        if (str_ieq(k, (str){E[i], E[i] + strlen(E[i])})) {
            *shift = 13;
            if (i == 4) {
                num s;
                str sh = trim(param("shift"));
                *shift = have(sh) && read_num(sh, &s, 0) && !s.frac ? (int)(((s.v % 26) + 26) % 26) : 3;
            }
            return i;
        }
    fail("encoding: base64, hex, url, rot13, caesar (with shift), morse or binary");
}

static void caesar(str t, int k) {
    for (const char *p = t.p; p < t.e; p++) {
        int c = (u8)*p;
        if ((unsigned)(c - 'a') < 26) c = 'a' + (c - 'a' + k) % 26;
        else if ((unsigned)(c - 'A') < 26) c = 'A' + (c - 'A' + k) % 26;
        put(c);
    }
}

void encode(void) {
    str t = need("text");
    int shift, e = encoding(need("encoding"), &shift);
    const u8 *p = (const u8 *)t.p, *end = (const u8 *)t.e;
    switch (e) {
    case 0: /* base64 */
        for (; p < end; p += 3) {
            u32 v = (u32)p[0] << 16 | (p + 1 < end ? (u32)p[1] << 8 : 0) | (p + 2 < end ? p[2] : 0);
            put(B64[v >> 18]), put(B64[v >> 12 & 63]);
            put(p + 1 < end ? B64[v >> 6 & 63] : '='), put(p + 2 < end ? B64[v & 63] : '=');
        }
        return;
    case 1:
        for (; p < end; p++) put("0123456789abcdef"[*p >> 4]), put("0123456789abcdef"[*p & 15]);
        return;
    case 2:
        for (; p < end; p++)
            if (alnum(*p) || *p == '-' || *p == '_' || *p == '.' || *p == '~') put(*p);
            else put('%'), put("0123456789ABCDEF"[*p >> 4]), put("0123456789ABCDEF"[*p & 15]);
        return;
    case 3: case 4: caesar(t, shift); return;
    case 5: {
        int first = 1;
        if (end - p > 800) fail("Morse of at most 800 characters (each is up to 6 symbols)");
        for (; p < end; p++) {
            int c = lower(*p), i = alpha(c) ? c - 'a' : digit(c) ? 26 + c - '0' : -1;
            if (space(c)) {
                if (!first) say(" /");
                continue;
            }
            if (i < 0) continue;
            if (!first) put(' ');
            say(MORSE[i]);
            first = 0;
        }
        return;
    }
    case 6:
        for (const u8 *q = p; q < end; q++) {
            if (q > p) put(' ');
            for (int b = 7; b >= 0; b--) put('0' + (*q >> b & 1));
        }
        return;
    }
}

void decode(void) {
    str t = trim(need("text"));
    u8 *index = 0;
    int shift, e = encoding(need("encoding"), &shift);
    const char *p = t.p, *end = t.e;
    switch (e) {
    case 0: {
        u32 v = 0;
        int bits = 0;
        for (; p < end; p++) {
            if (*p == '=' || space(*p)) continue;
            int c = *p, x = (unsigned)(c - 'A') < 26 ? c - 'A' : (unsigned)(c - 'a') < 26 ? c - 'a' + 26 : digit(c) ? c - '0' + 52
                    : c == '+' || c == '-' ? 62 : c == '/' || c == '_' ? 63 : -1;
            if (x < 0) fail("not base64");
            v = v << 6 | (u32)x, bits += 6;
            if (bits >= 8) put((v >> (bits - 8)) & 255), bits -= 8;
        }
        return;
    }
    case 1: {
        int hi = -1;
        for (; p < end; p++) {
            if (space(*p) || *p == ':') continue;
            int d = digit(*p) ? *p - '0' : (unsigned)((*p | 32) - 'a') < 6 ? (*p | 32) - 'a' + 10 : -1;
            if (d < 0) fail("not hex");
            if (hi < 0) hi = d;
            else put(hi << 4 | d), hi = -1;
        }
        if (hi >= 0) fail("an odd number of hex digits");
        return;
    }
    case 2:
        for (; p < end; p++) {
            if (*p == '%' && p + 2 < end && hexd((u8)p[1]) && hexd((u8)p[2])) {
                int a = digit(p[1]) ? p[1] - '0' : (p[1] | 32) - 'a' + 10, b = digit(p[2]) ? p[2] - '0' : (p[2] | 32) - 'a' + 10;
                put(a << 4 | b), p += 2;
            } else
                put(*p == '+' ? ' ' : (u8)*p);
        }
        return;
    case 3: case 4: caesar(t, (26 - shift) % 26); return;
    case 5:
        while (p < end) {
            while (p < end && *p == ' ') p++;
            if (p < end && *p == '/') {
                put(' ');
                p++;
                continue;
            }
            const char *s = p;
            u32 code = 1; /* (the symbols as bits after a leading 1: a letter's number, looked up at once) */
            while (p < end && (*p == '.' || *p == '-')) code = code << 1 | (*p == '-'), p++;
            if (s == p) {
                if (p < end) fail("morse: dots, dashes, spaces between letters and / between words");
                break;
            }
            int found = -1;
            if (p - s <= 6) {
                if (!index) {
                    index = alloc(128);
                    memset(index, 0xff, 128);
                    for (int i = 0; i < 36; i++) {
                        u32 c = 1;
                        for (const char *m = MORSE[i]; *m; m++) c = c << 1 | (*m == '-');
                        index[c] = (u8)i;
                    }
                }
                found = index[code] == 0xff ? -1 : index[code];
            }
            if (found < 0) fail("a morse letter it does not know");
            put(found < 26 ? 'a' + found : '0' + found - 26);
        }
        return;
    case 6: {
        int v = 0, bits = 0;
        for (; p < end; p++) {
            if (space(*p)) continue;
            if (*p != '0' && *p != '1') fail("binary: 0s and 1s, 8 a character");
            v = v << 1 | (*p - '0');
            if (++bits == 8) put(v), v = bits = 0;
        }
        if (bits) fail("binary: a multiple of 8 bits");
        return;
    }
    }
}

void edit_distance(void) {
    str a = need("first"), b = need("second");
    /* characters as code points */
    u32 *x = alloc(sizeof(u32) * 81), *y = alloc(sizeof(u32) * 81);
    int n = 0, m = 0, k;
    for (const char *p = a.p; p < a.e; p += k) {
        if (n == 80) fail("texts of up to 80 characters");
        x[n++] = code_point(p, a.e, &k);
    }
    for (const char *p = b.p; p < b.e; p += k) {
        if (m == 80) fail("texts of up to 80 characters");
        y[m++] = code_point(p, b.e, &k);
    }
    u8 *row = alloc(m + 1);
    for (int j = 0; j <= m; j++) row[j] = (u8)j;
    for (int i = 1; i <= n; i++) {
        int diag = row[0];
        row[0] = (u8)i;
        for (int j = 1; j <= m; j++) {
            int up = row[j], best = diag + (x[i - 1] != y[j - 1]);
            if (up + 1 < best) best = up + 1;
            if (row[j - 1] + 1 < best) best = row[j - 1] + 1;
            diag = up;
            row[j] = (u8)best;
        }
    }
    say_int(row[m]);
    say(row[m] == 1 ? " edit (insertions, deletions or replacements of a character)" : " edits (insertions, deletions or replacements of a character)");
}
