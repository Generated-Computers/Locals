/* run_python: a small Python the computer runs, for the computations a model would otherwise give a code
   interpreter. Integers (to 2^60), True / False / None, strings, lists, tuples (lists written with ( )), dicts and
   ranges; assignments (a, b = b, a + b too), += and the others; if / elif / else, while, for, break, continue, pass,
   def and return (recursion too), lambda, global, del; + - * / // % ** & | ^ << >>, comparisons (chained), and / or /
   not, in, is, x if c else y, [x for x in ... if ...] (in sum(...), any(...) too) and {k: v for ...}, f-strings
   ({x:,} {x:>5} {x:08b}), slices; print len range abs min max sum sorted reversed list tuple dict str int bool
   enumerate zip any all divmod pow chr ord bin hex oct, map and filter of a function; strings', lists' and dicts'
   common methods (as values too: key=d.get); math.gcd lcm isqrt factorial comb perm prod. No floats (/ gives an
   integer where it divides exactly, else an error), no try, class or set: each is an error that says so.

   The program is read into a tree of nodes, which is walked. A value is 8 bytes: an integer (to 2^62) shifted left
   1 with its lowest bit set, an object's address (8-byte aligned), or None / False / True (2, 4, 6). A run has a budget of steps (a node walked is one) within
   the computer's run limit. What it prints is the response. */
#include "tools.h"
#ifdef HOST
#include <stdio.h>
#include <stdlib.h>
#endif

typedef i64 val;
#define IS_INT(v) ((v) & 1)
#define INT(v) ((v) >> 1)
#define MKINT(i) ((val)(i) * 2 + 1)
#define NONE_V ((val)2)
#define FALSE_V ((val)4)
#define TRUE_V ((val)6)
#define IS_BOOL(v) ((v) == FALSE_V || (v) == TRUE_V)
#define IS_OBJ(v) (!((v) & 1) && (v) > 6)
#define OBJ(v) ((head *)(size_t)(v))
#define MKOBJ(o) ((val)(size_t)(o))
#define MAXI (((i64)1 << 62) - 1)

enum { O_STR = 1, O_LIST, O_DICT, O_FUNC, O_RANGE, O_BUILTIN };

typedef struct {
    u8 type, tuple;
    u16 x;
    i32 len;
} head;
typedef struct {
    head h;
    char s[];
} strobj;
typedef struct {
    head h;
    i32 cap, pad;
    val *items;
} listobj;
typedef struct {
    head h;
    i32 cap, pad;
    val *k, *v;
} dictobj;
typedef struct {
    head h;
    i64 start, stop, step;
} rangeobj;
typedef struct {
    head h;
    i32 node, pad; /* a def's or lambda's node; a builtin's name id */
    val self;      /* a method's object (NONE_V: a function) */
} fnobj;

static int line;

static __attribute__((noreturn)) void err3(const char *kind, const char *msg, str what, const char *after) {
    capture_at = 0; /* (an error while str() writes into its buffer: to the response) */
    say("error (line ");
    say_int(line);
    say(": ");
    say(kind);
    say(": ");
    say(msg);
    if (have(what)) say_str(what);
    say(after);
    put(')');
    done();
}

static __attribute__((noreturn)) void err(const char *kind, const char *msg) { err3(kind, msg, (str){0, 0}, ""); }

/* the lowest the stack may go (heap_reserve's floor, and room for an error's report below it): the parser, the walk
   and the printing go deeper in a nested program (a few hundred bytes a level), and stop there with an error */
static char *stack_floor, *parse_floor; /* (the same on the computer; on the host, each as deep as the computer's) */
static __attribute__((noinline, noreturn)) void deep(void) {
    err("RecursionError", "nested too deeply for the computer's memory");
}
#define DEEP() (stack_now() < stack_floor ? deep() : (void)0)
#define PARSE_DEEP() (stack_now() < parse_floor ? deep() : (void)0)

/* ---- names, interned; the builtins', methods' and keywords' first (their ids their places here) */
static const char *const FIXED[] = {
    "print", "len", "range", "abs", "min", "max", "sum", "sorted", "reversed", "list", "tuple", "dict", "str", "int",
    "bool", "enumerate", "zip", "any", "all", "divmod", "pow", "chr", "ord", "bin", "hex", "oct", "map", "filter",
    "gcd", "lcm", "isqrt", "factorial", "comb", "perm", "prod", "math",
    "append", "pop", "insert", "extend", "sort", "reverse", "index", "count", "remove", "upper", "lower", "strip",
    "lstrip", "rstrip", "split", "join", "replace", "find", "startswith", "endswith", "isdigit", "isalpha",
    "isalnum", "isspace", "isupper", "islower", "title", "capitalize", "zfill", "get", "keys", "values", "items",
    "setdefault", "copy", "rjust", "ljust", "center", "update", "clear", "splitlines", "rfind",
    "sep", "end", "key", "start", 0};
enum {
    B_PRINT, B_LEN, B_RANGE, B_ABS, B_MIN, B_MAX, B_SUM, B_SORTED, B_REVERSED, B_LIST, B_TUPLE, B_DICT, B_STR, B_INT,
    B_BOOL, B_ENUMERATE, B_ZIP, B_ANY, B_ALL, B_DIVMOD, B_POW, B_CHR, B_ORD, B_BIN, B_HEX, B_OCT, B_MAP, B_FILTER,
    B_GCD, B_LCM, B_ISQRT, B_FACTORIAL, B_COMB, B_PERM, B_PROD, B_MATH,
    M_APPEND, M_POP, M_INSERT, M_EXTEND, M_SORT, M_REVERSE, M_INDEX, M_COUNT, M_REMOVE, M_UPPER, M_LOWER, M_STRIP,
    M_LSTRIP, M_RSTRIP, M_SPLIT, M_JOIN, M_REPLACE, M_FIND, M_STARTSWITH, M_ENDSWITH, M_ISDIGIT, M_ISALPHA,
    M_ISALNUM, M_ISSPACE, M_ISUPPER, M_ISLOWER, M_TITLE, M_CAPITALIZE, M_ZFILL, M_GET, M_KEYS, M_VALUES, M_ITEMS,
    M_SETDEFAULT, M_COPY, M_RJUST, M_LJUST, M_CENTER, M_UPDATE, M_CLEAR, M_SPLITLINES, M_RFIND,
    K_SEP, K_END, K_KEY, K_START, N_FIXED
};
#define B_LAST B_MATH /* (the builtins: ids up to here) */
#define K_REVERSE M_REVERSE

#define NAMES 160
static str *names;
static int n_names;

static i16 *slots; /* (names by hash: 256 slots, open addressing, the name's id + 1) */

static int same_name(const char *z, const char *p, size_t n) { /* the NUL-terminated z is [p, p + n) */
    size_t i = 0;
    while (i < n && z[i] == p[i]) i++;
    return i == n && z[n] == 0;
}

/* first-letter chains of FIXED, KEYWORDS and OP_TEXT (made once in a computer: its RAM keeps them): first[c] the first
   entry that starts with c, next[i] the next after entry i (each + 1; 0: none). A new name's, a keyword's and an
   operator's search was through the whole list: half a parse's time */
static u8 fx_first[128], fx_next[sizeof FIXED / sizeof *FIXED], kw_first[128], kw_next[40], op_first[128], op_next[64];
static void chain(const char *const *list, u8 *first, u8 *next) {
    int n = 0;
    while (list[n]) n++;
    for (int i = n - 1; i >= 0; i--) next[i] = first[(u8)list[i][0]], first[(u8)list[i][0]] = (u8)(i + 1);
}

static int intern(str s) { /* (a builtin's name gets its fixed id the first time the program writes it) */
    u32 h = 2166136261u;
    size_t n = s.e - s.p;
    for (const char *p = s.p; p < s.e; p++) h = (h ^ (u8)*p) * 16777619u;
    for (h &= 255;; h = (h + 1) & 255) {
        int i = slots[h] - 1;
        if (i < 0) break;
        if ((size_t)(names[i].e - names[i].p) == n && !memcmp(names[i].p, s.p, n)) return i;
    }
    int id = -1;
    for (int i = (u8)*s.p < 128 ? fx_first[(u8)*s.p] : 0; i && id < 0; i = fx_next[i - 1])
        if (same_name(FIXED[i - 1], s.p, n)) id = i - 1;
    if (id < 0) {
        if (n_names == NAMES) err("SyntaxError", "more than 160 names");
        id = n_names++;
    }
    names[id] = s;
    slots[h] = (i16)(id + 1);
    return id;
}

/* ---- tokens */
enum { T_NAME, T_NUM, T_STR, T_FSTR, T_OP, T_NL, T_INDENT, T_DEDENT, T_END };
/* (an operator's code: its place in OP_TEXT + 1; a keyword's: its place in KEYWORDS + 1) */
static const char *const OP_TEXT[] = {"**=", "//=", ">>=", "<<=", "**", "//", "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "->", "<<", ">>", "+", "-", "*", "/", "%", "=", "<", ">", "!", "&", "|", "^", "~", "(", ")", "[", "]", "{", "}", ",", ":", ".", ";", "@", 0};
enum {
    P_NONE, P_STAR_STAR_EQ, P_SLASH_SLASH_EQ, P_GT_GT_EQ, P_LT_LT_EQ, P_STAR_STAR, P_SLASH_SLASH, P_EQ_EQ, P_BANG_EQ, P_LT_EQ, P_GT_EQ, P_PLUS_EQ, P_MINUS_EQ, P_STAR_EQ, P_SLASH_EQ, P_PCT_EQ, P_AMP_EQ, P_BAR_EQ, P_CARET_EQ, P_MINUS_GT, P_LT_LT, P_GT_GT, P_PLUS, P_MINUS, P_STAR, P_SLASH, P_PCT, P_EQ, P_LT, P_GT, P_BANG, P_AMP, P_BAR, P_CARET, P_TILDE, P_LPAR, P_RPAR, P_LSQ, P_RSQ, P_LBR, P_RBR, P_COMMA, P_COLON, P_DOT, P_SEMI, P_AT
};
static const char *const KEYWORDS[] = {"if", "else", "elif", "while", "for", "in", "def", "return", "and", "or", "not", "is", "None", "True", "False", "break", "continue", "pass", "lambda", "global", "del", "import", "from", "as", "try", "except", "finally", "class", "with", "yield", "raise", "assert", "nonlocal", "async", "await", 0};
enum {
    KW_0, KW_IF, KW_ELSE, KW_ELIF, KW_WHILE, KW_FOR, KW_IN, KW_DEF, KW_RETURN, KW_AND, KW_OR, KW_NOT, KW_IS, KW_NONE, KW_TRUE, KW_FALSE, KW_BREAK, KW_CONTINUE, KW_PASS, KW_LAMBDA, KW_GLOBAL, KW_DEL, KW_IMPORT, KW_FROM, KW_AS, KW_TRY, KW_EXCEPT, KW_FINALLY, KW_CLASS, KW_WITH, KW_YIELD, KW_RAISE, KW_ASSERT, KW_NONLOCAL, KW_ASYNC, KW_AWAIT
};
typedef struct {
    u8 type, code; /* an operator's or keyword's code; a string's: 1 if raw (r"...") */
    u16 line;
    u16 at, n; /* in the source: where, and how long */
} token;

static const char *src, *src_end;
static token *toks;
static int n_toks, max_toks, ti;

static int word_char(int c) { return (unsigned)((c | 32) - 'a') < 26 || digit(c) || c == '_' || c >= 0x80; }

static void add_tok(int type, const char *at, int n, int code, int ln) {
    if (n_toks == max_toks) line = ln, err("SyntaxError", "a program too long for the computer");
    toks[n_toks++] = (token){(u8)type, (u8)code, (u16)ln, (u16)(at - src), (u16)n};
}

/* tokens of [p, e) from line ln: a program's (lines: indentation, newlines), or an f-string's part's */
static void lex(const char *p, const char *e, int ln, int lines) {
    int depth = 0, ind[32] = {0}, n_ind = 1, start = lines;
    while (p < e) {
        if (start) {
            int col = 0;
            const char *q = p;
            while (q < e && (*q == ' ' || *q == '\t')) col += *q == '\t' ? 8 - col % 8 : 1, q++;
            if (q >= e || *q == '\n' || *q == '#' || *q == '\r') {
                while (q < e && *q != '\n') q++;
                if (q < e) q++, ln++;
                p = q;
                continue;
            }
            if (col > ind[n_ind - 1]) {
                if (n_ind == 32) line = ln, err("SyntaxError", "nested too deep");
                ind[n_ind++] = col;
                add_tok(T_INDENT, q, 0, 0, ln);
            }
            while (col < ind[n_ind - 1]) {
                n_ind--;
                add_tok(T_DEDENT, q, 0, 0, ln);
                if (col > ind[n_ind - 1]) line = ln, err("IndentationError", "an indentation that matches no outer one");
            }
            p = q;
            start = 0;
        }
        char c = *p;
        if (c == '\n') {
            if (!depth && lines) add_tok(T_NL, p, 0, 0, ln), start = 1;
            ln++, p++;
        } else if (c == ' ' || c == '\t' || c == '\r') {
            p++;
        } else if (c == '#') {
            while (p < e && *p != '\n') p++;
        } else if (c == '\\' && p + 1 < e && p[1] == '\n') {
            p += 2, ln++;
        } else if (digit(c)) {
            const char *s = p;
            while (p < e && word_char((u8)*p)) p++;
            if (p < e && *p == '.' && !(p + 1 < e && p[1] == '.')) {
                line = ln;
                if (p - s == 1 && *s == '0' && p + 1 < e && p[1] == '5')
                    err("ValueError", "no floats on the computer: for a square root use math.isqrt(n) (whole numbers)");
                err("ValueError", "no floats on the computer (whole numbers only: // divides; calc does decimals)");
            }
            add_tok(T_NUM, s, (int)(p - s), 0, ln);
        } else if ((c == '"' || c == '\'') ||
                   ((c == 'f' || c == 'F' || c == 'r' || c == 'R') && p + 1 < e && (p[1] == '"' || p[1] == '\''))) {
            int f = c == 'f' || c == 'F', raw = c == 'r' || c == 'R';
            if (c != '"' && c != '\'') p++;
            char q = *p;
            int triple = p + 2 < e && p[1] == q && p[2] == q;
            const char *s = p += triple ? 3 : 1;
            for (;;) {
                if (p >= e) line = ln, err("SyntaxError", "a string without its closing quote");
                if (*p == '\\') {
                    if (p + 1 < e && p[1] == '\n') ln++;
                    p += 2;
                    continue;
                }
                if (*p == '\n') {
                    if (!triple) line = ln, err("SyntaxError", "a string without its closing quote");
                    ln++;
                }
                if (*p == q && (!triple || (p + 2 < e && p[1] == q && p[2] == q))) break;
                p++;
            }
            add_tok(f ? T_FSTR : T_STR, s, (int)(p - s), raw, ln);
            p += triple ? 3 : 1;
        } else if (word_char((u8)c)) {
            const char *s = p;
            while (p < e && word_char((u8)*p)) p++;
            int code = KW_0, n = (int)(p - s);
            if (n >= 2 && n <= 8 && (u8)*s < 128)
                for (int i = kw_first[(u8)*s]; i && !code; i = kw_next[i - 1])
                    if (same_name(KEYWORDS[i - 1], s, n)) code = i;
            add_tok(T_NAME, s, n, code, ln);
        } else {
            int code = 0, n = 0;
            for (int i = (u8)c < 128 ? op_first[(u8)c] : 0; i && !code; i = op_next[i - 1]) { /* (the longest
                                                                                       first: OP_TEXT has them so) */
                const char *o = OP_TEXT[i - 1];
                int k = o[1] ? o[2] ? 3 : 2 : 1;
                if (e - p >= k && (k < 2 || p[1] == o[1]) && (k < 3 || p[2] == o[2])) code = i, n = k;
            }
            if (!code) line = ln, err("SyntaxError", "a character Python does not have there");
            if (n == 1 && (c == '(' || c == '[' || c == '{')) depth++;
            if (n == 1 && (c == ')' || c == ']' || c == '}') && depth) depth--;
            add_tok(T_OP, p, n, code, ln);
            p += n;
        }
    }
    if (lines) {
        if (!start) add_tok(T_NL, p, 0, 0, ln);
        while (n_ind > 1) n_ind--, add_tok(T_DEDENT, p, 0, 0, ln);
    }
    add_tok(T_END, p, 0, 0, ln);
}

/* ---- the tree */
enum {
    N_INT, N_STR, N_NAME, N_CONST, N_LIST, N_TUPLE, N_DICT, N_COMP, N_DICTCOMP, N_FOR_CLAUSE, N_BIN, N_UNARY, N_NOT,
    N_AND, N_OR, N_CMP, N_IF_EXP, N_CALL, N_KW, N_ATTR, N_INDEX, N_SLICE, N_LAMBDA, N_FSTR, N_FPART,
    S_EXPR, S_ASSIGN, S_AUG, S_IF, S_WHILE, S_FOR, S_DEF, S_RETURN, S_BREAK, S_CONTINUE, S_PASS, S_GLOBAL, S_DEL,
    S_BLOCK
};

typedef struct {
    u8 kind, op;
    u16 line;
    i16 a, b, c, next; /* children (node indices, -1: none), the next in a list */
} node;

static node *nodes;
static int n_nodes, max_nodes;
static val *consts; /* N_INT, N_STR: consts[a] */
static int n_consts, max_consts;

static int mk(int kind, int op, int a, int b, int c) {
    if (n_nodes == max_nodes) err("SyntaxError", "a program too long for the computer");
    nodes[n_nodes] = (node){(u8)kind, (u8)op, (u16)line, (i16)a, (i16)b, (i16)c, -1};
    return n_nodes++;
}

static int add_const(val v) {
    if (n_consts == max_consts) err("SyntaxError", "a program with too many constants for the computer");
    consts[n_consts] = v;
    return n_consts++;
}

/* the run's budget of steps (a node walked, a call three, a dict's 8 keys looked at one, a sort's comparison one) */
static int steps;
#define STEPS 1250 /* (what a run that uses them all costs at most: about 176,000 instructions, 215,000 trace
                       tokens, measured on the reference machine) */
static __attribute__((noinline, noreturn)) void timeout(void);
#define step() (++steps > STEPS ? timeout() : (void)0)
/* work besides the walk's (a string scanned or copied, a list's items looked at, a builtin's call): `units` of
   about 6 instructions, 12 of them a step (the rest kept for the next), so that the 1,250 steps bound what a run
   does, however it does it: within about 176,000 of the computer's instructions (at 16 a step, a loop of joins
   went to 180,000, and 234,000 trace tokens) */
static u32 work;
static __attribute__((noinline)) void charge_flush(void) {
    steps += (int)(work / 12), work %= 12;
    if (steps > STEPS) timeout();
}
#define charge(units) (work += (u32)(units), work >= 12 ? charge_flush() : (void)0)

/* ---- a loop's rounds give their memory back: what a round of a for or while loop, or of a comprehension of one for,
   made (the heap from its mark on) is free again at the round's end, unless something made in it went into what is
   older than the round (a variable, or a list or dict made before it: ``kept_in``, at each store). Without it a loop
   of str(x) or f-strings ran out of the computer's 15 KB in a few hundred rounds. */
#define ROUNDS 8
static char *round_at[ROUNDS];
static int round_frame[ROUNDS], rounds;
static u8 round_kept[ROUNDS];

static __attribute__((noinline)) void kept_in(const void *holder, const void *p) { /* p went into holder */
    const char *q = p, *top = heap_mark();
    for (int i = 0; i < rounds; i++)
        if (q >= round_at[i] && q < top && (const char *)holder < round_at[i]) round_kept[i] = 1;
}
/* (at once where it cannot matter: a holder made in the newest round, or what is older than the oldest) */
#define KEEPS(holder, p) ((const char *)(holder) < round_at[rounds - 1] && (const char *)(p) >= round_at[0])
#define KEEP(holder, v) (rounds && IS_OBJ(v) && KEEPS(holder, OBJ(v)) ? kept_in((holder), OBJ(v)) : (void)0)
#define KEEP_PTR(holder, p) (rounds && KEEPS(holder, p) ? kept_in((holder), (p)) : (void)0)

/* ---- objects */
static strobj *new_str(const char *s, size_t n) {
    if (n > 60000) err("MemoryError", "a string too long for the computer");
    charge((i64)n >> 2);
    strobj *o = alloc(sizeof(strobj) + n);
    o->h = (head){O_STR, 0, 0, (i32)n};
    if (s) memcpy(o->s, s, n);
    return o;
}

static val str_v(const char *s, size_t n) { return MKOBJ(new_str(s, n)); }
static val zstr(const char *z) { return str_v(z, strlen(z)); }

static listobj *new_list(int n) {
    listobj *o = alloc(sizeof(listobj));
    o->h = (head){O_LIST, 0, 0, n};
    o->cap = n < 4 ? 4 : n;
    o->items = alloc(sizeof(val) * o->cap);
    return o;
}

static void list_push(listobj *l, val v) {
    if (l->h.len == l->cap) {
        val *n = alloc(sizeof(val) * l->cap * 2);
        memcpy(n, l->items, sizeof(val) * l->h.len);
        l->items = n;
        l->cap *= 2;
        KEEP_PTR(l, n);
    }
    l->items[l->h.len++] = v;
    KEEP(l, v);
}

static dictobj *new_dict(void) {
    dictobj *o = alloc(sizeof(dictobj));
    o->h = (head){O_DICT, 0, 0, 0};
    o->cap = 4;
    o->k = alloc(sizeof(val) * 4);
    o->v = alloc(sizeof(val) * 4);
    return o;
}

static int type_of(val v) {
    if (IS_INT(v)) return 0;
    if (!IS_OBJ(v)) return -1;
    return OBJ(v)->type;
}

static int truthy(val v);
static int equal(val a, val b);

static int dict_find(dictobj *d, val k) {
    int n = d->h.len;
    if (IS_INT(k)) { /* (an integer key: compared as one, at once) */
        for (int i = 0; i < n; i++) {
            if ((i & 7) == 7) charge(24);
            if (d->k[i] == k) return i;
        }
        return -1;
    }
    if (IS_OBJ(k) && OBJ(k)->type == O_STR) { /* (a string key: the strings of its length compared) */
        strobj *x = (strobj *)OBJ(k);
        for (int i = 0; i < n; i++) {
            charge(8);
            val y = d->k[i];
            if (IS_OBJ(y) && OBJ(y)->type == O_STR && OBJ(y)->len == x->h.len) {
                charge(x->h.len / 4);
                if (!memcmp(((strobj *)OBJ(y))->s, x->s, x->h.len)) return i;
            }
        }
        return -1;
    }
    for (int i = 0; i < n; i++) {
        charge(6);
        if (equal(d->k[i], k)) return i;
    }
    return -1;
}

static void dict_set(dictobj *d, val k, val v) {
    int i = dict_find(d, k);
    KEEP(d, v);
    if (i >= 0) {
        d->v[i] = v;
        return;
    }
    if (d->h.len == d->cap) {
        val *nk = alloc(sizeof(val) * d->cap * 2), *nv = alloc(sizeof(val) * d->cap * 2);
        memcpy(nk, d->k, sizeof(val) * d->h.len);
        memcpy(nv, d->v, sizeof(val) * d->h.len);
        d->k = nk, d->v = nv, d->cap *= 2;
        KEEP_PTR(d, nk), KEEP_PTR(d, nv);
    }
    KEEP(d, k);
    d->k[d->h.len] = k;
    d->v[d->h.len++] = v;
}

static val mkint(i64 i) {
    if (i > MAXI || i < -MAXI) err("OverflowError", "an integer beyond 2^62 (the computer's; calc does bigger)");
    return MKINT(i);
}

static i64 as_int(val v, const char *what) {
    if (IS_INT(v)) return INT(v);
    if (IS_BOOL(v)) return v == TRUE_V;
    err3("TypeError", what, (str){0, 0}, " must be an integer");
}

static const char *type_name(val v) {
    static const char *const T[] = {"int", "str", "list", "dict", "function", "range", "function"};
    if (v == NONE_V) return "NoneType";
    if (IS_BOOL(v)) return "bool";
    int t = type_of(v);
    if (t == O_LIST && OBJ(v)->tuple) return "tuple";
    return T[t];
}

/* ---- printing */
static void say_val(val v, int repr);

static void say_seq(val *items, int n, int tuple) {
    charge(20 * (i64)n);
    put(tuple ? '(' : '[');
    for (int i = 0; i < n; i++) {
        if (i) say(", ");
        say_val(items[i], 1);
    }
    if (tuple && n == 1) put(',');
    put(tuple ? ')' : ']');
}

static void say_val(val v, int repr) {
    DEEP();
    if (IS_INT(v)) return say_int(INT(v));
    if (v == NONE_V) return say("None");
    if (v == TRUE_V) return say("True");
    if (v == FALSE_V) return say("False");
    head *h = OBJ(v);
    if (h->type == O_STR) {
        strobj *s = (strobj *)h;
        if (!repr) return say_str((str){s->s, s->s + s->h.len});
        int dq = 0, sq = 0;
        for (int i = 0; i < s->h.len; i++) sq |= s->s[i] == '\'', dq |= s->s[i] == '"';
        char q = sq && !dq ? '"' : '\'';
        put(q);
        for (int i = 0; i < s->h.len; i++) {
            char c = s->s[i];
            if (c == '\\' || c == q) put('\\'), put(c);
            else if (c == '\n') say("\\n");
            else if (c == '\t') say("\\t");
            else put((u8)c);
        }
        put(q);
    } else if (h->type == O_LIST) {
        listobj *l = (listobj *)h;
        say_seq(l->items, l->h.len, l->h.tuple);
    } else if (h->type == O_DICT) {
        dictobj *d = (dictobj *)h;
        put('{');
        for (int i = 0; i < d->h.len; i++) {
            if (i) say(", ");
            say_val(d->k[i], 1);
            say(": ");
            say_val(d->v[i], 1);
        }
        put('}');
    } else if (h->type == O_RANGE) {
        rangeobj *r = (rangeobj *)h;
        say("range(");
        say_int(r->start);
        say(", ");
        say_int(r->stop);
        if (r->step != 1) say(", "), say_int(r->step);
        put(')');
    } else
        say("<function>");
}

/* ---- the parser */
static token *tk(void) { return &toks[ti]; }

static str tok_text(const token *t) { return (str){src + t->at, src + t->at + (t->type == T_OP || t->type == T_NAME || t->type == T_NUM ? t->n : 0)}; }

#define is_op(c) (toks[ti].type == T_OP && toks[ti].code == (c))
#define is_kw(c) (toks[ti].type == T_NAME && toks[ti].code == (c))

static void expect(int c) {
    line = tk()->line;
    if (!is_op(c)) err3("SyntaxError", "expected ", (str){OP_TEXT[c - 1], OP_TEXT[c - 1] + strlen(OP_TEXT[c - 1])}, "");
    ti++;
}

static int accept_op(int c) {
    if (!is_op(c)) return 0;
    ti++;
    return 1;
}

static int accept_kw(int c) {
    if (!is_kw(c)) return 0;
    ti++;
    return 1;
}

static int keyword(const token *t) { return t->code != KW_0; }

static int expr(void);
static int expr_list(int *tuple);
static int block(void);
static int no_in; /* (a for's target being read: its "in" is the for's) */

static int string_const(const token *t) { /* a string literal's value (escapes read) */
    const char *p = src + t->at, *e = p + t->n;
    strobj *s = new_str(0, t->n);
    int n = 0;
    while (p < e) {
        char c = *p++;
        if (c == '\\' && !t->code && p < e) {
            c = *p++;
            if (c == 'n') c = '\n';
            else if (c == 't') c = '\t';
            else if (c == 'r') c = '\r';
            else if (c == '0') c = 0;
            else if (c == '\n') continue;
            else if (c != '\\' && c != '\'' && c != '"') s->s[n++] = '\\';
        }
        s->s[n++] = c;
    }
    s->h.len = n;
    return add_const(MKOBJ(s));
}

static int fstring(const token *t) { /* parts: text (N_STR) and {expr[!r][:spec]} (N_FPART: a expr, b spec const) */
    const char *p = src + t->at, *e = p + t->n;
    int first = -1, last = -1, saved = ti;
    while (p < e) {
        int part;
        if (*p == '{' && p + 1 < e && p[1] != '{') {
            const char *s = ++p, *q = s;
            int depth = 0;
            while (q < e && !(depth == 0 && (*q == '}' || *q == ':' || (*q == '!' && !(q + 1 < e && q[1] == '='))))) {
                if (*q == '(' || *q == '[' || *q == '{') depth++;
                if (*q == ')' || *q == ']' || *q == '}') depth--;
                if (*q == '\'' || *q == '"') { /* (a string inside) */
                    char quote = *q++;
                    while (q < e && *q != quote) q++;
                }
                q++;
            }
            int start = n_toks;
            lex(s, q, t->line, 0);
            int at = ti;
            ti = start;
            int x = expr();
            if (tk()->type != T_END) line = t->line, err("SyntaxError", "an f-string's {} that is not one expression");
            ti = at;
            int spec = -1, conv = 0;
            if (q < e && *q == '!') conv = q[1], q += 2;
            if (q < e && *q == ':') {
                const char *a = ++q;
                while (q < e && *q != '}') q++;
                spec = add_const(str_v(a, q - a));
            }
            if (q >= e) line = t->line, err("SyntaxError", "an f-string's { without its }");
            p = q + 1;
            part = mk(N_FPART, conv, x, spec, -1);
        } else {
            const char *s = p;
            strobj *o = new_str(0, e - p);
            int n = 0;
            while (p < e && !(*p == '{' && !(p + 1 < e && p[1] == '{'))) {
                if ((*p == '{' || *p == '}') && p + 1 < e && p[1] == *p) p++;
                else if (*p == '\\' && p + 1 < e) {
                    p++;
                    o->s[n++] = *p == 'n' ? '\n' : *p == 't' ? '\t' : *p;
                    p++;
                    continue;
                }
                o->s[n++] = *p++;
            }
            (void)s;
            o->h.len = n;
            part = mk(N_STR, 0, add_const(MKOBJ(o)), -1, -1);
        }
        if (last < 0) first = part;
        else nodes[last].next = (i16)part;
        last = part;
    }
    ti = saved;
    return mk(N_FSTR, 0, first, -1, -1);
}

static int comp_clauses(void) { /* for TARGET in ITER [if COND]... (after the element) */
    int first = -1, last = -1;
    while (is_kw(KW_FOR)) {
        ti++;
        int tuple;
        no_in++;
        int target = expr_list(&tuple);
        no_in--;
        if (!accept_kw(KW_IN)) line = tk()->line, err("SyntaxError", "a for without its in");
        int it = mk(N_FOR_CLAUSE, 0, target, -1, -1);
        nodes[it].b = (i16)expr();
        int conds = -1, cl = -1;
        while (accept_kw(KW_IF)) {
            int x = expr();
            if (cl < 0) conds = x;
            else nodes[cl].next = (i16)x;
            cl = x;
        }
        nodes[it].c = (i16)conds;
        if (last < 0) first = it;
        else nodes[last].next = (i16)it;
        last = it;
    }
    return first;
}

static int atom(void) {
    PARSE_DEEP();
    token *t = tk();
    line = t->line;
    if (t->type == T_NUM) {
        ti++;
        const char *p = src + t->at, *e = p + t->n;
        int base = 10;
        if (e - p > 2 && p[0] == '0' && (p[1] | 32) == 'x') base = 16, p += 2;
        else if (e - p > 2 && p[0] == '0' && (p[1] | 32) == 'b') base = 2, p += 2;
        else if (e - p > 2 && p[0] == '0' && (p[1] | 32) == 'o') base = 8, p += 2;
        i64 v = 0, lim = MAXI / base; /* (as int() reads them: no 64-bit division a digit) */
        int last = (int)(MAXI % base);
        for (; p < e; p++) {
            if (*p == '_') continue;
            int d = digit(*p) ? *p - '0' : (unsigned)((*p | 32) - 'a') < 6 ? (*p | 32) - 'a' + 10 : 99;
            if (d >= base) err("SyntaxError", "a number Python does not read (no floats, e or j)");
            if (v > lim || (v == lim && d > last)) err("OverflowError", "an integer beyond 2^62 (the computer's; calc does bigger)");
            v = v * base + d;
        }
        return mk(N_INT, 0, add_const(MKINT(v)), -1, -1);
    }
    if (t->type == T_STR || t->type == T_FSTR) { /* (adjacent ones joined) */
        int first = -1, last = -1, n = 0;
        while (tk()->type == T_STR || tk()->type == T_FSTR) {
            token *s = tk();
            ti++;
            int x = s->type == T_STR ? mk(N_STR, 0, string_const(s), -1, -1) : fstring(s);
            if (last < 0) first = x;
            else nodes[last].next = (i16)x;
            last = x, n++;
        }
        return n == 1 && nodes[first].kind == N_STR ? first : mk(N_FSTR, 1, first, -1, -1);
    }
    if (t->type == T_NAME) {
        if (is_kw(KW_NONE) || is_kw(KW_TRUE) || is_kw(KW_FALSE)) {
            int v = is_kw(KW_NONE) ? 0 : is_kw(KW_TRUE) ? 2 : 1;
            ti++;
            return mk(N_CONST, v, -1, -1, -1);
        }
        if (is_kw(KW_LAMBDA)) {
            ti++;
            int params = -1, last = -1;
            while (!is_op(P_COLON)) {
                if (tk()->type != T_NAME) err("SyntaxError", "a lambda's parameters are names");
                int p = mk(N_NAME, 0, intern(tok_text(tk())), -1, -1);
                ti++;
                if (last < 0) params = p;
                else nodes[last].next = (i16)p;
                last = p;
                if (!accept_op(P_COMMA)) break;
            }
            expect(P_COLON);
            return mk(N_LAMBDA, 0, params, expr(), -1);
        }
        if (keyword(t)) err3("SyntaxError", "a keyword where a value should be: ", tok_text(t), "");
        ti++;
        return mk(N_NAME, 0, intern(tok_text(t)), -1, -1);
    }
    if (accept_op(P_LPAR)) {
        if (accept_op(P_RPAR)) return mk(N_TUPLE, 0, -1, -1, -1);
        int x = expr();
        if (is_kw(KW_FOR)) {
            int c = comp_clauses();
            expect(P_RPAR);
            return mk(N_COMP, 0, x, c, -1);
        }
        if (!is_op(P_COMMA)) {
            expect(P_RPAR);
            return x;
        }
        int last = x;
        while (accept_op(P_COMMA) && !is_op(P_RPAR)) {
            int y = expr();
            nodes[last].next = (i16)y;
            last = y;
        }
        expect(P_RPAR);
        return mk(N_TUPLE, 0, x, -1, -1);
    }
    if (accept_op(P_LSQ)) {
        if (accept_op(P_RSQ)) return mk(N_LIST, 0, -1, -1, -1);
        int x = expr();
        if (is_kw(KW_FOR)) {
            int c = comp_clauses();
            expect(P_RSQ);
            return mk(N_COMP, 0, x, c, -1);
        }
        int last = x;
        while (accept_op(P_COMMA) && !is_op(P_RSQ)) {
            int y = expr();
            nodes[last].next = (i16)y;
            last = y;
        }
        expect(P_RSQ);
        return mk(N_LIST, 0, x, -1, -1);
    }
    if (accept_op(P_LBR)) {
        if (accept_op(P_RBR)) return mk(N_DICT, 0, -1, -1, -1);
        int k = expr();
        if (!is_op(P_COLON)) err("SyntaxError", "sets are not supported (a dict's key needs its : value)");
        ti++;
        int v = expr();
        if (is_kw(KW_FOR)) {
            int c = comp_clauses();
            expect(P_RBR);
            return mk(N_DICTCOMP, 0, k, v, c);
        }
        nodes[k].next = (i16)v;
        int last = v;
        while (accept_op(P_COMMA) && !is_op(P_RBR)) {
            int k2 = expr();
            expect(P_COLON);
            int v2 = expr();
            nodes[last].next = (i16)k2;
            nodes[k2].next = (i16)v2;
            last = v2;
        }
        expect(P_RBR);
        return mk(N_DICT, 0, k, -1, -1);
    }
    if (t->type == T_NL || t->type == T_END) err("SyntaxError", "the line ends where a value should be");
    err3("SyntaxError", "unexpected ", tok_text(t), "");
}

static int primary(void) {
    int x = atom();
    for (;;) {
        line = tk()->line;
        if (accept_op(P_LPAR)) {
            int args = -1, last = -1;
            while (!is_op(P_RPAR)) {
                int a;
                if (tk()->type == T_NAME && toks[ti + 1].type == T_OP && toks[ti + 1].code == P_EQ) {
                    int name = intern(tok_text(tk()));
                    ti += 2;
                    a = mk(N_KW, 0, name, expr(), -1);
                } else if (is_op(P_STAR))
                    err("SyntaxError", "*args are not supported");
                else {
                    a = expr();
                    if (is_kw(KW_FOR)) a = mk(N_COMP, 0, a, comp_clauses(), -1);
                }
                if (last < 0) args = a;
                else nodes[last].next = (i16)a;
                last = a;
                if (!accept_op(P_COMMA)) break;
            }
            expect(P_RPAR);
            x = mk(N_CALL, 0, x, args, -1);
        } else if (accept_op(P_LSQ)) {
            int a = -1, b = -1, c = -1, slice = 0;
            if (!is_op(P_COLON)) a = expr();
            if (accept_op(P_COLON)) {
                slice = 1;
                if (!is_op(P_RSQ) && !is_op(P_COLON)) b = expr();
                if (accept_op(P_COLON) && !is_op(P_RSQ)) c = expr();
            } else if (is_op(P_COMMA)) {
                int last = a;
                while (accept_op(P_COMMA) && !is_op(P_RSQ)) {
                    int y = expr();
                    nodes[last].next = (i16)y;
                    last = y;
                }
                a = mk(N_TUPLE, 0, a, -1, -1);
            }
            expect(P_RSQ);
            x = slice ? mk(N_SLICE, 0, x, mk(N_LIST, 0, -1, -1, -1), -1) : mk(N_INDEX, 0, x, a, -1);
            if (slice) { /* (a, b, c in the slice node's list node: a its a, b its b, c its c) */
                int l = nodes[x].b;
                nodes[l].a = (i16)a, nodes[l].b = (i16)b, nodes[l].c = (i16)c;
            }
        } else if (accept_op(P_DOT)) {
            if (tk()->type != T_NAME) err("SyntaxError", "a name after .");
            x = mk(N_ATTR, 0, x, intern(tok_text(tk())), -1);
            ti++;
        } else
            return x;
    }
}

enum { OP_ADD = 1, OP_SUB, OP_MUL, OP_DIV, OP_FLOORDIV, OP_MOD, OP_POW, OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR, OP_NEG,
       OP_POS, OP_INV, OP_LT, OP_LE, OP_GT, OP_GE, OP_EQ, OP_NE, OP_IN, OP_NOTIN, OP_IS, OP_ISNOT };

static int unary(void);

static int power(void) {
    int x = primary();
    if (accept_op(P_STAR_STAR)) return mk(N_BIN, OP_POW, x, unary(), -1);
    return x;
}

static int unary(void) {
    PARSE_DEEP();
    line = tk()->line;
    if (accept_op(P_MINUS)) return mk(N_UNARY, OP_NEG, unary(), -1, -1);
    if (accept_op(P_PLUS)) return mk(N_UNARY, OP_POS, unary(), -1, -1);
    if (accept_op(P_TILDE)) return mk(N_UNARY, OP_INV, unary(), -1, -1);
    return power();
}

static int bin_prec(int *op) { /* the next token's binding as a binary operator (6: * / // %, then + -, << >>, &, ^,
                                  1: |; 0: none) and its node's op */
    if (toks[ti].type != T_OP) return 0;
    switch (toks[ti].code) {
    case P_STAR: return *op = OP_MUL, 6;
    case P_SLASH: return *op = OP_DIV, 6;
    case P_SLASH_SLASH: return *op = OP_FLOORDIV, 6;
    case P_PCT: return *op = OP_MOD, 6;
    case P_PLUS: return *op = OP_ADD, 5;
    case P_MINUS: return *op = OP_SUB, 5;
    case P_LT_LT: return *op = OP_SHL, 4;
    case P_GT_GT: return *op = OP_SHR, 4;
    case P_AMP: return *op = OP_AND, 3;
    case P_CARET: return *op = OP_XOR, 2;
    case P_BAR: return *op = OP_OR, 1;
    }
    return 0;
}

static int climb(int x, int least) { /* x and the operators that follow binding at least `least` (left to right) */
    int op, p, op2;
    while ((p = bin_prec(&op)) >= least) {
        PARSE_DEEP();
        line = tk()->line;
        ti++;
        int y = unary();
        while (bin_prec(&op2) > p) y = climb(y, p + 1);
        x = mk(N_BIN, op, x, y, -1);
    }
    return x;
}

static int binary(void) { return climb(unary(), 1); } /* | ^ & << >> + - * / // % */

static int comparison(void) {
    int x = binary();
    int first = -1, last = -1;
    for (;;) {
        int op = is_op(P_LT) ? OP_LT : is_op(P_LT_EQ) ? OP_LE : is_op(P_GT) ? OP_GT : is_op(P_GT_EQ) ? OP_GE : is_op(P_EQ_EQ) ? OP_EQ
                 : is_op(P_BANG_EQ) ? OP_NE : is_kw(KW_IN) && !no_in ? OP_IN : is_kw(KW_IS) ? OP_IS : is_kw(KW_NOT) && toks[ti + 1].code == KW_IN ? OP_NOTIN : 0;
        if (!op) break;
        ti += op == OP_NOTIN ? 2 : 1;
        if (op == OP_IS && accept_kw(KW_NOT)) op = OP_ISNOT;
        int y = mk(N_CMP, op, binary(), -1, -1);
        if (last < 0) first = y;
        else nodes[last].next = (i16)y;
        last = y;
    }
    return first < 0 ? x : mk(N_CMP, 0, x, first, -1);
}

static int not_expr(void) {
    PARSE_DEEP();
    if (accept_kw(KW_NOT)) return mk(N_NOT, 0, not_expr(), -1, -1);
    return comparison();
}

static int and_expr(void) {
    int x = not_expr();
    while (accept_kw(KW_AND)) x = mk(N_AND, 0, x, not_expr(), -1);
    return x;
}

static int or_expr(void) {
    int x = and_expr();
    while (accept_kw(KW_OR)) x = mk(N_OR, 0, x, and_expr(), -1);
    return x;
}

static int expr(void) {
    int x = or_expr();
    if (is_kw(KW_IF) && toks[ti].type == T_NAME) {
        int at = ti;
        ti++;
        int c = or_expr();
        if (!accept_kw(KW_ELSE)) { /* (a comprehension's if: not this one's) */
            ti = at;
            return x;
        }
        return mk(N_IF_EXP, 0, c, x, expr());
    }
    return x;
}

static int expr_list(int *tuple) { /* a, b, c: a tuple; one alone itself */
    int x = expr(), last = x;
    *tuple = 0;
    while (is_op(P_COMMA)) {
        ti++;
        *tuple = 1;
        if (tk()->type == T_NL || is_op(P_EQ) || is_op(P_RPAR) || is_kw(KW_IN) || tk()->type == T_END) break;
        int y = expr();
        nodes[last].next = (i16)y;
        last = y;
    }
    return *tuple ? mk(N_TUPLE, 0, x, -1, -1) : x;
}

static void end_of_statement(void) {
    line = tk()->line;
    if (accept_op(P_SEMI)) return;
    if (tk()->type == T_NL) {
        ti++;
        return;
    }
    if (tk()->type == T_END || tk()->type == T_DEDENT) return;
    err3("SyntaxError", "unexpected ", tok_text(tk()), " after a statement");
}

static int simple(void) {
    token *t = tk();
    line = t->line;
    static const u8 NO[] = {KW_TRY, KW_EXCEPT, KW_CLASS, KW_WITH, KW_YIELD, KW_RAISE, KW_ASYNC, KW_AWAIT, KW_NONLOCAL, 0};
    for (int i = 0; NO[i]; i++)
        if (is_kw(NO[i])) err3("SyntaxError", "", tok_text(t), " is not supported on the computer");
    if (accept_kw(KW_PASS)) return mk(S_PASS, 0, -1, -1, -1);
    if (accept_kw(KW_BREAK)) return mk(S_BREAK, 0, -1, -1, -1);
    if (accept_kw(KW_CONTINUE)) return mk(S_CONTINUE, 0, -1, -1, -1);
    if (accept_kw(KW_RETURN)) {
        int tuple;
        int x = tk()->type == T_NL || is_op(P_SEMI) || tk()->type == T_END ? -1 : expr_list(&tuple);
        return mk(S_RETURN, 0, x, -1, -1);
    }
    if (accept_kw(KW_ASSERT)) {
        int x = expr();
        return mk(S_EXPR, 1, x, -1, -1); /* (op 1: an assert) */
    }
    if (accept_kw(KW_GLOBAL)) {
        int first = -1, last = -1;
        do {
            int x = mk(N_NAME, 0, intern(tok_text(tk())), -1, -1);
            ti++;
            if (last < 0) first = x;
            else nodes[last].next = (i16)x;
            last = x;
        } while (accept_op(P_COMMA));
        return mk(S_GLOBAL, 0, first, -1, -1);
    }
    if (accept_kw(KW_DEL)) {
        int tuple;
        return mk(S_DEL, 0, expr_list(&tuple), -1, -1);
    }
    if (is_kw(KW_IMPORT) || is_kw(KW_FROM)) { /* import math; from math import gcd: math's functions are builtins here */
        int from = is_kw(KW_FROM);
        ti++;
        str m = tok_text(tk());
        if (!str_is(m, "math") && !str_is(m, "functools") && !str_is(m, "itertools") && !str_is(m, "string"))
            err3("ImportError", "no module ", m, " on the computer (math only)");
        while (tk()->type != T_NL && tk()->type != T_END && !is_op(P_SEMI)) ti++;
        (void)from;
        return mk(S_PASS, 0, -1, -1, -1);
    }
    int tuple;
    int x = expr_list(&tuple);
    static const u8 AUG[] = {P_PLUS_EQ, P_MINUS_EQ, P_STAR_EQ, P_SLASH_EQ, P_SLASH_SLASH_EQ, P_PCT_EQ, P_STAR_STAR_EQ, P_AMP_EQ,
                             P_BAR_EQ, P_CARET_EQ, P_LT_LT_EQ, P_GT_GT_EQ, 0};
    static const u8 AUGOP[] = {OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_FLOORDIV, OP_MOD, OP_POW, OP_AND, OP_OR, OP_XOR, OP_SHL, OP_SHR};
    for (int i = 0; AUG[i]; i++)
        if (is_op(AUG[i])) {
            ti++;
            return mk(S_AUG, AUGOP[i], x, expr(), -1);
        }
    if (is_op(P_EQ)) { /* a = b = value too */
        int targets = x, last = x;
        int value = -1;
        while (accept_op(P_EQ)) {
            int y = expr_list(&tuple);
            if (is_op(P_EQ)) {
                nodes[last].next = (i16)y;
                last = y;
            } else
                value = y;
        }
        return mk(S_ASSIGN, 0, targets, value, -1);
    }
    if (is_op(P_COLON)) err("SyntaxError", "annotations (x: int = ...) are not supported");
    return mk(S_EXPR, 0, x, -1, -1);
}

static int statement(void) {
    PARSE_DEEP();
    line = tk()->line;
    if (accept_kw(KW_IF) || is_kw(KW_ELIF)) {
        if (is_kw(KW_ELIF)) ti++;
        int c = expr();
        expect(P_COLON);
        int body = block(), other = -1;
        if (is_kw(KW_ELIF)) other = statement();
        else if (accept_kw(KW_ELSE)) {
            expect(P_COLON);
            other = block();
        }
        return mk(S_IF, 0, c, body, other);
    }
    if (accept_kw(KW_WHILE)) {
        int c = expr();
        expect(P_COLON);
        int body = block(), other = -1;
        if (accept_kw(KW_ELSE)) expect(P_COLON), other = block();
        return mk(S_WHILE, 0, c, body, other);
    }
    if (accept_kw(KW_FOR)) {
        int tuple;
        no_in++;
        int target = expr_list(&tuple);
        no_in--;
        if (!accept_kw(KW_IN)) err("SyntaxError", "a for without its in");
        int it = expr_list(&tuple);
        expect(P_COLON);
        int body = block(), other = -1;
        if (accept_kw(KW_ELSE)) expect(P_COLON), other = block();
        int f = mk(S_FOR, 0, target, it, body);
        nodes[f].next = -1;
        if (other >= 0) err("SyntaxError", "for ... else is not supported");
        return f;
    }
    if (accept_kw(KW_DEF)) {
        if (tk()->type != T_NAME) err("SyntaxError", "a def without its name");
        int name = intern(tok_text(tk()));
        ti++;
        expect(P_LPAR);
        int params = -1, last = -1;
        while (!is_op(P_RPAR)) {
            if (tk()->type != T_NAME) err("SyntaxError", "a parameter that is not a name (no *args, no **kwargs)");
            int p = mk(N_NAME, 0, intern(tok_text(tk())), -1, -1);
            ti++;
            if (accept_op(P_COLON)) expr(); /* (an annotation: left) */
            if (accept_op(P_EQ)) nodes[p].b = (i16)expr();
            if (last < 0) params = p;
            else nodes[last].next = (i16)p;
            last = p;
            if (!accept_op(P_COMMA)) break;
        }
        expect(P_RPAR);
        if (accept_op(P_MINUS_GT)) expr();
        expect(P_COLON);
        return mk(S_DEF, 0, name, params, block());
    }
    int first = simple(), last = first;
    while (is_op(P_SEMI)) {
        ti++;
        if (tk()->type == T_NL || tk()->type == T_END) break;
        int y = simple();
        nodes[last].next = (i16)y;
        last = y;
    }
    end_of_statement();
    return first == last ? first : mk(S_BLOCK, 0, first, -1, -1);
}

static int statements(int until_dedent) {
    int first = -1, last = -1;
    for (;;) {
        while (tk()->type == T_NL) ti++;
        if (tk()->type == T_END) break;
        if (tk()->type == T_DEDENT) {
            if (until_dedent) {
                ti++;
                break;
            }
            ti++;
            continue;
        }
        if (tk()->type == T_INDENT) line = tk()->line, err("IndentationError", "an unexpected indent");
        int s = statement();
        if (last < 0) first = s;
        else nodes[last].next = (i16)s;
        last = s;
        while (nodes[last].next >= 0) last = nodes[last].next;
    }
    return mk(S_BLOCK, 0, first, -1, -1);
}

static int block(void) { /* after a ":": an indented block, or statements on the same line */
    if (tk()->type == T_NL) {
        ti++;
        while (tk()->type == T_NL) ti++;
        if (tk()->type != T_INDENT) line = tk()->line, err("IndentationError", "expected an indented block");
        ti++;
        return statements(1);
    }
    int s = statement();
    return mk(S_BLOCK, 0, s, -1, -1);
}

/* ---- running: variables */
static val *globals;
static u8 *gset;
typedef struct {
    i16 name;
    i16 global; /* (declared global in this frame) */
    val v;
} local;
static local *frames;
static int n_frame, frame_base, in_function;
#define FRAME_MAX 120

static val *find_var(int name) {
    if (in_function)
        for (int i = n_frame - 1; i >= frame_base; i--)
            if (frames[i].name == name) {
                if (frames[i].global) break;
                return &frames[i].v;
            }
    return gset[name] ? &globals[name] : 0;
}

static int declared_global(int name) {
    for (int i = n_frame - 1; i >= frame_base; i--)
        if (frames[i].name == name) return frames[i].global;
    return 0;
}

static void keep_frame(int at, val v) { /* (a frame's variable from before a round: what goes in it is kept) */
    for (int i = 0; i < rounds; i++)
        if ((const char *)OBJ(v) >= round_at[i] && at < round_frame[i]) round_kept[i] = 1;
}

static void set_var(int name, val v) {
    if (in_function && !declared_global(name)) {
        for (int i = n_frame - 1; i >= frame_base; i--)
            if (frames[i].name == name) {
                frames[i].v = v;
                if (rounds && IS_OBJ(v)) keep_frame(i, v);
                return;
            }
        if (n_frame == FRAME_MAX) err("RecursionError", "too many variables in the calls going on");
        frames[n_frame++] = (local){(i16)name, 0, v};
        return;
    }
    globals[name] = v;
    KEEP(globals, v);
    if (!gset[name]) gset[name] = 1;
}

/* a round from here (-1: none, ROUNDS deep already) */
static int round_start(void) {
    if (rounds == ROUNDS) return -1;
    round_at[rounds] = heap_mark(), round_frame[rounds] = n_frame, round_kept[rounds] = 0;
    return rounds++;
}

/* the round ended: as rounds do (`free`), its memory given back if nothing of it was kept; by break or return, kept
   (what it made may be what the loop leaves) */
static void round_end(int r, int give) {
    if (r < 0) return;
    charge(3); /* (a round's start and end) */
    for (int i = round_frame[r]; i < n_frame && !round_kept[r]; i++) /* (variables the round added to its frame) */
        if (IS_OBJ(frames[i].v) && (const char *)OBJ(frames[i].v) >= round_at[r]) round_kept[r] = 1;
    if (give && !round_kept[r]) heap_back(round_at[r]);
    rounds = r;
}

/* ---- running */

static void timeout(void) {
    err("TimeoutError", "the program takes more than the computer's 1,250 steps for a run (about a loop of 250 "
                        "rounds): make it smaller, e.g. a long loop's range in parts, a call each");
}

static int ctl; /* break, continue, return in flight */
enum { C_NONE, C_BREAK, C_CONTINUE, C_RETURN };
static val ret;

static val eval(int n);
static void run(int n);
static val call_value(val f, val *args, int n_args, int kw, val kwv);

/* ---- operations */
static int equal(val a, val b) {
    if (a == b) return 1;
    DEEP();
    if ((IS_INT(a) || IS_BOOL(a)) && (IS_INT(b) || IS_BOOL(b))) return (IS_INT(a) ? INT(a) : a == TRUE_V) == (IS_INT(b) ? INT(b) : b == TRUE_V);
    int ta = type_of(a), tb = type_of(b);
    if (ta != tb || ta <= 0) return 0;
    if (ta == O_STR) {
        strobj *x = (strobj *)OBJ(a), *y = (strobj *)OBJ(b);
        if (x->h.len != y->h.len) return 0;
        charge(x->h.len);
        return !memcmp(x->s, y->s, x->h.len);
    }
    if (ta == O_LIST) {
        listobj *x = (listobj *)OBJ(a), *y = (listobj *)OBJ(b);
        if (x->h.len != y->h.len) return 0;
        charge(2 * (i64)x->h.len);
        for (int i = 0; i < x->h.len; i++)
            if (!equal(x->items[i], y->items[i])) return 0;
        return 1;
    }
    if (ta == O_DICT) {
        dictobj *x = (dictobj *)OBJ(a), *y = (dictobj *)OBJ(b);
        if (x->h.len != y->h.len) return 0;
        for (int i = 0; i < x->h.len; i++) {
            int j = dict_find(y, x->k[i]);
            if (j < 0 || !equal(x->v[i], y->v[j])) return 0;
        }
        return 1;
    }
    return 0;
}

static int compare(val a, val b) { /* <0, 0, >0; ints and bools, strings, lists */
    if ((IS_INT(a) || IS_BOOL(a)) && (IS_INT(b) || IS_BOOL(b))) {
        i64 x = IS_INT(a) ? INT(a) : a == TRUE_V, y = IS_INT(b) ? INT(b) : b == TRUE_V;
        return x < y ? -1 : x > y;
    }
    int ta = type_of(a), tb = type_of(b);
    DEEP();
    if (ta == O_STR && tb == O_STR) {
        strobj *x = (strobj *)OBJ(a), *y = (strobj *)OBJ(b);
        int n = x->h.len < y->h.len ? x->h.len : y->h.len, d = memcmp(x->s, y->s, n);
        charge(n);
        return d ? d : x->h.len - y->h.len;
    }
    if (ta == O_LIST && tb == O_LIST) {
        listobj *x = (listobj *)OBJ(a), *y = (listobj *)OBJ(b);
        for (int i = 0; i < x->h.len && i < y->h.len; i++) {
            int d = compare(x->items[i], y->items[i]);
            if (d) return d;
        }
        return x->h.len - y->h.len;
    }
    say("error (line ");
    say_int(line);
    say(": TypeError: < between ");
    say(type_name(a));
    say(" and ");
    say(type_name(b));
    put(')');
    done();
}

static int truthy(val v) {
    if (IS_INT(v)) return INT(v) != 0;
    if (v == NONE_V || v == FALSE_V) return 0;
    if (v == TRUE_V) return 1;
    head *h = OBJ(v);
    if (h->type == O_STR || h->type == O_LIST || h->type == O_DICT) return h->len != 0;
    if (h->type == O_RANGE) {
        rangeobj *r = (rangeobj *)h;
        return r->step > 0 ? r->start < r->stop : r->start > r->stop;
    }
    return 1;
}

static i64 mul_checked(i64 a, i64 b) {
    i64 x = a < 0 ? -a : a, y = b < 0 ? -b : b;
    if (x >= ((i64)1 << 31) || y >= ((i64)1 << 31)) /* (the slow test only for big ones) */
        if (x && y > MAXI / x) err("OverflowError", "an integer beyond 2^62 (the computer's; calc does bigger)");
    return a * b;
}

static i64 floordiv(i64 a, i64 b) {
    if (!b) err("ZeroDivisionError", "integer division or modulo by zero");
    i64 q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q--;
    return q;
}

static i64 pymod(i64 a, i64 b) {
    if (!b) err("ZeroDivisionError", "integer division or modulo by zero");
    i64 m = a % b;
    if (m && ((m < 0) != (b < 0))) m += b;
    return m;
}

static i64 ipow(i64 b, i64 e) {
    if (e < 0) err("ValueError", "a negative power is a fraction (no floats on the computer)");
    i64 r = 1;
    while (e) {
        if (e & 1) r = mul_checked(r, b);
        e >>= 1;
        if (e) b = mul_checked(b, b);
        step();
    }
    return r;
}

static i64 mulmod(i64 a, i64 b, i64 m) {
    if (a < ((i64)1 << 30) && b < ((i64)1 << 30)) return (a * b) % m;
    i64 r = 0;
    a %= m;
    while (b) {
        if (b & 1) r = (r + a) % m;
        a = (a * 2) % m;
        b >>= 1;
    }
    return r;
}

static val concat(val a, val b) {
    int t = type_of(a);
    if (t == O_STR) {
        strobj *x = (strobj *)OBJ(a), *y = (strobj *)OBJ(b);
        strobj *o = new_str(0, x->h.len + y->h.len);
        memcpy(o->s, x->s, x->h.len);
        memcpy(o->s + x->h.len, y->s, y->h.len);
        return MKOBJ(o);
    }
    listobj *x = (listobj *)OBJ(a), *y = (listobj *)OBJ(b), *o = new_list(x->h.len + y->h.len);
    charge(x->h.len + y->h.len);
    memcpy(o->items, x->items, sizeof(val) * x->h.len);
    memcpy(o->items + x->h.len, y->items, sizeof(val) * y->h.len);
    o->h.tuple = x->h.tuple;
    return MKOBJ(o);
}

static val repeat(val s, i64 n) {
    if (n < 0) n = 0;
    if (type_of(s) == O_STR) {
        strobj *x = (strobj *)OBJ(s);
        if ((i64)x->h.len * n > 60000) err("MemoryError", "a string too long for the computer");
        strobj *o = new_str(0, x->h.len * n);
        i64 total = x->h.len * n, done = x->h.len < total ? x->h.len : total; /* (the copy doubled: few copies) */
        memcpy(o->s, x->s, (size_t)done);
        for (; done < total; done *= 2) memcpy(o->s + done, o->s, (size_t)(done < total - done ? done : total - done));
        return MKOBJ(o);
    }
    listobj *x = (listobj *)OBJ(s);
    if ((i64)x->h.len * n > 4000) err("MemoryError", "a list too long for the computer");
    listobj *o = new_list((int)(x->h.len * n));
    charge(x->h.len * n + n);
    for (i64 i = 0; i < n; i++) memcpy(o->items + i * x->h.len, x->items, sizeof(val) * x->h.len);
    o->h.tuple = x->h.tuple;
    return MKOBJ(o);
}

static __attribute__((noreturn)) void bad_operands(const char *op, val a, val b) {
    say("error (line ");
    say_int(line);
    say(": TypeError: unsupported operand types for ");
    say(op);
    say(": ");
    say(type_name(a));
    say(" and ");
    say(type_name(b));
    put(')');
    done();
}

static int as_index(val v) { return IS_INT(v) || IS_BOOL(v); }

static val binop(int op, val a, val b) {
    static const char *const SYM[] = {"", "+", "-", "*", "/", "//", "%", "**", "&", "|", "^", "<<", ">>"};
    if (as_index(a) && as_index(b)) {
        i64 x = as_int(a, ""), y = as_int(b, "");
        switch (op) {
        case OP_ADD: return mkint(x + y);
        case OP_SUB: return mkint(x - y);
        case OP_MUL: return mkint(mul_checked(x, y));
        case OP_DIV:
            if (!y) err("ZeroDivisionError", "division by zero");
            if (x % y) err("ValueError", "/ of numbers that do not divide is a fraction (no floats: use // or calc)");
            return mkint(x / y);
        case OP_FLOORDIV: return mkint(floordiv(x, y));
        case OP_MOD: return mkint(pymod(x, y));
        case OP_POW: return mkint(ipow(x, y));
        case OP_AND: return MKINT(x & y);
        case OP_OR: return MKINT(x | y);
        case OP_XOR: return MKINT(x ^ y);
        case OP_SHL:
            if (y < 0) err("ValueError", "negative shift count");
            if (y > 62 || (x && (x < 0 ? -x : x) >= ((i64)1 << (62 - y)))) err("OverflowError", "an integer beyond 2^62");
            return MKINT(x << y);
        case OP_SHR:
            if (y < 0) err("ValueError", "negative shift count");
            return MKINT(y > 62 ? (x < 0 ? -1 : 0) : x >> y);
        }
    }
    int ta = type_of(a), tb = type_of(b);
    if (op == OP_ADD && ta == tb && (ta == O_STR || ta == O_LIST)) return concat(a, b);
    if (op == OP_MUL && (ta == O_STR || ta == O_LIST) && as_index(b)) return repeat(a, as_int(b, ""));
    if (op == OP_MUL && (tb == O_STR || tb == O_LIST) && as_index(a)) return repeat(b, as_int(a, ""));
    if (op == OP_MOD && ta == O_STR) err("TypeError", "% formatting is not supported (use an f-string)");
    bad_operands(SYM[op], a, b);
}

/* a range's, list's, string's (a character each) or dict's (its keys) items, one by one */
typedef struct {
    val src;
    int i, n;
    i64 cur, step, stop;
    int kind;
} iter;

static void iter_start(iter *it, val v) {
    int t = type_of(v);
    it->src = v, it->i = 0, it->kind = t;
    if (t == O_RANGE) {
        rangeobj *r = (rangeobj *)OBJ(v);
        it->cur = r->start, it->step = r->step, it->stop = r->stop;
    } else if (t == O_LIST || t == O_STR || t == O_DICT)
        it->n = OBJ(v)->len;
    else {
        say("error (line ");
        say_int(line);
        say(": TypeError: ");
        say(type_name(v));
        say(" is not iterable)");
        done();
    }
}

static int iter_next(iter *it, val *out) {
    if (it->kind == O_RANGE) {
        if (it->step > 0 ? it->cur >= it->stop : it->cur <= it->stop) return 0;
        *out = MKINT(it->cur);
        it->cur += it->step;
        return 1;
    }
    if (it->i >= OBJ(it->src)->len) return 0; /* (a list that shrank meanwhile) */
    if (it->kind == O_LIST) *out = ((listobj *)OBJ(it->src))->items[it->i++];
    else if (it->kind == O_DICT) *out = ((dictobj *)OBJ(it->src))->k[it->i++];
    else { /* a UTF-8 character */
        strobj *s = (strobj *)OBJ(it->src);
        int a = it->i, b = a + 1;
        while (b < s->h.len && ((u8)s->s[b] & 0xc0) == 0x80) b++;
        it->i = b;
        charge(8); /* (a character's string made) */
        *out = str_v(s->s + a, b - a);
    }
    return 1;
}

static val eval(int n);

static void iter_node(iter *it, int n) { /* the items of node n's value; range(...) without making a range object */
    node *x = &nodes[n];
    if (x->kind == N_CALL && nodes[x->a].kind == N_NAME && nodes[x->a].a == B_RANGE && !find_var(B_RANGE)) {
        val a[3];
        int k = 0;
        for (int p = x->b; p >= 0 && k < 3; p = nodes[p].next) {
            if (nodes[p].kind == N_KW) break;
            a[k++] = eval(p);
        }
        if (k >= 1) {
            line = x->line;
            it->kind = O_RANGE, it->src = NONE_V;
            it->cur = k > 1 ? as_int(a[0], "range's start") : 0;
            it->stop = as_int(a[k > 1], "range's stop");
            it->step = k > 2 ? as_int(a[2], "range's step") : 1;
            if (!it->step) err("ValueError", "range() step must not be zero");
            return;
        }
    }
    iter_start(it, eval(n));
}

static listobj *to_list(val v) { /* a new list of an iterable's items */
    iter it;
    val x;
    iter_start(&it, v);
    listobj *l = new_list(0);
    while (iter_next(&it, &x)) {
        step();
        list_push(l, x);
    }
    return l;
}

static void assign(int target, val v);

static void unpack(int first, val v) { /* a, b = v */
    listobj *l = type_of(v) == O_LIST ? (listobj *)OBJ(v) : to_list(v);
    int n = 0;
    for (int t = first; t >= 0; t = nodes[t].next) n++;
    if (n != l->h.len) {
        say("error (line ");
        say_int(line);
        say(": ValueError: ");
        say_int(n);
        say(" names for ");
        say_int(l->h.len);
        say(" values)");
        done();
    }
    int i = 0;
    for (int t = first; t >= 0; t = nodes[t].next) assign(t, l->items[i++]);
}

static i64 norm_index(i64 i, int n) {
    if (i < 0) i += n;
    if (i < 0 || i >= n) err("IndexError", "index out of range");
    return i;
}

static void assign(int target, val v) {
    node *t = &nodes[target];
    line = t->line;
    if (t->kind == N_NAME) return set_var(t->a, v);
    if (t->kind == N_TUPLE || t->kind == N_LIST) return unpack(t->a, v);
    if (t->kind == N_INDEX) {
        val o = eval(t->a), k = eval(t->b);
        int ty = type_of(o);
        if (ty == O_LIST && !OBJ(o)->tuple) {
            listobj *l = (listobj *)OBJ(o);
            l->items[norm_index(as_int(k, "an index"), l->h.len)] = v;
            KEEP(l, v);
        } else if (ty == O_DICT)
            dict_set((dictobj *)OBJ(o), k, v);
        else
            err3("TypeError", "", (str){type_name(o), type_name(o) + strlen(type_name(o))}, " does not support item assignment");
        return;
    }
    err("SyntaxError", "cannot assign to that");
}

static val get_name(int name) {
    val *p = find_var(name);
    if (p) return *p;
    if (name <= B_LAST) { /* a builtin as a value (key=len) */
        fnobj *f = alloc(sizeof(fnobj));
        f->h = (head){O_BUILTIN, 0, 0, 0};
        f->node = name, f->self = NONE_V;
        return MKOBJ(f);
    }
    err3("NameError", "name '", names[name], "' is not defined");
}

static val index_of(val o, val k) {
    int t = type_of(o);
    if (t == O_LIST) {
        listobj *l = (listobj *)OBJ(o);
        return l->items[norm_index(as_int(k, "an index"), l->h.len)];
    }
    if (t == O_STR) {
        strobj *s = (strobj *)OBJ(o);
        i64 i = norm_index(as_int(k, "an index"), s->h.len); /* (bytes: ASCII text) */
        return str_v(s->s + i, 1);
    }
    if (t == O_DICT) {
        dictobj *d = (dictobj *)OBJ(o);
        int i = dict_find(d, k);
        if (i < 0) {
            say("error (line ");
            say_int(line);
            say(": KeyError: ");
            say_val(k, 1);
            put(')');
            done();
        }
        return d->v[i];
    }
    if (t == O_RANGE) {
        rangeobj *r = (rangeobj *)OBJ(o);
        i64 n = r->step > 0 ? (r->stop - r->start + r->step - 1) / r->step : (r->start - r->stop - r->step - 1) / -r->step;
        if (n < 0) n = 0;
        return MKINT(r->start + norm_index(as_int(k, "an index"), (int)n) * r->step);
    }
    err3("TypeError", "", (str){type_name(o), type_name(o) + strlen(type_name(o))}, " is not subscriptable");
}

static __attribute__((noinline)) val slice_of(val o, val a, val b, val c) {
    int t = type_of(o);
    if (t == O_RANGE) o = MKOBJ(to_list(o)), t = O_LIST;
    if (t != O_LIST && t != O_STR) err("TypeError", "only strings and lists can be sliced");
    int n = OBJ(o)->len;
    i64 st = c == NONE_V ? 1 : as_int(c, "a step");
    if (!st) err("ValueError", "a slice step of zero");
    i64 lo, hi;
    if (st > 0) {
        lo = a == NONE_V ? 0 : as_int(a, "a slice index"), hi = b == NONE_V ? n : as_int(b, "a slice index");
        if (lo < 0) lo += n;
        if (hi < 0) hi += n;
        lo = lo < 0 ? 0 : lo > n ? n : lo, hi = hi < 0 ? 0 : hi > n ? n : hi;
    } else {
        lo = a == NONE_V ? n - 1 : as_int(a, "a slice index"), hi = b == NONE_V ? -1 : as_int(b, "a slice index");
        if (a != NONE_V && lo < 0) lo += n;
        if (b != NONE_V && hi < 0) hi += n;
        lo = lo < -1 ? -1 : lo >= n ? n - 1 : lo, hi = hi < -1 ? -1 : hi >= n ? n - 1 : hi;
    }
    i64 count = st > 0 ? (hi > lo ? (hi - lo + st - 1) / st : 0) : (lo > hi ? (lo - hi - st - 1) / -st : 0);
    if (t == O_STR) {
        strobj *s = (strobj *)OBJ(o), *r = new_str(0, (size_t)count);
        if (st == 1) memcpy(r->s, s->s + lo, (size_t)count);
        else {
            charge(count);
            int k = 0;
            for (i64 i = lo; st > 0 ? i < hi : i > hi; i += st) r->s[k++] = s->s[i];
        }
        return MKOBJ(r);
    }
    listobj *l = (listobj *)OBJ(o), *r = new_list(0);
    charge(3 * count);
    for (i64 i = lo; st > 0 ? i < hi : i > hi; i += st) list_push(r, l->items[i]);
    r->h.tuple = l->h.tuple;
    return MKOBJ(r);
}

/* the first place of n in h from `from` on, or -1 (the first byte's places found 4 bytes at a time, then compared);
   the work charged */
static int find_in(strobj *h, strobj *n, int from) {
    int nn = n->h.len, hn = h->h.len;
    if (!nn) return from <= hn ? from : -1;
    const char *s = h->s, *p = s + from, *last = s + hn - nn + 1, *q = 0; /* (places [p, last)) */
    int tries = 0;
    while (p < last && (q = find_byte(p, last, n->s[0]))) {
        tries++;
        if (!memcmp(q + 1, n->s + 1, nn - 1)) break;
        p = q + 1, q = 0;
    }
    charge(8 + ((q ? q : last) - (s + from)) / 4 + tries * (20 + nn / 4)); /* (a place compared: about a step) */
    return q ? (int)(q - s) : -1;
}

static __attribute__((noinline)) int contains(val hay, val x) {
    int t = type_of(hay);
    if (t == O_STR) {
        if (type_of(x) != O_STR) err("TypeError", "'in <string>' needs a string on its left");
        strobj *h = (strobj *)OBJ(hay), *n = (strobj *)OBJ(x);
        return find_in(h, n, 0) >= 0;
    }
    if (t == O_DICT) return dict_find((dictobj *)OBJ(hay), x) >= 0;
    if (t == O_LIST) { /* (an integer among integers at once; the others as equal() compares them) */
        listobj *l = (listobj *)OBJ(hay);
        int i = 0, found = 0;
        for (; i < l->h.len && !found; i++) {
            val y = l->items[i];
            found = y == x || (!(IS_INT(x) && IS_INT(y)) && (charge(8), equal(x, y)));
        }
        charge(3 * (i64)i);
        return found;
    }
    if (t == O_RANGE && as_index(x)) {
        rangeobj *r = (rangeobj *)OBJ(hay);
        i64 v = as_int(x, "");
        if (r->step > 0 ? v < r->start || v >= r->stop : v > r->start || v <= r->stop) return 0;
        return (v - r->start) % r->step == 0;
    }
    iter it;
    val y;
    iter_start(&it, hay);
    while (iter_next(&it, &y)) {
        charge(12);
        if (equal(x, y)) return 1;
    }
    return 0;
}

/* ---- formatting: str(), f-strings' specs */
static char *str_buf; /* (str()'s output caught here, then copied: 1 KB, made once a run) */
#define STR_BUF 512

static val to_str(val v) {
    if (type_of(v) == O_STR) return v;
    charge(12);
    if (IS_INT(v) || IS_BOOL(v) || v == NONE_V || OBJ(v)->type != O_LIST || !capture_at) {
        char *save_at = capture_at, *save_end = capture_end;
        capture_at = str_buf, capture_end = str_buf + STR_BUF;
        say_val(v, 0);
        size_t n = capture_at - str_buf;
        capture_at = save_at, capture_end = save_end;
        if (n == STR_BUF) err("MemoryError", "a value too long to write as a string on the computer");
        return str_v(str_buf, n);
    }
    err("MemoryError", "str() of a list inside another str()");
}

static __attribute__((noinline)) val format(val v, val spec_v) { /* (noinline: its buffers not in eval_rest's frame) */ /* [[fill]align][0][width][,][type: d b o x X] */
    strobj *spec = (strobj *)OBJ(spec_v);
    const char *p = spec->s, *e = p + spec->h.len;
    char fill = ' ', align = 0, type = 0;
    int width = 0, comma = 0, zero = 0;
    if (e - p >= 2 && (p[1] == '<' || p[1] == '>' || p[1] == '^')) fill = p[0], align = p[1], p += 2;
    else if (p < e && (*p == '<' || *p == '>' || *p == '^')) align = *p++;
    if (p < e && *p == '0') zero = 1, p++;
    for (; p < e && digit(*p); p++)
        if (width < 1000000) width = width * 10 + (*p - '0');
    if (p < e && (*p == ',' || *p == '_')) comma = *p++;
    if (p < e && *p == '.') err("ValueError", "a precision (.2f) needs floats: no floats on the computer");
    if (p < e) type = *p++;
    if (p < e || (type && type != 'd' && type != 'b' && type != 'o' && type != 'x' && type != 'X' && type != 's'))
        err("ValueError", "a format spec the computer does not have (it has fill, <>^, 0, width, , and d b o x X s)");
    char tmp[96]; /* (a number's sign and digits: 62 bits in binary with a separator every 4 are 77) */
    const char *body = tmp;
    int n = 0, zeros = 0; /* (zeros: the 0 padding after the sign) */
    if (as_index(v) && type != 's') {
        i64 x = as_int(v, "");
        u64 m = x < 0 ? -(u64)x : (u64)x;
        int base = type == 'b' ? 2 : type == 'o' ? 8 : type == 'x' || type == 'X' ? 16 : 10;
        char d[96];
        int k = 0, g = 0;
        do {
            if (comma && g == (base == 10 ? 3 : 4)) d[k++] = (char)comma, g = 0;
            int r = (int)(m % base);
            d[k++] = (char)(r < 10 ? '0' + r : (type == 'X' ? 'A' : 'a') + r - 10);
            m /= base, g++;
        } while (m);
        if (x < 0) tmp[n++] = '-';
        if (comma && zero && !align) { /* (the zeros grouped too, as Python's: -000,042) */
            if (width > 90) err("ValueError", "a width above 90 with , and 0 (the computer's buffer)");
            for (int grp = base == 10 ? 3 : 4; k + n < width;) {
                if (g == grp) d[k++] = (char)comma, g = 0;
                d[k++] = '0', g++;
            }
        }
        zeros = zero && !align && width > k + n ? width - k - n : 0;
        while (k) tmp[n++] = d[--k];
        if (!align) align = '>';
    } else {
        strobj *o = (strobj *)OBJ(to_str(v));
        if (o->h.len >= width) return MKOBJ(o);
        body = o->s, n = o->h.len;
        if (!align) align = '<';
    }
    if (n + zeros >= width) width = n + zeros;
    strobj *r = new_str(0, width); /* (a width beyond 60,000: a MemoryError) */
    int left = align == '<' ? 0 : align == '>' ? width - n - zeros : (width - n - zeros) / 2, k = 0;
    for (int i = 0; i < left; i++) r->s[k++] = fill;
    if (zeros && *body == '-') r->s[k++] = '-', body++, n--;
    for (int i = 0; i < zeros; i++) r->s[k++] = '0';
    memcpy(r->s + k, body, n), k += n;
    while (k < width) r->s[k++] = fill;
    return MKOBJ(r);
}

/* ---- calls */
static val builtin(int id, val *a, int n, int kw, val kwv, val self);

static val call_value(val f, val *args, int n_args, int kw, val kwv) {
    int t = type_of(f);
    if (t == O_BUILTIN) {
        fnobj *b = (fnobj *)OBJ(f);
        return builtin(b->node, args, n_args, kw, kwv, b->self);
    }
    if (t != O_FUNC) err3("TypeError", "", (str){type_name(f), type_name(f) + strlen(type_name(f))}, " is not callable");
    fnobj *fn = (fnobj *)OBJ(f);
    node *def = &nodes[fn->node];
    step(), step();
    if (stack_now() < stack_floor) err("RecursionError", "calls nested too deep for the computer's memory");
    int params = def->kind == N_LAMBDA ? def->a : def->b, body = def->kind == N_LAMBDA ? def->b : def->c;
    int base = n_frame, i = 0, saved_base = frame_base, saved_in = in_function;
    for (int p = params; p >= 0; p = nodes[p].next, i++) {
        val v;
        if (i < n_args) v = args[i];
        else if (nodes[p].b >= 0) v = eval(nodes[p].b); /* (a default) */
        else err3("TypeError", "a missing argument: ", names[nodes[p].a], "");
        if (n_frame == FRAME_MAX) err("RecursionError", "too many variables in the calls going on");
        frames[n_frame++] = (local){(i16)nodes[p].a, 0, v};
    }
    if (i < n_args) err("TypeError", "too many arguments");
    frame_base = base, in_function = 1;
    val r;
    if (def->kind == N_LAMBDA) r = eval(body);
    else {
        run(body);
        r = ctl == C_RETURN ? ret : NONE_V;
        ctl = C_NONE;
    }
    n_frame = base, frame_base = saved_base, in_function = saved_in;
    return r;
}

static val comp(int n, int mode);

static int sort_rev;

static __attribute__((noinline)) val do_call(int n) { /* (its own frame: eval_rest's stays small for the other kinds) */
    node *c = &nodes[n];
    step(); /* (a call's arguments gathered: a step's work besides the node's) */
    int callee = c->a;
    if (nodes[callee].kind == N_NAME) { /* sum / any / all / min / max of a comprehension alone: folded as it goes */
        int f = nodes[callee].a, arg = c->b;
        if ((f == B_SUM || f == B_ANY || f == B_ALL || f == B_MIN || f == B_MAX) && arg >= 0 && nodes[arg].next < 0 &&
            nodes[arg].kind == N_COMP && !find_var(f))
            return comp(arg, f);
    }
    val args[12], kwv = NONE_V, self = NONE_V;
    int n_args = 0, kw = -1, id = -1;
    for (int a = c->b; a >= 0; a = nodes[a].next) {
        if (nodes[a].kind == N_KW) {
            if (kw >= 0 && nodes[a].a != kw) {
                /* (two keywords: sorted(key=..., reverse=...)) */
                if (nodes[a].a == K_REVERSE || kw == K_REVERSE) {
                    val v = eval(nodes[a].b);
                    if (nodes[a].a == K_REVERSE) self = truthy(v) ? TRUE_V : FALSE_V;
                    else self = truthy(kwv) ? TRUE_V : FALSE_V, kw = nodes[a].a, kwv = v;
                    continue;
                }
                if ((nodes[a].a == K_SEP && kw == K_END) || (nodes[a].a == K_END && kw == K_SEP)) {
                    val v = eval(nodes[a].b);
                    self = nodes[a].a == K_END ? v : kwv; /* (print: sep in kwv, end in self) */
                    if (nodes[a].a == K_SEP) kwv = v, kw = K_SEP;
                    else kw = K_SEP;
                    continue;
                }
                err("TypeError", "more keyword arguments than the computer's builtins take");
            }
            kw = nodes[a].a;
            kwv = eval(nodes[a].b);
            continue;
        }
        if (n_args == 12) err("TypeError", "more than 12 arguments");
        args[n_args++] = eval(a);
    }
    if (nodes[callee].kind == N_ATTR) { /* a method, or math.f */
        node *at = &nodes[callee];
        if (nodes[at->a].kind == N_NAME && nodes[at->a].a == B_MATH && !find_var(B_MATH))
            return builtin(at->b, args, n_args, kw, kwv, NONE_V);
        val o = eval(at->a);
        sort_rev = self == TRUE_V; /* (sort(key=..., reverse=True): the reverse, the object in self) */
        return builtin(at->b, args, n_args, kw, kwv, o == NONE_V ? MKINT(0) : o) /* (the method's object) */;
    }
    if (nodes[callee].kind == N_NAME) {
        id = nodes[callee].a;
        if (!find_var(id) && id <= B_LAST) {
            if (id == B_PRINT || id == B_SORTED) return builtin(id, args, n_args, kw, kwv, self);
            return builtin(id, args, n_args, kw, kwv, NONE_V);
        }
    }
    if (kw >= 0) err("TypeError", "keyword arguments to a def are not supported (pass them in order)");
    return call_value(eval(callee), args, n_args, -1, NONE_V);
}

/* ---- builtins and methods */
static val sorted_list(listobj *l, val key, int desc) { /* a stable merge sort, by key(x) where key is given */
    int n = l->h.len;
    val *keys = l->items;
    if (key != NONE_V) {
        keys = alloc(sizeof(val) * (n + 1));
        for (int i = 0; i < n; i++) keys[i] = call_value(key, &l->items[i], 1, -1, NONE_V);
    }
    int *a = alloc(sizeof(int) * (n + 1)), *b = alloc(sizeof(int) * (n + 1));
    for (int i = 0; i < n; i++) a[i] = i;
    for (int w = 1; w < n; w *= 2) {
        for (int lo = 0; lo < n; lo += 2 * w) {
            int mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n, i = lo, j = mid, k = lo;
            while (i < mid && j < hi) {
                step();
                int d = compare(keys[a[j]], keys[a[i]]);
                b[k++] = (desc ? d > 0 : d < 0) ? a[j++] : a[i++];
            }
            while (i < mid) b[k++] = a[i++];
            while (j < hi) b[k++] = a[j++];
        }
        int *t = a;
        a = b, b = t;
    }
    val *items = alloc(sizeof(val) * (n + 1));
    for (int i = 0; i < n; i++) items[i] = l->items[a[i]];
    memcpy(l->items, items, sizeof(val) * n);
    return MKOBJ(l);
}

static strobj *str_arg(val v, const char *what) {
    if (type_of(v) != O_STR) err3("TypeError", what, (str){0, 0}, " must be a string");
    return (strobj *)OBJ(v);
}

static i64 isqrt_(i64 n) { /* digit by digit, two bits a step (in 32 bits where n fits them) */
    if (n < 0) err("ValueError", "isqrt of a negative number");
    if (n < ((i64)1 << 32)) {
        u32 m = (u32)n, r = 0, b = 1u << 30;
        while (b > m) b >>= 2;
        while (b) {
            if (m >= r + b) m -= r + b, r = (r >> 1) + b;
            else r >>= 1;
            b >>= 2;
        }
        return r;
    }
    u64 m = (u64)n, r = 0, b = (u64)1 << 62;
    while (b > m) b >>= 2;
    while (b) {
        if (m >= r + b) m -= r + b, r = (r >> 1) + b;
        else r >>= 1;
        b >>= 2;
    }
    return (i64)r;
}

static i64 gcd_(i64 a, i64 b) {
    a = a < 0 ? -a : a, b = b < 0 ? -b : b;
    while (b) {
        i64 t = a % b;
        a = b, b = t;
        step();
    }
    return a;
}

static int memchr_(strobj *s, char c) {
    for (int i = 0; i < s->h.len; i++)
        if (s->s[i] == c) return 1;
    return 0;
}

static val str_method(int id, strobj *s, val *a, int n) {
    const char *p = s->s;
    int len = s->h.len;
    switch (id) {
    case M_UPPER: case M_LOWER: case M_TITLE: case M_CAPITALIZE: {
        strobj *r = new_str(p, len);
        charge(len);
        int word = 0;
        for (int i = 0; i < len; i++) {
            char c = r->s[i];
            int up = id == M_UPPER || ((id == M_TITLE && !word) || (id == M_CAPITALIZE && i == 0));
            if (id == M_UPPER || id == M_LOWER) up = id == M_UPPER;
            r->s[i] = up ? ((unsigned)(c - 'a') < 26 ? c - 32 : c) : ((unsigned)(c - 'A') < 26 ? c + 32 : c);
            word = (unsigned)((c | 32) - 'a') < 26;
        }
        return MKOBJ(r);
    }
    case M_STRIP: case M_LSTRIP: case M_RSTRIP: {
        int lo = 0, hi = len;
        strobj *chars = n ? str_arg(a[0], "strip's argument") : 0;
#define STRIPS(c) (chars ? memchr_(chars, c) : space(c))
        if (id != M_RSTRIP)
            while (lo < hi && STRIPS(p[lo])) lo++;
        if (id != M_LSTRIP)
            while (hi > lo && STRIPS(p[hi - 1])) hi--;
        charge((i64)(lo + len - hi) * (chars ? 1 + chars->h.len / 2 : 1));
        return str_v(p + lo, hi - lo);
    }
    case M_SPLIT: case M_SPLITLINES: {
        listobj *l = new_list(0);
        i64 most = id == M_SPLIT && n > 1 ? as_int(a[1], "split's maxsplit") : -1; /* (splits at most; -1: all) */
        if (id == M_SPLITLINES || !n || a[0] == NONE_V) {
            int i = 0;
            charge(len);
            while (i < len) {
                while (i < len && (id == M_SPLITLINES ? p[i] == '\n' : space(p[i])) && id != M_SPLITLINES) i++;
                if (i >= len) break;
                if (most >= 0 && l->h.len == most) { /* (the rest as it is, its spaces at the end too) */
                    list_push(l, str_v(p + i, len - i));
                    break;
                }
                int j = i;
                while (j < len && (id == M_SPLITLINES ? p[j] != '\n' : !space(p[j]))) j++;
                list_push(l, str_v(p + i, j - i));
                i = id == M_SPLITLINES ? j + 1 : j;
            }
            return MKOBJ(l);
        }
        strobj *sep = str_arg(a[0], "split's separator");
        if (!sep->h.len) err("ValueError", "an empty separator");
        int i = 0;
        for (int j; (most < 0 || l->h.len < most) && (j = find_in(s, sep, i)) >= 0; i = j + sep->h.len)
            list_push(l, str_v(p + i, j - i));
        list_push(l, str_v(p + i, len - i));
        return MKOBJ(l);
    }
    case M_JOIN: {
        if (n != 1) err("TypeError", "join takes one argument");
        listobj *l = type_of(a[0]) == O_LIST ? (listobj *)OBJ(a[0]) : to_list(a[0]);
        int total = 0;
        charge(7 * (i64)l->h.len);
        for (int i = 0; i < l->h.len; i++) total += str_arg(l->items[i], "join's items")->h.len + (i ? len : 0);
        strobj *r = new_str(0, total);
        int k = 0;
        for (int i = 0; i < l->h.len; i++) {
            strobj *x = (strobj *)OBJ(l->items[i]);
            if (i) memcpy(r->s + k, p, len), k += len;
            memcpy(r->s + k, x->s, x->h.len), k += x->h.len;
        }
        return MKOBJ(r);
    }
    case M_REPLACE: case M_COUNT: case M_FIND: case M_RFIND: case M_INDEX: case M_STARTSWITH: case M_ENDSWITH: {
        if (n < 1) err("TypeError", "a missing argument");
        strobj *x = str_arg(a[0], "the argument");
        int m = x->h.len;
        charge(m);
        if (id == M_STARTSWITH) return m <= len && !memcmp(p, x->s, m) ? TRUE_V : FALSE_V;
        if (id == M_ENDSWITH) return m <= len && !memcmp(p + len - m, x->s, m) ? TRUE_V : FALSE_V;
        if (id == M_FIND || id == M_INDEX || id == M_RFIND) {
            int found = find_in(s, x, 0);
            if (id == M_RFIND)
                for (int at = found; at >= 0 && at < len; at = find_in(s, x, at + 1)) found = at;
            if (found < 0 && id == M_INDEX) err("ValueError", "substring not found");
            return MKINT(found);
        }
        if (id == M_COUNT) {
            int c = 0;
            if (!m) return MKINT(len + 1);
            for (int at = find_in(s, x, 0); at >= 0; at = find_in(s, x, at + m)) c++;
            return MKINT(c);
        }
        strobj *y = str_arg(n > 1 ? a[1] : NONE_V, "replace's new text");
        if (!m) err("ValueError", "replace of an empty string is not supported");
        /* (as split at most count times, joined: count -1, or none, all) */
        listobj *parts = (listobj *)OBJ(str_method(M_SPLIT, s, (val[]){a[0], n > 2 ? a[2] : MKINT(-1)}, 2));
        val joined = str_method(M_JOIN, y, (val[]){MKOBJ(parts)}, 1);
        return joined;
    }
    case M_ISDIGIT: case M_ISALPHA: case M_ISALNUM: case M_ISSPACE: case M_ISUPPER: case M_ISLOWER: {
        if (!len) return FALSE_V;
        charge(len);
        int cased = 0;
        for (int i = 0; i < len; i++) {
            u8 c = p[i];
            int d = digit(c), al = (unsigned)((c | 32) - 'a') < 26 || c >= 0x80;
            int ok = id == M_ISDIGIT ? d : id == M_ISALPHA ? al : id == M_ISALNUM ? d || al : id == M_ISSPACE ? space(c)
                     : id == M_ISUPPER ? !((unsigned)(c - 'a') < 26) : !((unsigned)(c - 'A') < 26);
            cased |= (unsigned)((c | 32) - 'a') < 26;
            if (!ok) return FALSE_V;
        }
        return (id == M_ISUPPER || id == M_ISLOWER) && !cased ? FALSE_V : TRUE_V;
    }
    case M_ZFILL: case M_RJUST: case M_LJUST: case M_CENTER: {
        int w = (int)as_int(n ? a[0] : MKINT(0), "the width");
        char fill = id == M_ZFILL ? '0' : n > 1 ? str_arg(a[1], "the fill")->s[0] : ' ';
        if (w <= len) return MKOBJ(s);
        strobj *r = new_str(0, w);
        int left = id == M_LJUST ? 0 : id == M_CENTER ? (w - len) / 2 : w - len, k = 0;
        if (id == M_ZFILL && len && (p[0] == '-' || p[0] == '+')) r->s[k++] = p[0], p++, len--, left = w - len - 1;
        for (int i = 0; i < left; i++) r->s[k++] = fill;
        memcpy(r->s + k, p, len), k += len;
        while (k < w) r->s[k++] = fill;
        return MKOBJ(r);
    }
    }
    err3("AttributeError", "str has no method ", names[id], "");
}

/* the positional arguments each builtin and method needs at least (a call with fewer: a TypeError) */
static const u8 MIN_ARGS[N_FIXED] = {
    [B_LEN] = 1, [B_RANGE] = 1, [B_ABS] = 1, [B_MIN] = 1, [B_MAX] = 1, [B_SUM] = 1, [B_SORTED] = 1, [B_REVERSED] = 1,
    [B_ENUMERATE] = 1, [B_ANY] = 1, [B_ALL] = 1, [B_DIVMOD] = 2, [B_POW] = 2, [B_CHR] = 1, [B_ORD] = 1, [B_BIN] = 1,
    [B_HEX] = 1, [B_OCT] = 1, [B_MAP] = 2, [B_FILTER] = 2, [B_ISQRT] = 1, [B_FACTORIAL] = 1, [B_COMB] = 1,
    [B_PERM] = 1, [B_PROD] = 1, [M_APPEND] = 1, [M_INSERT] = 2, [M_EXTEND] = 1, [M_INDEX] = 1, [M_COUNT] = 1,
    [M_REMOVE] = 1, [M_JOIN] = 1, [M_REPLACE] = 2, [M_FIND] = 1, [M_STARTSWITH] = 1, [M_ENDSWITH] = 1, [M_ZFILL] = 1,
    [M_GET] = 1, [M_SETDEFAULT] = 1, [M_RJUST] = 1, [M_LJUST] = 1, [M_CENTER] = 1, [M_RFIND] = 1};

/* d with a dict's items, or (key, value) pairs' */
static void dict_fill(dictobj *d, val v) {
    if (type_of(v) == O_DICT) {
        dictobj *o = (dictobj *)OBJ(v);
        charge(3 * (i64)o->h.len);
        for (int i = 0; i < o->h.len; i++) dict_set(d, o->k[i], o->v[i]);
        return;
    }
    listobj *l = to_list(v);
    for (int i = 0; i < l->h.len; i++) {
        val x = l->items[i];
        if (type_of(x) != O_LIST || OBJ(x)->len != 2) err("TypeError", "a dict is made of (key, value) pairs");
        dict_set(d, ((listobj *)OBJ(x))->items[0], ((listobj *)OBJ(x))->items[1]);
    }
}

static val builtin(int id, val *a, int n, int kw, val kwv, val self) {
    step(), step(); /* (a builtin's call and dispatch: about two steps' work) */
    if (id < N_FIXED && n < MIN_ARGS[id]) err3("TypeError", "", names[id], "() is missing an argument");
    if (self != NONE_V && id > B_LAST) { /* methods */
        int t = type_of(self);
        if (t == O_STR) return str_method(id, (strobj *)OBJ(self), a, n);
        if (t == O_LIST) {
            listobj *l = (listobj *)OBJ(self);
            switch (id) {
            case M_APPEND: list_push(l, a[0]); return NONE_V;
            case M_EXTEND: {
                listobj *x = to_list(a[0]);
                for (int i = 0; i < x->h.len; i++) list_push(l, x->items[i]);
                return NONE_V;
            }
            case M_POP: {
                if (!l->h.len) err("IndexError", "pop from an empty list");
                i64 i = n ? norm_index(as_int(a[0], "an index"), l->h.len) : l->h.len - 1;
                val v = l->items[i];
                memmove(l->items + i, l->items + i + 1, sizeof(val) * (l->h.len - i - 1));
                l->h.len--;
                return v;
            }
            case M_INSERT: {
                i64 i = as_int(a[0], "an index");
                if (i < 0) i += l->h.len;
                if (i < 0) i = 0;
                if (i > l->h.len) i = l->h.len;
                list_push(l, NONE_V);
                memmove(l->items + i + 1, l->items + i, sizeof(val) * (l->h.len - 1 - i));
                l->items[i] = a[1];
                KEEP(l, a[1]);
                return NONE_V;
            }
            case M_SORT: sorted_list(l, kw == K_KEY ? kwv : NONE_V, kw == K_REVERSE ? truthy(kwv) : sort_rev); return NONE_V;
            case M_REVERSE:
                for (int i = 0, j = l->h.len - 1; i < j; i++, j--) {
                    val t2 = l->items[i];
                    l->items[i] = l->items[j], l->items[j] = t2;
                }
                return NONE_V;
            case M_INDEX: case M_COUNT: case M_REMOVE: {
                int c = 0;
                charge(5 * (i64)l->h.len);
                for (int i = 0; i < l->h.len; i++)
                    if (equal(l->items[i], a[0])) {
                        if (id == M_INDEX) return MKINT(i);
                        if (id == M_REMOVE) {
                            memmove(l->items + i, l->items + i + 1, sizeof(val) * (l->h.len - i - 1));
                            l->h.len--;
                            return NONE_V;
                        }
                        c++;
                    }
                if (id == M_COUNT) return MKINT(c);
                err("ValueError", "the value is not in the list");
            }
            case M_COPY: {
                listobj *r = new_list(l->h.len);
                memcpy(r->items, l->items, sizeof(val) * l->h.len);
                return MKOBJ(r);
            }
            case M_CLEAR: l->h.len = 0; return NONE_V;
            }
        }
        if (t == O_DICT) {
            dictobj *d = (dictobj *)OBJ(self);
            switch (id) {
            case M_GET: {
                int i = dict_find(d, a[0]);
                return i >= 0 ? d->v[i] : n > 1 ? a[1] : NONE_V;
            }
            case M_SETDEFAULT: {
                int i = dict_find(d, a[0]);
                if (i >= 0) return d->v[i];
                dict_set(d, a[0], n > 1 ? a[1] : NONE_V);
                return n > 1 ? a[1] : NONE_V;
            }
            case M_POP: {
                if (!n) err("TypeError", "pop() of a dict is missing its key");
                int i = dict_find(d, a[0]);
                if (i < 0) {
                    if (n > 1) return a[1];
                    err("KeyError", "the key is not in the dict");
                }
                val v = d->v[i];
                memmove(d->k + i, d->k + i + 1, sizeof(val) * (d->h.len - i - 1));
                memmove(d->v + i, d->v + i + 1, sizeof(val) * (d->h.len - i - 1));
                d->h.len--;
                return v;
            }
            case M_KEYS: case M_VALUES: case M_ITEMS: {
                listobj *r = new_list(0);
                charge((id == M_ITEMS ? 8 : 3) * (i64)d->h.len);
                for (int i = 0; i < d->h.len; i++) {
                    if (id == M_ITEMS) {
                        listobj *p = new_list(2);
                        p->items[0] = d->k[i], p->items[1] = d->v[i], p->h.tuple = 1;
                        list_push(r, MKOBJ(p));
                    } else
                        list_push(r, id == M_KEYS ? d->k[i] : d->v[i]);
                }
                return MKOBJ(r);
            }
            case M_UPDATE:
                if (n) dict_fill(d, a[0]);
                if (kw >= 0) dict_set(d, str_v(names[kw].p, names[kw].e - names[kw].p), kwv);
                return NONE_V;
            case M_COPY: {
                dictobj *r = new_dict();
                for (int i = 0; i < d->h.len; i++) dict_set(r, d->k[i], d->v[i]);
                return MKOBJ(r);
            }
            case M_CLEAR: d->h.len = 0; return NONE_V;
            }
        }
        err3("AttributeError", "no method ", names[id], " of that type");
    }
    switch (id) {
    case B_PRINT: {
        val sep = kw == K_SEP ? kwv : NONE_V, end = kw == K_END ? kwv : self != NONE_V && kw == K_SEP ? self : NONE_V;
        for (int i = 0; i < n; i++) {
            if (i) sep == NONE_V ? put(' ') : say_val(sep, 0);
            say_val(a[i], 0);
        }
        if (end == NONE_V) put('\n');
        else say_val(end, 0);
        return NONE_V;
    }
    case B_LEN: {
        int t = type_of(a[0]);
        if (t == O_STR) { /* (characters) */
            strobj *s = (strobj *)OBJ(a[0]);
            int c = 0;
            charge(s->h.len / 2);
            for (int i = 0; i < s->h.len; i++) c += ((u8)s->s[i] & 0xc0) != 0x80;
            return MKINT(c);
        }
        if (t == O_LIST || t == O_DICT) return MKINT(OBJ(a[0])->len);
        if (t == O_RANGE) {
            rangeobj *r = (rangeobj *)OBJ(a[0]);
            i64 k = r->step > 0 ? (r->stop - r->start + r->step - 1) / r->step : (r->start - r->stop - r->step - 1) / -r->step;
            return MKINT(k < 0 ? 0 : k);
        }
        err("TypeError", "len() of something without a length");
    }
    case B_RANGE: {
        rangeobj *r = alloc(sizeof(rangeobj));
        r->h = (head){O_RANGE, 0, 0, 0};
        r->start = n > 1 ? as_int(a[0], "range's start") : 0;
        r->stop = as_int(a[n > 1], "range's stop");
        r->step = n > 2 ? as_int(a[2], "range's step") : 1;
        if (!r->step) err("ValueError", "range() step must not be zero");
        return MKOBJ(r);
    }
    case B_ABS: {
        i64 x = as_int(a[0], "abs's argument");
        return MKINT(x < 0 ? -x : x);
    }
    case B_MIN: case B_MAX: case B_SUM: case B_ANY: case B_ALL: case B_PROD: {
        listobj *l = n == 1 || (n == 2 && id == B_SUM) ? (type_of(a[0]) == O_LIST ? (listobj *)OBJ(a[0]) : to_list(a[0])) : 0;
        val *items = l ? l->items : a;
        int m = l ? l->h.len : n;
        charge(6 * (i64)m);
        if (id == B_SUM || id == B_PROD) {
            if (id == B_SUM && m && type_of(items[0]) == O_STR) err("TypeError", "sum() of strings: use ''.join(...)");
            i64 s = id == B_SUM ? (l && n > 1 ? as_int(a[1], "") : 0) : 1;
            if (!l && id == B_SUM) err("TypeError", "sum() takes a list");
            for (int i = 0; i < m; i++) s = id == B_SUM ? s + as_int(items[i], "a summand") : mul_checked(s, as_int(items[i], "a factor")),
                                        (void)mkint(s);
            return mkint(s);
        }
        if (id == B_ANY || id == B_ALL) {
            for (int i = 0; i < m; i++)
                if (truthy(items[i]) == (id == B_ANY)) return id == B_ANY ? TRUE_V : FALSE_V;
            return id == B_ANY ? FALSE_V : TRUE_V;
        }
        if (!m) err("ValueError", "min() or max() of nothing");
        val best = items[0], bk = kw == K_KEY ? call_value(kwv, &items[0], 1, -1, NONE_V) : best;
        for (int i = 1; i < m; i++) {
            val k = kw == K_KEY ? call_value(kwv, &items[i], 1, -1, NONE_V) : items[i];
            int d = IS_INT(k) && IS_INT(bk) ? (k > bk) - (k < bk) : compare(k, bk); /* (MKINT keeps the order) */
            if (id == B_MIN ? d < 0 : d > 0) best = items[i], bk = k;
        }
        return best;
    }
    case B_SORTED: {
        listobj *l = to_list(a[0]);
        val key = kw == K_KEY ? kwv : NONE_V;
        int desc = (kw == K_REVERSE && truthy(kwv)) || self == TRUE_V;
        if (kw == K_REVERSE) key = NONE_V;
        return sorted_list(l, key, desc);
    }
    case B_REVERSED: {
        listobj *l = to_list(a[0]);
        for (int i = 0, j = l->h.len - 1; i < j; i++, j--) {
            val t = l->items[i];
            l->items[i] = l->items[j], l->items[j] = t;
        }
        return MKOBJ(l);
    }
    case B_LIST: case B_TUPLE: {
        listobj *l = n ? to_list(a[0]) : new_list(0);
        l->h.tuple = id == B_TUPLE;
        return MKOBJ(l);
    }
    case B_DICT: {
        dictobj *d = new_dict();
        if (n) dict_fill(d, a[0]);
        if (kw >= 0) dict_set(d, str_v(names[kw].p, names[kw].e - names[kw].p), kwv); /* (dict(a=1)) */
        return MKOBJ(d);
    }
    case B_STR: return n ? to_str(a[0]) : zstr("");
    case B_INT: {
        if (!n) return MKINT(0);
        if (as_index(a[0])) return MKINT(as_int(a[0], ""));
        strobj *s = str_arg(a[0], "int()'s argument");
        int base = n > 1 ? (int)as_int(a[1], "the base") : 10;
        str t = trim((str){s->s, s->s + s->h.len});
        int neg = t.p < t.e && *t.p == '-';
        if (t.p < t.e && (*t.p == '-' || *t.p == '+')) t.p++;
        if (base == 0 || base == 16 || base == 2 || base == 8)
            if (t.e - t.p > 2 && t.p[0] == '0' && ((t.p[1] | 32) == 'x' || (t.p[1] | 32) == 'b' || (t.p[1] | 32) == 'o')) {
                base = (t.p[1] | 32) == 'x' ? 16 : (t.p[1] | 32) == 'b' ? 2 : 8;
                t.p += 2;
            }
        if (!base) base = 10;
        if (base < 2 || base > 36) err("ValueError", "int() base must be 2 to 36");
        i64 v = 0, lim = MAXI / base; /* (v * base + d past MAXI: v above lim, or at it with d past MAXI's last digit) */
        int last = (int)(MAXI % base);
        if (t.p == t.e) err("ValueError", "invalid literal for int()");
        charge(2 * (t.e - t.p));
        for (const char *p = t.p; p < t.e; p++) {
            if (*p == '_') continue;
            int d = digit(*p) ? *p - '0' : (unsigned)((*p | 32) - 'a') < 26 ? (*p | 32) - 'a' + 10 : 99;
            if (d >= base) err3("ValueError", "invalid literal for int(): ", (str){s->s, s->s + s->h.len}, "");
            if (v > lim || (v == lim && d > last)) err("OverflowError", "an integer beyond 2^62");
            v = v * base + d;
        }
        return MKINT(neg ? -v : v);
    }
    case B_BOOL: return n && truthy(a[0]) ? TRUE_V : FALSE_V;
    case B_ENUMERATE: case B_ZIP: {
        listobj *r = new_list(0);
        if (id == B_ENUMERATE) {
            listobj *l = to_list(a[0]);
            i64 st = kw == K_START ? as_int(kwv, "start") : n > 1 ? as_int(a[1], "start") : 0;
            for (int i = 0; i < l->h.len; i++) {
                listobj *p = new_list(2);
                p->items[0] = MKINT(st + i), p->items[1] = l->items[i], p->h.tuple = 1;
                list_push(r, MKOBJ(p));
            }
            return MKOBJ(r);
        }
        listobj *ls[8];
        int m = 1 << 30;
        if (n > 8) err("TypeError", "zip of more than 8 lists");
        for (int k = 0; k < n; k++) {
            ls[k] = to_list(a[k]);
            if (ls[k]->h.len < m) m = ls[k]->h.len;
        }
        if (!n) m = 0;
        for (int i = 0; i < m; i++) {
            listobj *p = new_list(n);
            for (int k = 0; k < n; k++) p->items[k] = ls[k]->items[i];
            p->h.tuple = 1;
            list_push(r, MKOBJ(p));
        }
        return MKOBJ(r);
    }
    case B_MAP: case B_FILTER: {
        listobj *l = to_list(a[1]), *r = new_list(0);
        for (int i = 0; i < l->h.len; i++) {
            val v = a[0] == NONE_V ? l->items[i] : call_value(a[0], &l->items[i], 1, -1, NONE_V);
            if (id == B_MAP) list_push(r, v);
            else if (truthy(v)) list_push(r, l->items[i]);
        }
        return MKOBJ(r);
    }
    case B_DIVMOD: {
        i64 x = as_int(a[0], ""), y = as_int(a[1], "");
        listobj *p = new_list(2);
        p->items[0] = MKINT(floordiv(x, y)), p->items[1] = MKINT(pymod(x, y)), p->h.tuple = 1;
        return MKOBJ(p);
    }
    case B_POW: {
        i64 x = as_int(a[0], ""), y = as_int(a[1], "");
        if (n < 3) return mkint(ipow(x, y));
        i64 m = as_int(a[2], "the modulus"), r = 1 % m;
        if (y < 0 || m <= 0) err("ValueError", "pow(x, y, m) with y < 0 or m <= 0");
        x = pymod(x, m);
        while (y) {
            if (y & 1) r = mulmod(r, x, m);
            x = mulmod(x, x, m);
            y >>= 1;
            step();
        }
        return MKINT(r);
    }
    case B_CHR: {
        i64 c = as_int(a[0], "chr's argument");
        char b[4];
        int k;
        if (c < 0 || c > 0x10ffff) err("ValueError", "chr() arg not in range");
        if (c < 0x80) b[0] = (char)c, k = 1;
        else if (c < 0x800) b[0] = (char)(0xc0 | c >> 6), b[1] = (char)(0x80 | (c & 63)), k = 2;
        else if (c < 0x10000) b[0] = (char)(0xe0 | c >> 12), b[1] = (char)(0x80 | (c >> 6 & 63)), b[2] = (char)(0x80 | (c & 63)), k = 3;
        else b[0] = (char)(0xf0 | c >> 18), b[1] = (char)(0x80 | (c >> 12 & 63)), b[2] = (char)(0x80 | (c >> 6 & 63)), b[3] = (char)(0x80 | (c & 63)), k = 4;
        return str_v(b, k);
    }
    case B_ORD: {
        strobj *s = str_arg(a[0], "ord's argument");
        const u8 *p = (const u8 *)s->s;
        if (!s->h.len) err("TypeError", "ord() of an empty string");
        u32 c = p[0];
        int k = c < 0x80 ? 1 : c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : 2;
        if (k > 1) c &= 0x3f >> (k - 1);
        for (int i = 1; i < k && i < s->h.len; i++) c = c << 6 | (p[i] & 0x3f);
        return MKINT(c);
    }
    case B_BIN: case B_HEX: case B_OCT: {
        i64 x = as_int(a[0], "the number");
        char b[72];
        int k = 0;
        u64 m = x < 0 ? -(u64)x : (u64)x;
        int base = id == B_BIN ? 2 : id == B_HEX ? 16 : 8;
        do {
            int r = (int)(m % base);
            b[k++] = (char)(r < 10 ? '0' + r : 'a' + r - 10);
            m /= base;
        } while (m);
        char out[76];
        int j = 0;
        if (x < 0) out[j++] = '-';
        out[j++] = '0', out[j++] = id == B_BIN ? 'b' : id == B_HEX ? 'x' : 'o';
        while (k) out[j++] = b[--k];
        return str_v(out, j);
    }
    case B_GCD: case B_LCM: {
        i64 g = n ? as_int(a[0], "") : 0;
        for (int i = 1; i < n; i++) {
            i64 y = as_int(a[i], "");
            if (id == B_GCD) g = gcd_(g, y);
            else {
                i64 d = gcd_(g, y);
                g = d ? mul_checked(g / d, y) : 0;
                g = g < 0 ? -g : g;
            }
        }
        return mkint(id == B_GCD ? (g < 0 ? -g : g) : g);
    }
    case B_ISQRT: return MKINT(isqrt_(as_int(a[0], "isqrt's argument")));
    case B_FACTORIAL: {
        i64 x = as_int(a[0], "factorial's argument"), r = 1;
        if (x < 0) err("ValueError", "factorial() of a negative number");
        for (i64 i = 2; i <= x; i++) r = mul_checked(r, i), (void)mkint(r);
        return mkint(r);
    }
    case B_COMB: case B_PERM: {
        i64 x = as_int(a[0], ""), k = n > 1 ? as_int(a[1], "") : x, r = 1;
        if (k < 0 || x < 0) err("ValueError", "comb() or perm() of a negative number");
        if (k > x) return MKINT(0);
        if (id == B_COMB && k > x - k) k = x - k;
        for (i64 i = 0; i < k; i++) {
            step();
            if (id == B_COMB) { /* (C(x, i + 1) = C(x, i) (x - i) / (i + 1), the division first: no overflow before the result's) */
                i64 g = r, h = i + 1;
                while (h) {
                    i64 t = g % h;
                    g = h, h = t;
                }
                r = mul_checked(r / g, (x - i) / ((i + 1) / g));
            } else
                r = mul_checked(r, x - i);
            (void)mkint(r);
        }
        return mkint(r);
    }
    case B_MATH: err("TypeError", "math is a module: math.gcd(...) and the others");
    }
    err3("NameError", "no builtin ", names[id], " on the computer");
}

/* ---- the walk */
/* a comprehension's items: a list (mode 0; a dict for {k: v for ...}), or folded as it goes, made one by one and
   never kept (mode B_SUM, B_ANY, B_ALL, B_MIN, B_MAX: sum(x for x in ...) takes no memory for its items) */
static val comp(int n, int mode) {
    node *x = &nodes[n];
    listobj *l = x->kind == N_COMP && !mode ? new_list(0) : 0;
    dictobj *d = x->kind == N_DICTCOMP ? new_dict() : 0;
    int clauses = x->kind == N_COMP ? x->b : x->c, stack[8], k = 0, c = clauses, any = 0;
    /* (a comprehension of one for: each item a round, its memory given back unless kept; not min's or max's, whose
       best so far is held here) */
    int rounds_too = nodes[clauses].next < 0 && (d || mode == 0 || mode == B_SUM || mode == B_ANY || mode == B_ALL);
    iter its[8];
    i64 total = 0;
    val best = NONE_V;
    for (;;) {
        if (c >= 0) {
            if (k == 8) err("SyntaxError", "more than 8 for clauses");
            stack[k] = c;
            iter_node(&its[k], nodes[c].b);
            k++;
            c = -1;
        }
        if (!k) break;
        int cl = stack[k - 1];
        val v;
        if (!iter_next(&its[k - 1], &v)) {
            k--;
            continue;
        }
        step();
        assign(nodes[cl].a, v);
        int r = rounds_too ? round_start() : -1, ok = 1;
        for (int cond = nodes[cl].c; cond >= 0 && ok; cond = nodes[cond].next) ok = truthy(eval(cond));
        if (!ok) {
            round_end(r, 1);
            continue;
        }
        if (nodes[cl].next >= 0) {
            c = nodes[cl].next;
            continue;
        }
        if (d) {
            val key = eval(x->a);
            dict_set(d, key, eval(x->b));
            round_end(r, 1);
            continue;
        }
        val e = eval(x->a);
        switch (mode) {
        case 0: list_push(l, e); break;
        case B_SUM: total += as_int(e, "a summand"), (void)mkint(total); break;
        case B_ANY:
            if (truthy(e)) return round_end(r, 1), TRUE_V;
            break;
        case B_ALL:
            if (!truthy(e)) return round_end(r, 1), FALSE_V;
            break;
        default:
            if (!any || (mode == B_MIN ? compare(e, best) < 0 : compare(e, best) > 0)) best = e;
            any = 1;
        }
        round_end(r, 1);
    }
    if (d) return MKOBJ(d);
    switch (mode) {
    case 0: return MKOBJ(l);
    case B_SUM: return mkint(total);
    case B_ANY: return FALSE_V;
    case B_ALL: return TRUE_V;
    }
    if (!any) err("ValueError", "min() or max() of nothing");
    return best;
}

static __attribute__((noinline)) val eval_rest(int n);

static __attribute__((noinline)) val eval_bin(const node *x) {
    DEEP();
    val a = eval(x->a), b = eval(x->b);
    if (IS_INT(a) & IS_INT(b)) { /* (integers: at once) */
        i64 p = INT(a), q = INT(b), r;
        switch (x->op) {
        case OP_ADD: r = p + q; break;
        case OP_SUB: r = p - q; break;
        case OP_MUL:
            if (p < (1 << 30) && p > -(1 << 30) && q < (1 << 30) && q > -(1 << 30)) return MKINT(p * q);
            return binop(x->op, a, b);
        case OP_MOD:
            if (q > 0 && p >= 0) return MKINT(p % q);
            return binop(x->op, a, b);
        case OP_FLOORDIV:
            if (q > 0 && p >= 0) return MKINT(p / q);
            return binop(x->op, a, b);
        default: return binop(x->op, a, b);
        }
        if (r > MAXI || r < -MAXI) return binop(x->op, a, b);
        return MKINT(r);
    }
    return binop(x->op, a, b);
}

static val eval(int n) { /* (the kinds walked most, here; the others in eval_rest) */
    const node *x = &nodes[n];
    step();
    switch (x->kind) {
    case N_INT: case N_STR: return consts[x->a];
    case N_NAME:
        if (!in_function && gset[x->a]) return globals[x->a];
        return get_name(x->a);
    case N_BIN: return eval_bin(x);
    }
    return eval_rest(n);
}

static val eval_rest(int n) {
    DEEP();
    node *x = &nodes[n];
    switch (x->kind) {
    case N_INT: case N_STR: return consts[x->a];
    case N_CONST: return x->op == 0 ? NONE_V : x->op == 2 ? TRUE_V : FALSE_V;
    case N_NAME: return get_name(x->a);
    case N_LIST: case N_TUPLE: {
        listobj *l = new_list(0);
        for (int a = x->a; a >= 0; a = nodes[a].next) list_push(l, eval(a));
        l->h.tuple = x->kind == N_TUPLE;
        return MKOBJ(l);
    }
    case N_DICT: {
        dictobj *d = new_dict();
        for (int a = x->a; a >= 0; a = nodes[nodes[a].next].next) {
            val k = eval(a);
            dict_set(d, k, eval(nodes[a].next));
        }
        return MKOBJ(d);
    }
    case N_COMP: case N_DICTCOMP: return comp(n, 0);
    case N_UNARY: {
        val v = eval(x->a);
        if (!as_index(v)) err("TypeError", "a sign before something that is not a number");
        i64 i = as_int(v, "");
        return x->op == OP_NEG ? mkint(-i) : x->op == OP_INV ? mkint(~i) : MKINT(i);
    }
    case N_NOT: return truthy(eval(x->a)) ? FALSE_V : TRUE_V;
    case N_AND: {
        val v = eval(x->a);
        return truthy(v) ? eval(x->b) : v;
    }
    case N_OR: {
        val v = eval(x->a);
        return truthy(v) ? v : eval(x->b);
    }
    case N_CMP: {
        val left = eval(x->a);
        for (int c = x->b; c >= 0; c = nodes[c].next) {
            val right = eval(nodes[c].a);
            int op = nodes[c].op, ok;
            line = nodes[c].line;
            switch (op) {
            case OP_EQ: ok = equal(left, right); break;
            case OP_NE: ok = !equal(left, right); break;
            case OP_IN: ok = contains(right, left); break;
            case OP_NOTIN: ok = !contains(right, left); break;
            case OP_IS: ok = left == right || (IS_INT(left) && left == right); break;
            case OP_ISNOT: ok = left != right; break;
            default: {
                int d = compare(left, right);
                ok = op == OP_LT ? d < 0 : op == OP_LE ? d <= 0 : op == OP_GT ? d > 0 : d >= 0;
            }
            }
            if (!ok) return FALSE_V;
            left = right;
        }
        return TRUE_V;
    }
    case N_IF_EXP: return truthy(eval(x->a)) ? eval(x->b) : eval(x->c);
    case N_CALL: return do_call(n);
    case N_ATTR: { /* a method as a value (key=d.get), or math.f */
        int math = nodes[x->a].kind == N_NAME && nodes[x->a].a == B_MATH && !find_var(B_MATH);
        if (!math && x->b <= B_LAST) err3("AttributeError", "no attribute ", names[x->b], "");
        fnobj *f = alloc(sizeof(fnobj));
        f->h = (head){O_BUILTIN, 0, 0, 0};
        f->node = x->b, f->self = math ? NONE_V : eval(x->a);
        return MKOBJ(f);
    }
    case N_INDEX: {
        val o = eval(x->a);
        return index_of(o, eval(x->b));
    }
    case N_SLICE: {
        node *s = &nodes[x->b];
        val o = eval(x->a), a = s->a >= 0 ? eval(s->a) : NONE_V, b = s->b >= 0 ? eval(s->b) : NONE_V, c = s->c >= 0 ? eval(s->c) : NONE_V;
        return slice_of(o, a, b, c);
    }
    case N_LAMBDA: {
        fnobj *f = alloc(sizeof(fnobj));
        f->h = (head){O_FUNC, 0, 0, 0};
        f->node = n, f->self = NONE_V;
        return MKOBJ(f);
    }
    case N_FSTR: {
        val acc = zstr("");
        for (int p = x->a; p >= 0; p = nodes[p].next) {
            val part;
            if (nodes[p].kind == N_STR) part = consts[nodes[p].a];
            else if (nodes[p].kind == N_FSTR) part = eval(p);
            else {
                val v = eval(nodes[p].a);
                if (nodes[p].op == 'r') {
                    part = to_str(v); /* (repr: strings quoted) */
                    if (type_of(v) == O_STR) {
                        strobj *s = (strobj *)OBJ(v);
                        strobj *q = new_str(0, s->h.len + 2);
                        q->s[0] = '\'', memcpy(q->s + 1, s->s, s->h.len), q->s[s->h.len + 1] = '\'';
                        part = MKOBJ(q);
                    }
                } else
                    part = nodes[p].b >= 0 ? format(v, consts[nodes[p].b]) : to_str(v);
            }
            acc = concat(acc, part);
        }
        return acc;
    }
    }
    err("SyntaxError", "that is not an expression");
}

static __attribute__((noinline)) void exec_rest(int n);

static void run(int n) { /* a statement, and those after it in its block (the light ones here) */
    DEEP();
    for (; n >= 0 && !ctl; n = nodes[n].next) {
        const node *s = &nodes[n];
        step();
        line = s->line;
        switch (s->kind) {
        case S_BLOCK: run(s->a); continue;
        case S_IF:
            if (truthy(eval(s->a))) run(s->b);
            else if (s->c >= 0) run(s->c);
            continue;
        case S_ASSIGN:
            if (nodes[s->a].kind == N_NAME && nodes[s->a].next < 0 && !in_function) {
                val v = eval(s->b);
                globals[nodes[s->a].a] = v;
                KEEP(globals, v);
                if (!gset[nodes[s->a].a]) gset[nodes[s->a].a] = 1;
                continue;
            }
            break;
        case S_AUG: {
            const node *t = &nodes[s->a];
            if (t->kind == N_NAME && !in_function && (s->op == OP_ADD || s->op == OP_SUB) && gset[t->a] && IS_INT(globals[t->a])) {
                val v = eval(s->b);
                if (IS_INT(v)) {
                    i64 r = s->op == OP_ADD ? INT(globals[t->a]) + INT(v) : INT(globals[t->a]) - INT(v);
                    if (r > MAXI || r < -MAXI) err("OverflowError", "an integer beyond 2^62 (the computer's; calc does bigger)");
                    globals[t->a] = MKINT(r);
                    continue;
                }
                globals[t->a] = binop(s->op, globals[t->a], v);
                KEEP(globals, globals[t->a]);
                continue;
            }
            break;
        }
        }
        exec_rest(n);
    }
}

static void exec_rest(int n) {
    {
        node *s = &nodes[n];
        switch (s->kind) {
        case S_BLOCK: run(s->a); break;
        case S_EXPR: {
            val v = eval(s->a);
            if (s->op == 1 && !truthy(v)) err("AssertionError", "the assert failed");
            break;
        }
        case S_ASSIGN: {
            val v = eval(s->b);
            for (int t = s->a; t >= 0; t = nodes[t].next) assign(t, v); /* (a = b = v: a target's next the next) */
            break;
        }
        case S_AUG: {
            node *t = &nodes[s->a];
            val v = eval(s->b);
            if (t->kind == N_NAME && !in_function && gset[t->a] && IS_INT(globals[t->a]) && IS_INT(v) && (s->op == OP_ADD || s->op == OP_SUB)) {
                i64 r = s->op == OP_ADD ? INT(globals[t->a]) + INT(v) : INT(globals[t->a]) - INT(v);
                globals[t->a] = mkint(r);
            } else if (t->kind == N_NAME)
                set_var(t->a, binop(s->op, get_name(t->a), v));
            else if (t->kind == N_INDEX) {
                val o = eval(t->a), k = eval(t->b);
                val cur = index_of(o, k), nv = binop(s->op, cur, v);
                if (type_of(o) == O_LIST) ((listobj *)OBJ(o))->items[norm_index(as_int(k, ""), OBJ(o)->len)] = nv, KEEP(OBJ(o), nv);
                else if (type_of(o) == O_DICT) dict_set((dictobj *)OBJ(o), k, nv);
                else err("TypeError", "that does not support item assignment");
            } else
                err("SyntaxError", "cannot assign to that");
            break;
        }
        case S_IF:
            if (truthy(eval(s->a))) run(s->b);
            else if (s->c >= 0) run(s->c);
            break;
        case S_WHILE:
            for (;;) {
                int r = round_start();
                if (!truthy(eval(s->a))) {
                    round_end(r, 1);
                    break;
                }
                run(s->b);
                round_end(r, ctl == C_NONE || ctl == C_CONTINUE);
                if (ctl == C_BREAK) {
                    ctl = C_NONE;
                    break;
                }
                if (ctl == C_CONTINUE) ctl = C_NONE;
                if (ctl == C_RETURN) break;
            }
            break;
        case S_FOR: {
            iter it;
            val v;
            iter_node(&it, s->b);
            if (it.kind == O_RANGE && nodes[s->a].kind == N_NAME) { /* (for i in range(...): no detours) */
                int name = nodes[s->a].a;
                for (i64 i = it.cur; it.step > 0 ? i < it.stop : i > it.stop; i += it.step) {
                    if (!in_function) {
                        globals[name] = MKINT(i);
                        if (!gset[name]) gset[name] = 1;
                    } else
                        set_var(name, MKINT(i));
                    int r = round_start();
                    run(nodes[s->c].a);
                    round_end(r, ctl == C_NONE || ctl == C_CONTINUE);
                    if (ctl == C_BREAK) {
                        ctl = C_NONE;
                        break;
                    }
                    if (ctl == C_CONTINUE) ctl = C_NONE;
                    if (ctl == C_RETURN) break;
                }
                break;
            }
            while (iter_next(&it, &v)) {
                assign(s->a, v);
                int r = round_start();
                run(s->c);
                round_end(r, ctl == C_NONE || ctl == C_CONTINUE);
                if (ctl == C_BREAK) {
                    ctl = C_NONE;
                    break;
                }
                if (ctl == C_CONTINUE) ctl = C_NONE;
                if (ctl == C_RETURN) break;
            }
            break;
        }
        case S_DEF: {
            fnobj *f = alloc(sizeof(fnobj));
            f->h = (head){O_FUNC, 0, 0, 0};
            f->node = n, f->self = NONE_V;
            set_var(s->a, MKOBJ(f));
            break;
        }
        case S_RETURN:
            ret = s->a >= 0 ? eval(s->a) : NONE_V;
            ctl = C_RETURN;
            break;
        case S_BREAK: ctl = C_BREAK; break;
        case S_CONTINUE: ctl = C_CONTINUE; break;
        case S_PASS: break;
        case S_GLOBAL:
            for (int g = s->a; g >= 0; g = nodes[g].next)
                if (in_function) {
                    if (n_frame == FRAME_MAX) err("RecursionError", "too many variables in the calls going on");
                    frames[n_frame++] = (local){nodes[g].a, 1, NONE_V};
                }
            break;
        case S_DEL: {
            node *t = &nodes[s->a];
            if (t->kind == N_INDEX) {
                val o = eval(t->a), k = eval(t->b);
                if (type_of(o) == O_LIST) {
                    listobj *l = (listobj *)OBJ(o);
                    i64 i = norm_index(as_int(k, "an index"), l->h.len);
                    memmove(l->items + i, l->items + i + 1, sizeof(val) * (l->h.len - i - 1));
                    l->h.len--;
                } else if (type_of(o) == O_DICT)
                    builtin(M_POP, &k, 1, -1, NONE_V, o);
                else
                    err("TypeError", "del of that is not supported");
            } else if (t->kind == N_NAME) {
                if (!find_var(t->a)) err3("NameError", "name '", names[t->a], "' is not defined");
                gset[t->a] = 0;
            } else
                err("SyntaxError", "del of that is not supported");
            break;
        }
        default: eval(n);
        }
    }
}

/* ---- the tool */

void run_python(void) {
    str code = param("code");
    if (!have(code)) code = param("program");
    if (!have(code)) fail("missing argument: code, the program");
    /* (a fenced block's fences) */
    str c = trim(code);
    if (c.e - c.p > 6 && !memcmp(c.p, "```", 3)) {
        const char *nl = find_byte(c.p, c.e, '\n');
        if (nl) c.p = nl + 1;
        if (c.e - c.p >= 3 && !memcmp(c.e - 3, "```", 3)) c.e -= 3;
    }
    src = c.p, src_end = c.e;
    steps = 0, work = 0; /* (the parser's strings are charged too: from 0, not where the last run stopped) */
    {
        char here;
        /* (the stack the calls may take, below this frame; 1 KB of it for what runs below a check, and an error's
           report. The host's frames are larger than the computer's, the parser's about 1.6 times, the walk's about
           1.45: as much more room there, for the depths the computer reaches) */
#ifdef HOST
        stack_floor = heap_reserve(&here, 9216 * 29 / 20) + 1024 * 29 / 20;
        parse_floor = &here - 9216 * 16 / 10 + 1024 * 16 / 10;
#else
        parse_floor = stack_floor = heap_reserve(&here, 9216) + 1024;
#endif
    }
    oom_why = "out of memory: the program is too long for the computer (about 700 bytes at most): make it shorter, "
              "or run it in parts, a call each";
    names = alloc_low(sizeof(str) * NAMES);
    slots = alloc_low(sizeof(i16) * 256);
    memset(slots, 0, sizeof(i16) * 256);
    frames = alloc_low(sizeof(local) * FRAME_MAX);
    str_buf = alloc_low(STR_BUF);
    max_toks = (int)(src_end - src) / 2 + 64;
    toks = alloc_top(sizeof(token) * max_toks); /* (the parse's only: given back for the run) */
    n_toks = ti = 0;
    n_names = N_FIXED;
    for (int i = 0; i < N_FIXED; i++) names[i] = (str){FIXED[i], FIXED[i] + strlen(FIXED[i])};
    if (!op_first['+']) chain(FIXED, fx_first, fx_next), chain(KEYWORDS, kw_first, kw_next), chain(OP_TEXT, op_first, op_next);
    lex(src, src_end, 1, 1);
    max_nodes = n_toks + 64;
    nodes = alloc(sizeof(node) * max_nodes);
    max_consts = n_toks / 2 + 32;
    consts = alloc_low(sizeof(val) * max_consts);
    n_nodes = n_consts = 0;
    no_in = 0; /* (a run that ended inside a for's target left it set) */
    int prog = statements(0);
    free_top();
    toks = 0;
    oom_why = "out of memory: the program's values take more than the computer's memory (about 10 KB): fewer or "
              "smaller ones, or the work in parts, a call each";
    globals = alloc(sizeof(val) * n_names);
    gset = alloc(n_names);
    memset(gset, 0, n_names);
    n_frame = frame_base = in_function = rounds = 0; /* (rounds: a run that failed in a loop left some) */
    steps = 0, work = 0;
    ctl = C_NONE;
    capture_at = 0;
    run(nodes[prog].a);
#ifdef HOST
    if (getenv("PYTHON_STEPS")) fprintf(stderr, "steps %lld\n", (long long)steps);
#endif
}
