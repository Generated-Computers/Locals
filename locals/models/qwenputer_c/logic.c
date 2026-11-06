/* Propositional logic: truth_table and logic_equivalent. An expression of variables (names: p, q, rain, x1),
   true / false (or 1 / 0), parentheses and, from the tightest: not (! ~ ¬), and (&& & ∧), xor (^ ⊕), or (|| | ∨),
   implies (-> => → , right to left), iff (<-> <=> ↔ ==). Up to 12 variables; a row's values are bits of a 64-bit
   word, so 64 rows are worked out at once. */
#include "tools.h"

#define VARS 12
enum { VAR, CONST, NOT, AND, XOR, OR, IMP, IFF, LP, RP, END };

typedef struct {
    int op, arg; /* VAR: the variable; CONST: 0 or 1 */
} tok;

static str vars[VARS];
static int n_vars;
static const char *lp, *le; /* (the expression being read) */
static const char *err;

static int name_char(int c) { return (unsigned)((c | 32) - 'a') < 26 || digit(c) || c == '_' || c == '\''; }

static int match(const char *z) {
    size_t n = strlen(z);
    if ((size_t)(le - lp) < n || memcmp(lp, z, n)) return 0;
    if (name_char((u8)z[0]) && lp + n < le && name_char((u8)lp[n])) return 0; /* ("android" is not "and") */
    lp += n;
    return 1;
}

static int word_is(str w, const char *z) {
    str l = w;
    const char *q = z;
    for (; l.p < l.e && *q; l.p++, q++)
        if (lower((u8)*l.p) != *q) return 0;
    return l.p == l.e && !*q;
}

static tok next(void) {
    while (lp < le && space(*lp)) lp++;
    if (lp >= le) return (tok){END, 0};
    static const struct {
        const char *z;
        int op;
    } SYM[] = {{"<->", IFF}, {"<=>", IFF}, {"\xe2\x86\x94", IFF}, {"\xe2\x89\xa1", IFF}, {"==", IFF},
               {"->", IMP},  {"=>", IMP},  {"\xe2\x86\x92", IMP}, {"&&", AND}, {"||", OR},
               {"!=", XOR},  {"&", AND},   {"\xe2\x88\xa7", AND}, {"*", AND},  {"|", OR}, {"\xe2\x88\xa8", OR},
               {"+", OR},    {"^", XOR},   {"\xe2\x8a\x95", XOR}, {"!", NOT},  {"~", NOT}, {"\xc2\xac", NOT},
               {"(", LP},    {")", RP}};
    for (size_t i = 0; i < sizeof SYM / sizeof SYM[0]; i++) /* (the ones that start with this byte) */
        if (SYM[i].z[0] == *lp && match(SYM[i].z)) return (tok){SYM[i].op, 0};
    if (!name_char((u8)*lp)) {
        err = "a character that is not part of an expression";
        return (tok){END, 0};
    }
    const char *s = lp;
    while (lp < le && name_char((u8)*lp)) lp++;
    str w = {s, lp};
    static const struct {
        const char *z;
        int op, arg;
    } WORDS[] = {{"not", NOT, 0}, {"and", AND, 0}, {"or", OR, 0}, {"xor", XOR, 0}, {"implies", IMP, 0},
                 {"iff", IFF, 0}, {"true", CONST, 1}, {"false", CONST, 0}, {"1", CONST, 1}, {"0", CONST, 0}};
    for (size_t i = 0; i < sizeof WORDS / sizeof WORDS[0]; i++)
        if (WORDS[i].z[0] == lower((u8)*w.p) && word_is(w, WORDS[i].z)) return (tok){WORDS[i].op, WORDS[i].arg};
    for (int i = 0; i < n_vars; i++)
        if (str_ieq(vars[i], w) && memcmp(vars[i].p, w.p, w.e - w.p) == 0) return (tok){VAR, i};
    if (n_vars == VARS) {
        err = "more than 12 variables";
        return (tok){END, 0};
    }
    vars[n_vars] = w;
    return (tok){VAR, n_vars++};
}

/* the expression in postfix order (shunting-yard) */
static tok *code;
static int n_code;

static int prec(int op) { return op == NOT ? 6 : op == AND ? 5 : op == XOR ? 4 : op == OR ? 3 : op == IMP ? 2 : 1; }

static int compile(str e) {
    lp = e.p, le = e.e, err = 0;
    tok ops[64];
    int n_ops = 0, want_operand = 1, depth = 0;
    code = alloc(sizeof(tok) * (size_t)(e.e - e.p + 2));
    n_code = 0;
    for (;;) {
        tok t = next();
        if (err) return 0;
        if (want_operand) {
            if (t.op == VAR || t.op == CONST) code[n_code++] = t, want_operand = 0;
            else if (t.op == NOT || t.op == LP) {
                if (n_ops == 64) return err = "nested too deep", 0;
                ops[n_ops++] = t;
                depth += t.op == LP;
            } else
                return err = t.op == END ? "the expression ends where a value should come" : "an operator where a value should come", 0;
            continue;
        }
        if (t.op == RP || t.op == END) {
            while (n_ops && ops[n_ops - 1].op != LP) code[n_code++] = ops[--n_ops];
            if (t.op == END) {
                if (n_ops) return err = "a ( without its )", 0;
                return 1;
            }
            if (!n_ops) return err = "a ) without its (", 0;
            n_ops--, depth--;
            continue;
        }
        if (t.op < NOT || t.op > IFF || t.op == NOT) return err = "two values with no operator between them", 0;
        while (n_ops && ops[n_ops - 1].op != LP &&
               (prec(ops[n_ops - 1].op) > prec(t.op) || (prec(ops[n_ops - 1].op) == prec(t.op) && t.op != IMP)))
            code[n_code++] = ops[--n_ops];
        if (n_ops == 64) return err = "nested too deep", 0;
        ops[n_ops++] = t;
        want_operand = 1;
    }
}

/* the expression's values in rows 64 c .. 64 c + 63 (row r: variable i is bit n_vars - 1 - i of r) */
static u64 eval(int c) {
    u64 st[66]; /* (at most 64 operators pending: 65 values) */
    int sp = 0;
    for (int i = 0; i < n_code; i++) {
        tok t = code[i];
        if (t.op == VAR) {
            int bit = n_vars - 1 - t.arg;
            u64 v;
            if (bit >= 6) v = (u64)c >> (bit - 6) & 1 ? ~0ULL : 0;
            else {
                static const u64 PAT[6] = {0xAAAAAAAAAAAAAAAAULL, 0xCCCCCCCCCCCCCCCCULL, 0xF0F0F0F0F0F0F0F0ULL,
                                           0xFF00FF00FF00FF00ULL, 0xFFFF0000FFFF0000ULL, 0xFFFFFFFF00000000ULL};
                v = PAT[bit];
            }
            st[sp++] = v;
        } else if (t.op == CONST)
            st[sp++] = t.arg ? ~0ULL : 0;
        else if (t.op == NOT)
            st[sp - 1] = ~st[sp - 1];
        else {
            u64 b = st[--sp], a = st[sp - 1];
            st[sp - 1] = t.op == AND ? a & b : t.op == OR ? a | b : t.op == XOR ? a ^ b : t.op == IMP ? ~a | b : ~(a ^ b);
        }
    }
    return st[0];
}

static int rows(void) { return 1 << n_vars; }

static u64 mask(void) { return n_vars >= 6 ? ~0ULL : (1ULL << rows()) - 1; }

static void read_expr(const char *name) {
    str e = trim(param(name));
    if (!have(e) || e.p == e.e) {
        say("error (missing argument: ");
        say(name);
        put(')');
        done();
    }
    if (!compile(e)) fail(err);
}

static void say_row(int r) {
    for (int i = 0; i < n_vars; i++) {
        if (i) put(' ');
        int w = (int)(vars[i].e - vars[i].p), bit = r >> (n_vars - 1 - i) & 1;
        put(bit ? 'T' : 'F');
        for (int k = 1; k < w; k++) put(' '); /* (under its name) */
    }
}

void truth_table(void) {
    n_vars = 0;
    read_expr("expression");
    str e = trim(param("expression"));
    int n = rows(), chunks = n > 64 ? n / 64 : 1, t = 0;
    for (int c = 0; c < chunks; c++) {
        u64 v = eval(c) & mask();
        while (v) t++, v &= v - 1;
    }
    if (n_vars <= 6) {
        for (int i = 0; i < n_vars; i++) {
            if (i) put(' ');
            say_str(vars[i]);
        }
        if (n_vars) say(" | ");
        say_str(e);
        put('\n');
        u64 v = eval(0);
        for (int r = 0; r < n; r++) {
            say_row(r);
            if (n_vars) say(" | ");
            put(v >> r & 1 ? 'T' : 'F');
            put('\n');
        }
    }
    say("true in ");
    say_int(t);
    say(" of ");
    say_int(n);
    say(n == 1 ? " row: " : " rows: ");
    say(t == n ? "a tautology (always true)" : t == 0 ? "a contradiction (never true)" : "satisfiable, not a tautology");
    if (n_vars > 6 && t && t < n) { /* (too many rows to show: the first that make it true) */
        say("\ntrue where (");
        for (int i = 0; i < n_vars; i++) {
            if (i) put(' ');
            say_str(vars[i]);
        }
        say("):");
        int shown = 0;
        for (int c = 0; c < n / 64 && shown < 16; c++) {
            u64 v = eval(c);
            for (int b = 0; b < 64 && shown < 16; b++)
                if (v >> b & 1) {
                    put('\n');
                    say_row(c * 64 + b);
                    shown++;
                }
        }
        if (t > 16) say("\n...");
    }
}

void logic_equivalent(void) {
    n_vars = 0;
    read_expr("first");
    tok *first = code;
    int n_first = n_code;
    read_expr("second"); /* (its own variables after the first's) */
    tok *second = code;
    int n_second = n_code, n = rows(), chunks = n > 64 ? n / 64 : 1;
    for (int c = 0; c < chunks; c++) {
        code = first, n_code = n_first;
        u64 a = eval(c);
        code = second, n_code = n_second;
        u64 b = eval(c), d = (a ^ b) & mask();
        if (!d) continue;
        int bit = 0;
        while (!(d >> bit & 1)) bit++;
        int r = c * 64 + bit;
        say("not equivalent: where ");
        for (int i = 0; i < n_vars; i++) {
            if (i) say(", ");
            say_str(vars[i]);
            say(r >> (n_vars - 1 - i) & 1 ? " = T" : " = F");
        }
        say(a >> bit & 1 ? ", the first is true and the second false" : ", the first is false and the second true");
        return;
    }
    say("equivalent: the same value in all ");
    say_int(n);
    say(n == 1 ? " row" : " rows");
}
