/* Regular expressions as Python's re reads them, a subset: literals and escapes, ., [classes] (ranges, ^, \d \w \s
   in them), \d \D \w \W \s \S \b \B, ^ and $ (the text's start and end), groups ( ) and (?: ), |, the quantifiers
   * + ? {n} {n,} {n,m} and their lazy forms (*? ...), and (?i) in front for letters in either case (ASCII).
   The pattern is compiled to a small program for a backtracking machine (Pike's instructions), which tries the
   alternatives in Python's order, so the matches are Python's. regex_find and regex_replace. */
#include "tools.h"

enum { CHAR, ANY, CLASS, SPLIT, JMP, SAVE, MATCH, BOL, EOL, WORDB, NWORDB, CHK, STAR };

typedef struct {
    u8 op;
    u16 x; /* SPLIT: the preferred branch; JMP: where; SAVE, CHK: the slot; CLASS: the class; STAR: its atom's kind */
    u32 c; /* CHAR: the code point; SPLIT: the other branch; CHK: where an empty round goes; STAR: its atom's c or x */
} inst;

/* a class: its ranges (code points) and the \d \w \s (and \D \W \S) in it; negated with ^ */
enum { C_D = 1, C_W = 2, C_S = 4, C_ND = 8, C_NW = 16, C_NS = 32 };
typedef struct {
    u16 first, n; /* ranges[first .. first + n) */
    u8 neg, sets;
} cls;

#define PROG 400
#define RANGES 128
#define CLASSES 32
#define GROUPS 10
#define LOOPS 8 /* (repeats of what can match nothing, e.g. (a*)*: each keeps where its round began) */
#define NEST 20 /* (groups in groups: the parser's stack) */

/* the parsed pattern: a tree of nodes; a sequence (N_CAT) and alternatives (N_ALT) are lists of cells, each its
   item in a and the next cell in b (-1 at the end), so the code is written a list at a time, not deeper and deeper */
enum { N_CHAR, N_ANY, N_CLASS, N_CAT, N_ALT, N_REP, N_GROUP, N_BOL, N_EOL, N_WB, N_NWB, N_EMPTY };
typedef struct {
    u8 type, greedy;
    i16 min, max; /* N_REP: max -1 for no limit */
    i16 b;        /* the next cell, or N_GROUP's number (-1: none) */
    int a;        /* the item or child, or N_CHAR's code point, N_CLASS's class */
} node;

static node *nodes;
static int n_nodes, max_nodes, n_groups, icase, nest, n_loops;
static u32 (*ranges)[2];
static int n_ranges;
static cls *classes;
static int n_classes;
static inst *prog;
static int n_prog;
static u8 (*cbits)[16]; /* (each class's ASCII characters, a bit each: the machine's test at once) */
static int lead_rep;    /* (the pattern's first item a repeat of one character's atom, C+ or C*: that atom, else -1) */
static const char *rp, *re_; /* (the pattern being read) */

static int mk(int type, int a, int b) {
    if (n_nodes == max_nodes) fail("the pattern: too many parts for the computer");
    nodes[n_nodes] = (node){(u8)type, 1, 0, 0, (i16)b, a};
    return n_nodes++;
}

/* one UTF-8 character at p (in [p, e)): its code point, and its length in *n */
static u32 utf8(const char *p, const char *e, int *n) {
    u8 c = *p;
    int len = c < 0x80 ? 1 : c >= 0xf0 ? 4 : c >= 0xe0 ? 3 : c >= 0xc0 ? 2 : 1;
    if (p + len > e) len = 1;
    u32 u = len == 1 ? c : c & (0x3f >> (len - 1));
    for (int i = 1; i < len; i++) u = u << 6 | ((u8)p[i] & 0x3f);
    *n = len;
    return u;
}

static int is_word(u32 c) {
    if (c < 0x80) return (unsigned)((c | 32) - 'a') < 26 || digit((int)c) || c == '_';
    return c >= 0xaa && c != 0xd7 && c != 0xf7 && !(c >= 0x2000 && c < 0x2c00) && !(c >= 0x3000 && c < 0x3040) &&
           !(c >= 0xfe30 && c < 0xfe50) && !(c >= 0xff00 && c < 0xff21);
}

static int is_space(u32 c) {
    return c == ' ' || (c >= 9 && c <= 13) || (c >= 0x1c && c <= 0x1f) || c == 0x85 || c == 0xa0 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

static u32 fold(u32 c) { return icase && c - 'A' < 26 ? c + 32 : c; }

static int class_has(const cls *k, u32 c) {
    int in = 0;
    u32 f = fold(c), g = icase && c - 'a' < 26 ? c - 32 : c;
    for (int i = 0; i < k->n && !in; i++) {
        u32 lo = ranges[k->first + i][0], hi = ranges[k->first + i][1];
        in = (c >= lo && c <= hi) || (icase && ((f >= lo && f <= hi) || (g >= lo && g <= hi)));
    }
    if (!in && k->sets) { /* (each set's test only if the class has it) */
        int x = k->sets;
        in = (x & (C_D | C_ND) && (c - '0' < 10) == !!(x & C_D)) || (x & (C_D | C_ND)) == (C_D | C_ND) ||
             (x & (C_W | C_NW) && is_word(c) == !!(x & C_W)) || (x & (C_W | C_NW)) == (C_W | C_NW) ||
             (x & (C_S | C_NS) && is_space(c) == !!(x & C_S)) || (x & (C_S | C_NS)) == (C_S | C_NS);
    }
    return in != k->neg;
}

static int new_class(int neg) {
    if (n_classes == CLASSES) fail("a pattern of more than 32 character classes");
    classes[n_classes] = (cls){(u16)n_ranges, 0, (u8)neg, 0};
    return n_classes++;
}

static void add_range(int k, u32 lo, u32 hi) {
    if (n_ranges == RANGES) fail("a pattern with more than 128 ranges in its classes");
    ranges[n_ranges][0] = lo, ranges[n_ranges][1] = hi;
    n_ranges++;
    classes[k].n++;
}

static __attribute__((noreturn)) void bad_pattern(const char *why) {
    say("error (the pattern: ");
    say(why);
    put(')');
    done();
}

/* an escape after \ : a character (its code point), or -1 with a set in *set (C_D ...) */
static long escape(int *set) {
    if (rp >= re_) bad_pattern("a \\ at its end");
    int n;
    u32 c = utf8(rp, re_, &n);
    rp += n;
    *set = 0;
    switch (c) {
    case 'd': return *set = C_D, -1;
    case 'w': return *set = C_W, -1;
    case 's': return *set = C_S, -1;
    case 'D': return *set = C_ND, -1;
    case 'W': return *set = C_NW, -1;
    case 'S': return *set = C_NS, -1;
    case 'n': return '\n';
    case 't': return '\t';
    case 'r': return '\r';
    case 'f': return '\f';
    case 'v': return '\v';
    case '0': return 0;
    }
    if (c < 0x80 && ((unsigned)((c | 32) - 'a') < 26 || digit((int)c)) && c != 'b' && c != 'B')
        bad_pattern("an escape it does not know (\\1 back references and \\p{...} are not supported)");
    return c;
}

static int parse_alt(void);

static int parse_class(void) { /* after [ */
    int neg = rp < re_ && *rp == '^';
    if (neg) rp++;
    int k = new_class(neg), first = 1;
    while (rp < re_ && (*rp != ']' || first)) {
        first = 0;
        int n, set = 0;
        long lo;
        if (*rp == '\\') {
            rp++;
            lo = escape(&set);
            if (lo == 'b') lo = '\b';
        } else {
            lo = utf8(rp, re_, &n);
            rp += n;
        }
        if (lo < 0) {
            classes[k].sets |= set;
            continue;
        }
        long hi = lo;
        if (rp + 1 < re_ && *rp == '-' && rp[1] != ']') {
            rp++;
            if (*rp == '\\') {
                rp++;
                hi = escape(&set);
                if (hi < 0) bad_pattern("a range that ends in a \\d, \\w or \\s");
            } else {
                hi = utf8(rp, re_, &n);
                rp += n;
            }
            if (hi < lo) bad_pattern("a range whose end is before its start");
        }
        add_range(k, (u32)lo, (u32)hi);
    }
    if (rp >= re_) bad_pattern("a [ without its ]");
    rp++;
    return mk(N_CLASS, k, 0);
}

static int parse_atom(void) {
    int n;
    u32 c = utf8(rp, re_, &n);
    rp += n;
    switch (c) {
    case '(': {
        int group = -1;
        if (++nest > NEST) bad_pattern("groups nested more than 20 deep");
        if (rp + 1 < re_ && rp[0] == '?' && rp[1] == ':') rp += 2;
        else if (rp < re_ && *rp == '?') bad_pattern("a (? group it does not know (lookarounds and names are not supported)");
        else {
            if (n_groups == GROUPS - 1) bad_pattern("more than 9 groups");
            group = ++n_groups;
        }
        int a = parse_alt();
        if (rp >= re_ || *rp != ')') bad_pattern("a ( without its )");
        rp++, nest--;
        return mk(N_GROUP, a, group);
    }
    case '[': return parse_class();
    case '.': return mk(N_ANY, 0, 0);
    case '^': return mk(N_BOL, 0, 0);
    case '$': return mk(N_EOL, 0, 0);
    case ')': bad_pattern("a ) without its (");
    case '*': case '+': case '?': bad_pattern("a quantifier with nothing before it");
    case '\\': {
        if (rp < re_ && (*rp == 'b' || *rp == 'B')) return mk(*rp++ == 'b' ? N_WB : N_NWB, 0, 0);
        int set;
        long e = escape(&set);
        if (e >= 0) return mk(N_CHAR, (int)e, 0);
        int k = new_class(0);
        classes[k].sets = set;
        return mk(N_CLASS, k, 0);
    }
    }
    return mk(N_CHAR, (int)c, 0);
}

static int number(void) {
    int v = 0, any = 0;
    for (; rp < re_ && digit(*rp); rp++, any = 1)
        if (v <= 1000) v = v * 10 + (*rp - '0'); /* (more digits than that: too many, however many) */
    if (v > 1000) bad_pattern("a repeat count of more than 1,000");
    return any ? v : -1;
}

static int parse_repeat(void) {
    int a = parse_atom();
    for (;;) {
        if (rp >= re_) return a;
        int min, max;
        const char *save = rp;
        if (*rp == '*') min = 0, max = -1, rp++;
        else if (*rp == '+') min = 1, max = -1, rp++;
        else if (*rp == '?') min = 0, max = 1, rp++;
        else if (*rp == '{') {
            rp++;
            min = number();
            max = min;
            if (rp < re_ && *rp == ',') rp++, max = number(), min = min < 0 ? 0 : min; /* ({,n}: from 0, as Python's) */
            if (min < 0 || rp >= re_ || *rp != '}') { /* (not a repeat: a literal {) */
                rp = save;
                return a;
            }
            rp++;
            if (max >= 0 && max < min) bad_pattern("a {n,m} with m less than n");
        } else
            return a;
        int t = nodes[a].type;
        if (t == N_BOL || t == N_EOL || t == N_WB || t == N_NWB) bad_pattern("a quantifier after an anchor");
        if (t == N_REP) bad_pattern("two quantifiers in a row (write (?:a*)* for a repeat of a repeat)");
        int r = mk(N_REP, a, 0);
        nodes[r].min = (i16)min, nodes[r].max = (i16)max;
        if (rp < re_ && *rp == '?') nodes[r].greedy = 0, rp++;
        a = r;
    }
}

static int parse_cat(void) { /* one item alone, or a list of N_CAT cells */
    int first = -1, head = -1, last = -1;
    while (rp < re_ && *rp != '|' && *rp != ')') {
        int b = parse_repeat();
        if (first < 0) {
            first = b;
            continue;
        }
        if (head < 0) head = last = mk(N_CAT, first, -1);
        int c = mk(N_CAT, b, -1);
        nodes[last].b = (i16)c, last = c;
    }
    return head >= 0 ? head : first >= 0 ? first : mk(N_EMPTY, 0, 0);
}

static int parse_alt(void) { /* one alternative alone, or a list of N_ALT cells */
    int first = parse_cat(), head = -1, last = -1;
    while (rp < re_ && *rp == '|') {
        rp++;
        if (head < 0) head = last = mk(N_ALT, first, -1);
        int c = mk(N_ALT, parse_cat(), -1);
        nodes[last].b = (i16)c, last = c;
    }
    return head >= 0 ? head : first;
}

static int nullable(int i) { /* node i can match nothing */
    const node *d = &nodes[i];
    switch (d->type) {
    case N_CHAR: case N_ANY: case N_CLASS: return 0;
    case N_CAT:
        for (int c = i; c >= 0; c = nodes[c].b)
            if (!nullable(nodes[c].a)) return 0;
        return 1;
    case N_ALT:
        for (int c = i; c >= 0; c = nodes[c].b)
            if (nullable(nodes[c].a)) return 1;
        return 0;
    case N_REP: return d->min == 0 || nullable(d->a);
    case N_GROUP: return nullable(d->a);
    }
    return 1; /* (anchors, nothing) */
}

/* ---- code */
static int emit(int op, int x, u32 c) {
    if (n_prog == PROG) bad_pattern("too long once its repeats are written out (at most 400 steps)");
    prog[n_prog] = (inst){(u8)op, (u16)x, c};
    return n_prog++;
}

static void gen(int i) {
    node *d = &nodes[i];
    switch (d->type) {
    case N_CHAR: emit(CHAR, 0, fold((u32)d->a)); break;
    case N_ANY: emit(ANY, 0, 0); break;
    case N_CLASS: emit(CLASS, d->a, 0); break;
    case N_BOL: emit(BOL, 0, 0); break;
    case N_EOL: emit(EOL, 0, 0); break;
    case N_WB: emit(WORDB, 0, 0); break;
    case N_NWB: emit(NWORDB, 0, 0); break;
    case N_EMPTY: break;
    case N_CAT:
        for (int c = i; c >= 0; c = nodes[c].b) gen(nodes[c].a);
        break;
    case N_ALT: { /* split a, next; a; jmp end; next: split b, ... ; the last alternative */
        int c = i, jmps = -1; /* (the jumps to the end, chained through their x until it is known) */
        for (; nodes[c].b >= 0; c = nodes[c].b) {
            int s = emit(SPLIT, 0, 0);
            prog[s].x = (u16)n_prog;
            gen(nodes[c].a);
            int j = emit(JMP, 0, 0);
            prog[j].x = (u16)(jmps < 0 ? 0xffff : jmps), jmps = j;
            prog[s].c = (u32)n_prog;
        }
        gen(nodes[c].a);
        while (jmps >= 0) {
            int next = prog[jmps].x == 0xffff ? -1 : prog[jmps].x;
            prog[jmps].x = (u16)n_prog, jmps = next;
        }
        break;
    }
    case N_GROUP:
        if (d->b >= 0) emit(SAVE, 2 * d->b, 0);
        gen(d->a);
        if (d->b >= 0) emit(SAVE, 2 * d->b + 1, 0);
        break;
    case N_REP: {
        int a = d->a, greedy = d->greedy, t = nodes[a].type;
        for (int k = 0; k < d->min; k++) gen(a);
        if (d->max < 0 && greedy && (t == N_CHAR || t == N_CLASS || t == N_ANY)) {
            /* (C* greedy: the longest run of C's at once, then shorter ones as the rest fails: STAR) */
            emit(STAR, t == N_CHAR ? CHAR : t == N_CLASS ? CLASS : ANY, t == N_CHAR ? fold((u32)nodes[a].a) : (u32)nodes[a].a);
        } else if (d->max < 0) { /* L: split body, out; body; jmp L */
            int s = emit(SPLIT, 0, 0), loop = -1, chk = -1;
            if (nullable(a)) { /* (a round that matched nothing ends the repeat, as in Python: (a*)* stops) */
                if (n_loops == LOOPS) bad_pattern("more than 8 repeats of parts that can match nothing");
                loop = 2 * GROUPS + n_loops++;
                emit(SAVE, loop, 0);
            }
            gen(a);
            if (loop >= 0) chk = emit(CHK, loop, 0);
            emit(JMP, s, 0);
            prog[s].x = (u16)(greedy ? s + 1 : n_prog);
            prog[s].c = (u32)(greedy ? n_prog : s + 1);
            if (chk >= 0) prog[chk].c = (u32)n_prog;
        } else {
            int opt = d->max - d->min, first = n_prog, loop = -1;
            if (opt > 0 && nullable(a)) { /* (here too a round that matched nothing ends the repeat: Python's) */
                if (n_loops == LOOPS) bad_pattern("more than 8 repeats of parts that can match nothing");
                loop = 2 * GROUPS + n_loops++;
            }
            for (int k = 0; k < opt; k++) { /* (each split's x for now: where its body ends) */
                int s = emit(SPLIT, 0, 0);
                if (loop >= 0) emit(SAVE, loop, 0);
                gen(a);
                if (loop >= 0) emit(CHK, loop, 0xffffffffu); /* (to the repeat's end, known below) */
                prog[s].x = (u16)n_prog;
            }
            for (int s = first; s < n_prog && loop >= 0; s++)
                if (prog[s].op == CHK && prog[s].x == loop && prog[s].c == 0xffffffffu) prog[s].c = (u32)n_prog;
            for (int s = first; s < n_prog;) {
                int body_end = prog[s].x;
                prog[s].x = (u16)(greedy ? s + 1 : n_prog);
                prog[s].c = (u32)(greedy ? n_prog : s + 1);
                s = body_end;
            }
        }
        break;
    }
    }
}

static void compile(str pattern) {
    max_nodes = 2 * (int)(pattern.e - pattern.p) + 4;
    nodes = alloc(sizeof(node) * max_nodes);
    ranges = alloc(sizeof(u32) * 2 * RANGES);
    classes = alloc(sizeof(cls) * CLASSES);
    prog = alloc_low(sizeof(inst) * PROG); /* (the program and the machine's stack in the memory kept for tables) */
    n_nodes = n_groups = n_ranges = n_classes = n_prog = icase = nest = n_loops = 0;
    rp = pattern.p, re_ = pattern.e;
    if (re_ - rp >= 4 && !memcmp(rp, "(?i)", 4)) icase = 1, rp += 4;
    int root = parse_alt();
    if (rp < re_) bad_pattern("a ) without its (");
    cbits = alloc(16 * (size_t)(n_classes + 1));
    for (int k = 0; k < n_classes; k++)
        for (u32 c = 0; c < 0x80; c++) cbits[k][c >> 3] = (u8)(cbits[k][c >> 3] & ~(1 << (c & 7))) | (u8)(class_has(&classes[k], c) << (c & 7));
    lead_rep = -1;
    for (int r = root;;) { /* (its first item, through groups and sequences) */
        const node *d = &nodes[r];
        if (d->type == N_GROUP) r = d->a;
        else if (d->type == N_CAT) r = d->a;
        else {
            if (d->type == N_REP && d->max < 0 && d->min <= 1 &&
                (nodes[d->a].type == N_CHAR || nodes[d->a].type == N_CLASS || nodes[d->a].type == N_ANY))
                lead_rep = d->a;
            break;
        }
    }
    emit(SAVE, 0, 0);
    gen(root);
    emit(SAVE, 1, 0);
    emit(MATCH, 0, 0);
}

/* ---- the machine: positions as offsets from the text's start (0xffff: none) */
static const char *ts, *te_; /* the text */
#define NONE 0xffff
static u16 caps[2 * GROUPS + LOOPS];
static long steps;
#define STEPS 1500

typedef struct {
    i16 pc; /* -1 - slot: a capture to restore */
    u16 sp;
} entry;

static entry *stack;
#define STACK 400
#define STAR_MARK 0x7fff  /* (an entry below a STAR's: where its run began) */
#define STAR_NEXT 0x4000  /* (with a STAR's place: its run a character shorter, next) */

#define ASCII_WORD(c) ((unsigned)(((c) | 32) - 'a') < 26 || digit(c) || (c) == '_')

static int is_word_at(const char *p) { /* the character at p is a word character */
    int n;
    return p >= ts && p < te_ && is_word(utf8(p, te_, &n));
}

static int word_before(const char *p) {
    if (p <= ts) return 0;
    const char *q = p - 1;
    while (q > ts && ((u8)*q & 0xc0) == 0x80) q--;
    return is_word_at(q);
}

static void too_many(void) {
    fail("the pattern needs too many steps on this text for the computer: make it simpler, or the text shorter");
}

/* the ASCII character the instruction after a STAR at pc takes (as its byte, not a letter of either case), or -1 */
static int star_lit(int pc) { /* (past a group's end: SAVE takes no character) */
    const inst *in = &prog[pc + 1];
    while (in->op == SAVE) in++;
    return in->op == CHAR && in->c < 0x80 && !(icase && in->c - 'a' < 26) ? (int)in->c : -1;
}

static int run(const char *start, int no_empty) { /* (no_empty: a match must not end where it starts) */
    int top = 0;
    for (int i = 0; i < 2 * n_groups + 2; i++) caps[i] = NONE; /* (the groups', and the repeats' marks, in use) */
    for (int i = 2 * GROUPS; i < 2 * GROUPS + n_loops; i++) caps[i] = NONE;
    steps++; /* (a try: its start) */
    stack[top++] = (entry){0, (u16)(start - ts)};
    while (top) {
        entry t = stack[--top];
        if (t.pc < 0) {
            caps[-1 - t.pc] = t.sp;
            continue;
        }
        if (t.pc == STAR_MARK) continue; /* (a STAR whose shorter runs are all tried) */
        int pc = t.pc;
        const char *sp = ts + t.sp;
        if (pc & STAR_NEXT) { /* (a STAR's run a character shorter: the mark below has where it began) */
            const char *s0 = ts + stack[top - 1].sp;
            int at = pc & ~STAR_NEXT, lit = star_lit(at);
            steps++;
            if (lit >= 0) { /* (the next instruction a character: the run's last place before it that has it) */
                const char *r = sp - 1;
                while (r >= s0 && (u8)*r != lit) r--;
                steps += (int)((sp - r) >> 2);
                if (r < s0) { /* (none: the STAR is done) */
                    top--;
                    continue;
                }
                sp = r;
            } else {
                sp--;
                while (sp > s0 && ((u8)*sp & 0xc0) == 0x80) sp--;
            }
            if (sp > s0) stack[top++] = (entry){(i16)pc, (u16)(sp - ts)};
            else top--;
            pc = at + 1;
        }
        for (;;) {
            if (++steps > STEPS) too_many();
            const inst *in = &prog[pc];
            int n;
            if (in->op == CHAR) {
                if (sp >= te_) break;
                u32 c = (u8)*sp < 0x80 ? (n = 1, (u8)*sp) : utf8(sp, te_, &n);
                if (fold(c) != in->c) break;
                sp += n, pc++;
            } else if (in->op == ANY) {
                if (sp >= te_ || *sp == '\n') break;
                utf8(sp, te_, &n);
                sp += n, pc++;
            } else if (in->op == CLASS) {
                if (sp >= te_) break;
                u8 b = (u8)*sp;
                if (b < 0x80) { /* (an ASCII character: its bit) */
                    if (!(cbits[in->x][b >> 3] >> (b & 7) & 1)) break;
                    sp++, pc++;
                } else {
                    if (!class_has(&classes[in->x], utf8(sp, te_, &n))) break;
                    steps += 2 + (classes[in->x].n >> 2); /* (beyond ASCII: 2 steps more, and one for 4 ranges) */
                    sp += n, pc++;
                }
            } else if (in->op == SPLIT) {
                if (top >= STACK - 1) too_many();
                stack[top++] = (entry){(i16)in->c, (u16)(sp - ts)};
                pc = in->x;
            } else if (in->op == JMP)
                pc = in->x;
            else if (in->op == SAVE) {
                if (top >= STACK - 1) too_many();
                stack[top++] = (entry){(i16)(-1 - in->x), caps[in->x]};
                caps[in->x] = (u16)(sp - ts);
                pc++;
            } else if (in->op == BOL) {
                if (sp != ts) break;
                pc++;
            } else if (in->op == EOL) {
                if (sp != te_ && !(sp + 1 == te_ && *sp == '\n')) break;
                pc++;
            } else if (in->op == WORDB || in->op == NWORDB) {
                int b = (sp > ts && (u8)sp[-1] < 0x80 ? ASCII_WORD(sp[-1]) : word_before(sp)) !=
                        (sp < te_ && (u8)*sp < 0x80 ? ASCII_WORD(*sp) : is_word_at(sp));
                if (b != (in->op == WORDB) || ts == te_) break; /* (Python's \B: not in an empty text either) */
                pc++;
            } else if (in->op == STAR) {
                const char *s0 = sp;
                while (sp < te_) {
                    u8 b = (u8)*sp;
                    if (b < 0x80) { /* (ASCII: at once) */
                        if (in->x == CLASS ? !(cbits[in->c][b >> 3] >> (b & 7) & 1) : in->x == CHAR ? fold(b) != in->c : b == '\n') break;
                        sp++;
                    } else {
                        u32 c = utf8(sp, te_, &n);
                        if (in->x == CLASS ? !class_has(&classes[in->c], c) : in->x == CHAR ? fold(c) != in->c : 0) break;
                        steps += 2 + (in->x == CLASS ? classes[in->c].n >> 2 : 0);
                        sp += n;
                    }
                }
                steps += (sp - s0) >> 2; /* (a step for 4 characters) */
                int lit = star_lit(pc);
                if (lit >= 0 && !(sp < te_ && (u8)*sp == lit)) { /* (the character after it not there: back to one) */
                    const char *r = sp - 1;
                    while (r >= s0 && (u8)*r != lit) r--;
                    steps += (int)((sp - r) >> 2);
                    if (r < s0) break;
                    sp = r;
                }
                if (sp > s0) {
                    if (top >= STACK - 2) too_many();
                    stack[top++] = (entry){STAR_MARK, (u16)(s0 - ts)};
                    stack[top++] = (entry){(i16)(pc | STAR_NEXT), (u16)(sp - ts)};
                }
                pc++;
            } else if (in->op == CHK) /* (a repeat's round that matched nothing: the repeat ends) */
                pc = sp - ts == caps[in->x] ? (int)in->c : pc + 1;
            else { /* MATCH */
                if (no_empty && sp == start) break;
                return 1;
            }
        }
    }
    return 0;
}

/* the bytes a match can start with (a set of 256 bits), or none if it may start with anything or match nothing:
   the next match is looked for only where one of them is (a character's first byte: never 0x80..0xbf) */
static u8 first_set[32];
static int has_first, first_one; /* (first_one: the only byte, or -1) */

static int class_high(const cls *k) { /* the class may take a character beyond ASCII */
    if (k->neg || (k->sets & (C_W | C_S | C_ND | C_NW | C_NS))) return 1;
    for (int i = 0; i < k->n; i++)
        if (ranges[k->first + i][1] >= 0x80) return 1;
    return 0;
}

static void first_bytes(void) {
    has_first = 0;
    memset(first_set, 0, 32);
    u16 *work = alloc(sizeof(u16) * (n_prog + 1));
    u8 *seen = alloc(n_prog + 1);
    memset(seen, 0, n_prog + 1);
    int top = 0;
    work[top++] = 1, seen[1] = 1;
#define ADD(b) (first_set[(b) >> 3] |= (u8)(1 << ((b) & 7)))
    while (top) {
        int pc = work[--top], a = -1, b = -1;
        const inst *in = &prog[pc];
        switch (in->op) {
        case CHAR: {
            u32 c = in->c;
            ADD(c < 0x80 ? c : c < 0x800 ? 0xc0 | c >> 6 : c < 0x10000 ? 0xe0 | c >> 12 : 0xf0 | c >> 18);
            if (icase && c - 'a' < 26) ADD(c - 32);
            break;
        }
        case CLASS: {
            const cls *k = &classes[in->x];
            for (int i = 0; i < 16; i++) first_set[i] |= cbits[in->x][i];
            if (class_high(k))
                for (u32 c = 0xc0; c < 0x100; c++) ADD(c);
            break;
        }
        case STAR:
            if (in->x == ANY) return;
            if (in->x == CHAR) {
                u32 c = in->c;
                ADD(c < 0x80 ? c : c < 0x800 ? 0xc0 | c >> 6 : c < 0x10000 ? 0xe0 | c >> 12 : 0xf0 | c >> 18);
                if (icase && c - 'a' < 26) ADD(c - 32);
            } else {
                const cls *k = &classes[in->c];
                for (int i = 0; i < 16; i++) first_set[i] |= cbits[in->c][i];
                if (class_high(k))
                    for (u32 c = 0xc0; c < 0x100; c++) ADD(c);
            }
            a = pc + 1;
            break;
        case ANY: case MATCH: return; /* (anything, or nothing: no skipping) */
        case SPLIT: a = in->x, b = (int)in->c; break;
        case JMP: a = in->x; break;
        case CHK: a = pc + 1, b = (int)in->c; break;
        default: a = pc + 1; /* (SAVE and the anchors: the next one's first byte is still the match's) */
        }
        if (a >= 0 && !seen[a]) seen[a] = 1, work[top++] = (u16)a;
        if (b >= 0 && !seen[b]) seen[b] = 1, work[top++] = (u16)b;
    }
#undef ADD
    int count = 0;
    first_one = -1;
    for (int c = 0; c < 256; c++)
        if (first_set[c >> 3] >> (c & 7) & 1) count++, first_one = c;
    if (count != 1 || first_one >= 0x80) first_one = -1; /* (find_byte takes an ASCII byte) */
    has_first = 1;
}

static str cap(int g) { return caps[2 * g] == NONE || caps[2 * g + 1] == NONE ? (str){0, 0} : (str){ts + caps[2 * g], ts + caps[2 * g + 1]}; }

/* the next match from *pos on, as Python's finditer finds it: after an empty match, one that is not empty may
   start at the same place (*no_empty). Its bounds in caps[0], caps[1]; *pos and *no_empty for the next. */
static int atom_at(const node *a, const char *p, int *n) { /* the one-character atom matches at p */
    if (p >= te_) return 0;
    u8 b = (u8)*p;
    if (b < 0x80 && a->type == N_CLASS) return *n = 1, cbits[a->a][b >> 3] >> (b & 7) & 1; /* (ASCII: at once) */
    u32 c = utf8(p, te_, n);
    return a->type == N_ANY ? c != '\n' : a->type == N_CHAR ? fold(c) == fold((u32)a->a)
           : c < 0x80 ? cbits[a->a][c >> 3] >> (c & 7) & 1 : class_has(&classes[a->a], c);
}

static int start_wb; /* (the pattern starts with \b: a match starts only where one is) */

static int next_match(const char **pos, int *no_empty) {
    const char *p = *pos;
    int ne = *no_empty, n;
    for (;;) {
        if (has_first) { /* (to the next place a match can start: its first byte one of the pattern's first bytes) */
            const char *q = p;
            if (first_one >= 0) q = find_byte(p, te_, first_one);
            else
                while (q < te_ && !(first_set[(u8)*q >> 3] >> (*q & 7) & 1)) q++;
            if (!q || q >= te_) return 0;
            if (q != p) ne = 0;
            p = q;
        }
        if (start_wb && p < te_) { /* (no \b here: no match starts here; the next place one is) */
            int wb = p > ts && (u8)p[-1] < 0x80 ? ASCII_WORD(p[-1]) : word_before(p);
            const char *q = p;
            for (;;) { /* (along ASCII characters of the same kind at once) */
                int wa = q < te_ && (u8)*q < 0x80 ? ASCII_WORD(*q) : q < te_ ? is_word_at(q) : 0;
                if (wa != wb || q >= te_) break;
                if ((u8)*q < 0x80) q++;
                else utf8(q, te_, &n), q += n, steps++;
            }
            steps += (int)((q - p) >> 2);
            if (q != p) {
                p = q, ne = 0;
                continue;
            }
        }
        if (run(p, ne)) break;
        if (p >= te_) return 0;
        /* (a pattern C+ R or C* R that failed here, at a run of C's: from a later place in the run it tries fewer of
           the same ends of the run, so it fails as well; the next place to try is the run's end) */
        const char *q = p;
        if (lead_rep >= 0 && !ne)
            for (const node *a = &nodes[lead_rep]; atom_at(a, q, &n); q += n)
                if ((u8)*q >= 0x80) steps += 1 + (a->type == N_CLASS ? classes[a->a].n >> 2 : 0); /* (beyond ASCII: a step) */
        steps += (int)((q - p) >> 2);
        if (q == p) utf8(p, te_, &n), q = p + n;
        p = q;
        ne = 0;
    }
    str m = cap(0);
    *pos = m.e;
    *no_empty = m.e == m.p;
    return 1;
}

static void setup(void) {
    str pat = param("pattern"), t = param("text");
    if (!have(pat)) fail("missing argument: pattern");
    if (!have(t)) fail("missing argument: text");
    compile(pat);
    first_bytes();
    int pc = 1;
    while (prog[pc].op == SAVE) pc++;
    start_wb = prog[pc].op == WORDB;
    ts = t.p, te_ = t.e;
    steps = 0;
    stack = alloc_low(sizeof(entry) * STACK);
}

void regex_find(void) {
    setup();
    int n = 0, w = n_groups + 1;
    u16 *found = alloc(sizeof(u16) * 2 * w * 101); /* (each match's groups' bounds) */
    const char *p = ts;
    int ne = 0;
    while (n < 101 && next_match(&p, &ne)) {
        for (int i = 0; i < 2 * w; i++) found[2 * w * n + i] = caps[i];
        n++;
    }
    if (!n) {
        say("no matches");
        str pat = trim(param("pattern"));
        int end = pat.e - pat.p >= 1 && pat.e[-1] == '$' && (pat.e - pat.p < 2 || pat.e[-2] != '\\'), start = *pat.p == '^';
        if ((end || start) && (find_byte(ts, te_, ' ') || find_byte(ts, te_, '\n'))) /* (a model's slip: $ for a word's end) */
            say(" (^ and $ are the whole text's start and end here: for each word's, use \\b)");
        return;
    }
    if (n > 100) say("more than 100 matches; the first 100:");
    else say_int(n), say(n == 1 ? " match:" : " matches:");
    for (int i = 0; i < n && i < 100; i++) {
        for (int k = 0; k < 2 * w; k++) caps[k] = found[2 * w * i + k];
        put('\n');
        say_str(cap(0));
        if (n_groups) {
            say(" (groups: ");
            for (int g = 1; g <= n_groups; g++) {
                if (g > 1) say(", ");
                str x = cap(g);
                if (have(x)) say_str(x);
                else say("none");
            }
            put(')');
        }
    }
}

void regex_replace(void) {
    setup();
    str with = param("replacement");
    static const char none[] = "";
    if (!have(with)) with = (str){none, none};
    /* the replacement's parts are checked first: \1 to \9 (\g<n>), \n, \t, \\ */
    for (const char *q = with.p; q < with.e; q++)
        if (*q == '\\') {
            q++;
            if (q >= with.e) fail("the replacement ends in a \\");
            if (digit(*q) && *q - '0' > n_groups) fail("the replacement names a group the pattern does not have");
        }
    char *out = alloc(1), *o = out; /* (the new text, written after the call's text in memory) */
    size_t room = heap_left();
    const char *p = ts, *last = ts;
    int n = 0, ne = 0;
#define OUT(c)                                                                                                       \
    do {                                                                                                             \
        if ((size_t)(o - out) + 1 >= room) fail("the new text does not fit the computer's memory");                 \
        *o++ = (c);                                                                                                  \
    } while (0)
    while (next_match(&p, &ne)) {
        str m = cap(0);
        for (const char *q = last; q < m.p; q++) OUT(*q);
        for (const char *q = with.p; q < with.e; q++) {
            if (*q != '\\') {
                OUT(*q);
                continue;
            }
            q++;
            int g = -1;
            if (digit(*q)) g = *q - '0';
            else if (*q == 'g' && q + 2 < with.e && q[1] == '<' && digit(q[2])) {
                g = q[2] - '0';
                q += 2;
                while (q < with.e && *q != '>') q++;
            } else if (*q == 'n') OUT('\n');
            else if (*q == 't') OUT('\t');
            else OUT(*q);
            str x = g >= 0 ? cap(g) : (str){0, 0};
            if (have(x))
                for (const char *c = x.p; c < x.e; c++) OUT(*c);
        }
        n++;
        last = m.e;
    }
    for (const char *q = last; q < te_; q++) OUT(*q);
    say_int(n);
    say(n == 1 ? " replacement:\n" : " replacements:\n");
    say_str((str){out, o});
}
