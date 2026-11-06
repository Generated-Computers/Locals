/* Qwenputer's toolbox in C: tools the model's own RV32 computer runs. build.py compiles them for RV32IM + Zba and
   links them at fixed addresses after calc's firmware and qwenputer_tools.s, whose dispatcher calls c_main for a
   name it does not have (see that file for a run, the text buffer and the memory map). With -DHOST the same code
   builds for this Mac (host.c), to test it against Python's own json, re, csv... */
#ifndef TOOLS_H
#define TOOLS_H

typedef unsigned char u8;
typedef unsigned short u16;
typedef short i16;
typedef unsigned int u32;
typedef int i32;
typedef long long i64;
typedef unsigned long long u64;
typedef __SIZE_TYPE__ size_t;

typedef struct {
    const char *p, *e; /* [p, e); p == 0: none */
} str;

/* the call: the function's name, the call's text [text, text_end) and its parameters (read by c_main) */
extern const char *call_name;
extern const char *text, *text_end;

str param(const char *name); /* a parameter's value, without the newline after <parameter=...> and before its end */
static inline int have(str s) { return s.p != 0; }
str trim(str s);
int str_is(str s, const char *z);    /* s equals the NUL-terminated z */
int str_ieq(str a, str b);           /* equal, ASCII letters in either case */
static inline int space(int c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
static inline int digit(int c) { return (unsigned)(c - '0') < 10; }
static inline int lower(int c) { return (unsigned)(c - 'A') < 26 ? c + 32 : c; }

/* output, a byte a token (256.. the special tokens); into [capture_at, capture_end) instead while capture_at is set */
extern char *capture_at, *capture_end;
#ifdef HOST
void put(int c);
#else
static inline void put(int c) { /* (inline: a call a byte cost the computer 3 more trace tokens) */
    if (capture_at) {
        if (capture_at < capture_end) *capture_at++ = (char)c;
        return;
    }
    register int a0 asm("a0") = c;
    register int a1 asm("a1") = 0;
    register int a2 asm("a2") = 0;
    register int a7 asm("a7") = 1;
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
}
#endif
void say(const char *z);
void say_str(str s);
void say_int(i64 v);
void say_u(u64 v);
void say_fixed(i64 v, int frac); /* v / 10^frac, its fraction's trailing zeros dropped */

/* an error response, "error (MSG)", and the run's end (no return) */
__attribute__((noreturn)) void fail(const char *msg);
/* the run's end after what the tool has printed */
__attribute__((noreturn)) void done(void);

/* the first byte c in [p, e), or 0 (4 bytes at a time) */
const char *find_byte(const char *p, const char *e, int c);
/* the first z (n bytes, n >= 2) in [p, e), or 0; find() of a string literal */
const char *find_n(const char *p, const char *e, const char *z, size_t n);
#define find(p, e, lit) find_n((p), (e), (lit), sizeof(lit) - 1)
/* a word with a byte b among its 4 */
#define HAS_BYTE(w, b) ((((w) ^ (b) * 0x01010101u) - 0x01010101u) & ~((w) ^ (b) * 0x01010101u) & 0x80808080u)

/* memory for one run: from the call's text's end up to 0x6c00 (the stack is above it) */
void *alloc(size_t n);
size_t heap_left(void);
char *heap_mark(void);       /* (where the heap is: what heap_back gives back to) */
void heap_back(char *at);    /* (what was made from `at` on is free again) */
void *alloc_top(size_t n);   /* (from the heap's end, for a while: free_top gives it back) */
void free_top(void);
/* what a call says when its memory runs out (0: the computer's 15 KB; set again at each call) */
extern const char *oom_why;
void *alloc_low(size_t n); /* (5 KB for tables, apart from the heap; then the heap) */
size_t low_left(void);
/* memory between the heap and `here` (a local's address) kept for the stack: the floor the stack may go to */
char *heap_reserve(char *here, size_t stack);
/* where the stack is now (to compare with such a floor) */
static inline __attribute__((always_inline)) char *stack_now(void) {
#ifdef HOST
    return (char *)__builtin_frame_address(0);
#else
    char *sp;
    asm volatile("mv %0, sp" : "=r"(sp));
    return sp;
#endif
}
void *memcpy(void *d, const void *s, size_t n);
void *memset(void *d, int c, size_t n);
void *memmove(void *d, const void *s, size_t n);
int memcmp(const void *a, const void *b, size_t n);
size_t strlen(const char *z);

/* numbers as tools read them: up to 18 significant digits, as v / 10^frac (frac <= 9); 0 if the text is not one.
   Thousands separators (groups of 3) and a leading $, EUR or GBP sign or a trailing % are allowed where `loose`. */
typedef struct {
    i64 v;
    int frac;
} num;
int read_num(str s, num *out, int loose);
int num_cmp(num a, num b);

/* big whole numbers (big.c): base 2^32 limbs, least significant first; 512 bits (154 decimal digits) */
#define BIG_LIMBS 16
typedef struct {
    int n;
    u32 d[BIG_LIMBS];
} big;
void big_set(big *a, u64 v);
int big_cmp(const big *a, const big *b);
void big_add(big *r, const big *a, const big *b);
void big_sub(big *r, const big *a, const big *b); /* a >= b */
void big_mul_small(big *r, const big *a, u32 m, u32 add);
void big_mul(big *r, const big *a, const big *b);
u32 big_div_small(big *q, const big *a, u32 m);
void big_divmod(big *q, big *r, const big *a, const big *b);
void big_pow10(big *a, int k);
void big_isqrt(big *r, const big *a);
int big_digits(const big *a, char *out);
void say_big(const big *a);
void say_decimal(const big *a, int neg, int scale, int max_frac);
int read_decimal(str s, big *a, int *neg, int *scale);

/* the tools */
void json_check(void);
void json_format(void);
void json_get(void);
void json_count(void);
void column_stats(void);
void table_sort(void);
void table_filter(void);
void table_totals(void);
void compare_versions(void);
void sort_versions(void);
void compare_lists(void);
void truth_table(void);
void logic_equivalent(void);
void regex_find(void);
void regex_replace(void);
void remember(void);
void recall(void);
void forget(void);
void check_call(void);
void copy_check(void);
void run_python(void);
void stats(void);
void percent_change(void);
void round_number(void);
void compound_interest(void);
void loan_payment(void);
void is_prime(void);
void factor(void);
void convert_units(void);
void roman_numeral(void);
void number_to_words(void);
void replace_text(void);
void change_case(void);
void text_part(void);
void sort_lines(void);
void unique_items(void);
void word_frequency(void);
void is_palindrome(void);
void is_anagram(void);
void validate(void);
void encode(void);
void decode(void);
void edit_distance(void);
void time_add(void);
void time_between(void);
void age(void);
void date_info(void);
void unix_time(void);
void convert_timezone(void);
void check_digit(void);
void hash_text(void);

/* JSON, for the tools that read it (json.c) */
typedef struct {
    const char *at;  /* where reading stopped */
    const char *err; /* 0, or what is wrong at `at` */
    int depth;
    u32 *ix;          /* if not 0: each object's, array's and long string's start << 16 | end, from base, by start */
    int nix, cap;
    const char *base;
} jread;
const char *json_value(jread *r, const char *p, const char *e); /* the end of the value at p (after spaces), or 0 */
const char *json_space(const char *p, const char *e);

#endif
