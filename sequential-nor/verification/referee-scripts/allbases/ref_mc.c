/* Referee's Monte Carlo: max_i D(i) (relevance-pruned depth) on the eps-smoothing of a base.
   usage: ref_mc <base> <eps> <n> <seeds>
   base: 2 -> j_i=i-2 ; 3 -> j_i=i-3 ; 23 -> reach 2 at even i, 3 at odd i ; 0 -> j_i=0
   D by the four cases of Lemma lem:status (checked against the iteration in ref_check.py part0).
   memory: 3 bytes per position (uint16 depth + uint8 value). */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

static uint64_t st;
static inline uint64_t rnd(void) {            /* splitmix64 */
    uint64_t z = (st += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline double unif(void) { return (rnd() >> 11) * (1.0 / 9007199254740992.0); }
static inline uint64_t below(uint64_t m) {    /* uniform on 0..m-1, rejection */
    uint64_t lim = UINT64_MAX - (UINT64_MAX % m), r;
    do { r = rnd(); } while (r >= lim);
    return r % m;
}

int main(int argc, char **argv) {
    if (argc < 5) return 1;
    int base = atoi(argv[1]); double eps = atof(argv[2]);
    long n = atol(argv[3]); int seeds = atoi(argv[4]);
    uint8_t *x = malloc(n + 1); uint16_t *D = malloc(2 * (n + 1));
    if (!x || !D) return 2;
    double sum = 0; long mn = 1L << 60, mx = 0;
    printf("base %d eps %.3f n %ld :", base, eps, n);
    for (int s = 1; s <= seeds; s++) {
        st = 0x1234567ULL * (uint64_t)s + 77;
        x[0] = 1; D[0] = 0; x[1] = 0; D[1] = 1;
        long best = 1;
        for (long k = 2; k <= n; k++) {
            long p;
            if (unif() < eps) p = (long)below((uint64_t)k);
            else {
                long d = base == 2 ? 2 : base == 3 ? 3 : base == 23 ? ((k & 1) ? 3 : 2) : k;
                p = k - d; if (p < 0) p = 0;
            }
            int notg = (p == k - 1);
            int rel = (x[k-1] == 0);
            int reset = (!notg) && x[p] == 1;
            x[k] = (uint8_t)(rel && x[p] == 0);
            long dk;
            if (reset && rel) dk = 1 + D[p];
            else if (reset) dk = D[k-1] < 1 + D[p] ? D[k-1] : 1 + D[p];
            else if (rel && !notg) dk = D[k-1] > 1 + D[p] ? D[k-1] : 1 + D[p];
            else dk = D[k-1];
            if (dk > 65000) { printf(" overflow\n"); return 3; }
            D[k] = (uint16_t)dk;
            if (dk > best) best = dk;
        }
        printf(" %ld", best);
        sum += best; if (best < mn) mn = best; if (best > mx) mx = best;
    }
    printf("   mean %.1f  mean/ln(n) %.2f\n", sum / seeds, sum / seeds / log((double)n));
    free(x); free(D);
    return 0;
}
