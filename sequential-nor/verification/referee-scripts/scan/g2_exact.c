/* Referee's independent evaluation of G_2(W,d) for scanning words (floating point).
   Method: explicit 16-state joint law (window copy a, window copy b, chain copy a, chain copy b),
   stepped from the semantic definition; negative j summed explicitly (no closed form) down to -JMAX.
   usage: g2_exact eps maxT           all words of period <= maxT at all self-consistent densities
          g2_exact eps word UURR...   one word
          g2_exact eps runs           U R^k for k=0..120, and random words with long blocks      */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define JMAX 600
static double eps, vp, A, B, c;
static double d, qP, alpha;

/* one gate of the plain chain acting on value w with read status s (0 kept,1 res->1,2 res->0) and target t */
static inline int pstep(int w, int s, int t) { int y = s == 0 ? t : (s == 1 ? 1 : 0); return !(w || y); }
static inline double sprob(int s) { return s == 0 ? c : (s == 1 ? A : B); }

/* advance the window pair distribution over 16 states idx = wa*8+wb*4+ea*2+eb.  diff: targets (0,1). */
static void win_adv(double *st, int diff) {
    double n[16] = {0};
    for (int i = 0; i < 16; i++) { if (st[i] == 0) continue; int wa = i >> 3, wb = (i >> 2) & 1, e = i & 3;
        for (int s = 0; s < 3; s++) {
            if (diff) { int a2 = pstep(wa, s, 0), b2 = pstep(wb, s, 1); n[a2 * 8 + b2 * 4 + e] += st[i] * sprob(s); }
            else for (int t = 0; t < 2; t++) { int a2 = pstep(wa, s, t), b2 = pstep(wb, s, t);
                n[a2 * 8 + b2 * 4 + e] += st[i] * sprob(s) * (t ? d : 1 - d); } } }
    memcpy(st, n, sizeof n);
}
/* a chain gate reads the window value */
static void e_read(double *st) {
    double n[16] = {0};
    for (int i = 0; i < 16; i++) { if (st[i] == 0) continue; int wa = i >> 3, wb = (i >> 2) & 1, ea = (i >> 1) & 1, eb = i & 1;
        for (int s = 0; s < 3; s++) { int ya = s == 0 ? wa : (s == 1 ? 1 : 0), yb = s == 0 ? wb : (s == 1 ? 1 : 0);
            int a2 = !(ea || ya), b2 = !(eb || yb); n[(i & 12) + a2 * 2 + b2] += st[i] * sprob(s); } }
    memcpy(st, n, sizeof n);
}
static double ediff(const double *st) { double s = 0; for (int i = 0; i < 16; i++) if (((i >> 1) & 1) != (i & 1)) s += st[i]; return s; }

/* clean block: start P[e=0]=p; returns out (P[e=0] after) and ones */
static void clean_block(int k, double p, double *out, double *ones) {
    double st[16] = {0};
    /* window before position 0: stationary: value 0 w.p. alpha */
    for (int w = 0; w < 2; w++) for (int e = 0; e < 2; e++) st[w * 12 + e * 3] = (w ? 1 - alpha : alpha) * (e ? 1 - p : p);
    double on = 0;
    for (int t = 0; t <= k; t++) { win_adv(st, 0); e_read(st); for (int i = 0; i < 16; i++) if (i & 2) on += st[i]; }
    double o = 0; for (int i = 0; i < 16; i++) if (!(i & 2)) o += st[i];
    *out = o; *ones = on;
}
/* pass-through: S = expected number of gates up to which all reads are 0, Q = P[all reads 0] */
static void pass(int k, double *S, double *Q) {
    /* use the pair machinery: chain copies (0,1)?? no: direct: track window value and "all reads zero" */
    double w[2] = {alpha, 1 - alpha}, s = 0, z = 0;
    for (int t = 0; t <= k; t++) {
        double n[2] = {0, 0};
        for (int v = 0; v < 2; v++) for (int st = 0; st < 3; st++) for (int tg = 0; tg < 2; tg++)
            n[pstep(v, st, tg)] += w[v] * sprob(st) * (tg ? d : 1 - d);
        /* read of the chain gate is 0: resampled to 0, or kept and value 0 */
        w[0] = n[0] * (B + c); w[1] = n[1] * B; z = w[0] + w[1]; s += z; }
    *S = s; *Q = z;
}
#define MAXB 4096
static int nb, kk[MAXB];
static double pin[MAXB];
static double density_of(void) {
    /* affine composition */
    double a = 0, b = 1; /* p -> a + b p */
    for (int i = 0; i < nb; i++) { double o0, o1, on; clean_block(kk[i], 0, &o0, &on); clean_block(kk[i], 1, &o1, &on); a = o0 + (o1 - o0) * a; b = (o1 - o0) * b; }
    double p = a / (1 - b), ones = 0; int T = 0;
    for (int i = 0; i < nb; i++) { double o, on; pin[i] = p; clean_block(kk[i], p, &o, &on); ones += on; p = o; T += kk[i] + 1; }
    return ones / T;
}
static void setd(double dd) { d = dd; qP = B + c * (1 - d); alpha = 1 / (1 + qP); }
static double G2(void) {
    int T = 0; for (int i = 0; i < nb; i++) T += kk[i] + 1;
    density_of();
    static double S[MAXB], Q[MAXB], V[MAXB + 1];
    for (int i = 0; i < nb; i++) pass(kk[i], &S[i], &Q[i]);
    /* V[i]: expected number of differing gates from block i on, given a difference at the gate before block i */
    { double a = 0, b = 1; for (int i = 0; i < nb; i++) { a += b * S[i]; b *= Q[i]; } V[0] = a / (1 - b); V[nb] = V[0];
      for (int i = nb - 1; i >= 1; i--) V[i] = S[i] + Q[i] * V[i + 1]; }
    /* window pair law at position -1... for j<=0: accumulate over j the law of the window pair after position 0's gate?  No:
       accumulate the pair law just BEFORE position 0 for j<0 (window positions j..-1 processed), plus j=0 handled by diff at 0 */
    double tot = 0;
    for (int i = 0; i < nb; i++) { int k = kk[i]; double p = pin[i];
        /* j < 0: window-only evolution; 4 states (wa,wb); use the 16-state arrays with e=(0,0) */
        double acc[16] = {0};
        { double st[16] = {0}; st[0] = alpha; st[12] = 1 - alpha; win_adv(st, 1); /* after position j */
          /* keep only differing states */
          for (int j = -1; j >= -JMAX; j--) { /* st = law after position j ... advance through positions j+1..-1 lazily: iterate */
              for (int s = 0; s < 16; s++) acc[s] += (((s >> 3) & 1) != ((s >> 2) & 1)) ? st[s] : 0;
              win_adv(st, 0); for (int s = 0; s < 16; s++) if (((s >> 3) & 1) == ((s >> 2) & 1)) st[s] = 0; } }
        /* acc = sum over j<0 of the law of a differing window pair after position -1 */
        { double st[16] = {0};
          for (int s = 0; s < 16; s++) if (acc[s] > 0) { st[(s & 12) + 0] += acc[s] * p; st[(s & 12) + 3] += acc[s] * (1 - p); }
          for (int t = 0; t <= k; t++) { win_adv(st, 0); e_read(st); tot += ediff(st); }
          tot += ediff(st) * V[i + 1]; }
        for (int j = 0; j <= k; j++) { double st[16] = {0};
          for (int w = 0; w < 2; w++) for (int e = 0; e < 2; e++) st[w * 12 + e * 3] = (w ? 1 - alpha : alpha) * (e ? 1 - p : p);
          for (int t = 0; t <= k; t++) { win_adv(st, t == j); e_read(st); if (t >= j) tot += ediff(st); }
          tot += ediff(st) * V[i + 1]; } }
    return tot / T;
}
static double fden(double dd) { setd(dd); return density_of() - dd; }
/* all roots of density(d)=d; returns count */
static int roots(double *r, int grid) {
    int n = 0; double x0 = 0, f0 = fden(0);
    for (int i = 1; i <= grid; i++) { double x1 = (double)i / grid, f1 = fden(x1);
        if (f0 == 0) r[n++] = x0;
        else if (f0 * f1 < 0) { double a = x0, b = x1, fa = f0; for (int it = 0; it < 60; it++) { double m = 0.5 * (a + b), fm = fden(m); if (fa * fm <= 0) b = m; else { a = m; fa = fm; } } r[n++] = 0.5 * (a + b); }
        x0 = x1; f0 = f1; if (n >= 8) break; }
    return n;
}
static void setword(const char *w) { nb = 0; for (const char *s = w; *s; s++) { if (*s == 'U') kk[nb++] = 0; else kk[nb - 1]++; } }
static unsigned long long rs = 88172645463325252ULL;
static double rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }
int main(int argc, char **argv) {
    eps = atof(argv[1]); vp = ((2 * eps + 1) - sqrt(4 * eps + 1)) / (2 * eps); A = eps * vp; B = eps * (1 - vp); c = 1 - eps;
    double q = (-(1 - B) + sqrt((1 - B) * (1 - B) + 4 * (1 - A))) / 2, lam = (1 - eps) / (1 - q * q);
    printf("eps %.4f varpi %.6f q %.6f pi0 %.6f lambda %.6f lambda^2 %.6f\n", eps, vp, q, q / (1 + q), lam, lam * lam);
    double r[8];
    if (!strcmp(argv[2], "word")) { setword(argv[3]); int n = roots(r, 400);
        for (int i = 0; i < n; i++) { setd(r[i]); printf("%s d %.8f G2 %.8f\n", argv[3], r[i], G2()); } return 0; }
    if (!strcmp(argv[2], "runs")) { double best = 0; int multi = 0;
        for (int k = 0; k <= 120; k++) { nb = 1; kk[0] = k; int n = roots(r, 200); if (n != 1) multi++;
            for (int i = 0; i < n; i++) { setd(r[i]); double g = G2(); if (g > best) best = g; if (k <= 6 || k % 20 == 0) printf("U R^%d d %.6f G2 %.6f\n", k, r[i], g); } }
        printf("U R^k best %.6f, words with roots != 1: %d\n", best, multi);
        /* random words with long blocks, and mixes of U with long runs */
        double best2 = 0; char desc[256] = "";
        for (int trial = 0; trial < 3000; trial++) { nb = 1 + (int)(rnd() * 8); int mode = trial % 3;
            for (int i = 0; i < nb; i++) kk[i] = mode == 0 ? (int)(rnd() * 121) : (mode == 1 ? (rnd() < 0.5 ? 0 : (int)(rnd() * 121)) : (rnd() < 0.7 ? (int)(rnd() * 3) : (int)(rnd() * 121)));
            int n = roots(r, 100); if (n != 1) multi++;
            for (int i = 0; i < n; i++) { setd(r[i]); double g = G2(); if (g > best2) { best2 = g; desc[0] = 0; for (int b = 0; b < nb; b++) { char t[16]; sprintf(t, "%d ", kk[b]); strcat(desc, t); } } } }
        printf("random long words best %.6f blocks k: %s ; roots != 1 count %d\n", best2, desc, multi);
        return 0; }
    int maxT = atoi(argv[2]); double best = 0; char bw[64] = ""; long nw = 0, multi = 0, over = 0; double second = 0; char sw[64] = "";
    for (int T = 1; T <= maxT; T++) for (unsigned m = 0; m < (1u << (T - 1)); m++) {
        char w[64]; w[0] = 'U'; for (int i = 1; i < T; i++) w[i] = (m >> (i - 1)) & 1 ? 'R' : 'U'; w[T] = 0; setword(w); nw++;
        int n = roots(r, T <= 8 ? 400 : 60); if (n != 1) { multi++; printf("word %s has %d densities\n", w, n); }
        for (int i = 0; i < n; i++) { setd(r[i]); double g = G2(); if (g > lam * lam + 1e-9) over++;
            if (g > best) { best = g; strcpy(bw, w); } if (m != 0 && g > second) { second = g; strcpy(sw, w); } } }
    printf("words %ld, with a number of densities != 1: %ld, above lambda^2: %ld\nbest G2 %.8f word %s\nbest with an R: %.8f word %s\n", nw, multi, over, best, bw, second, sw);
    return 0;
}
