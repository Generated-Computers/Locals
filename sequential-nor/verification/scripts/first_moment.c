// first_moment.c: exact growth rate of the expected number of base-pointer chains, for a periodic
// base with short pointers (gate q reads q - d[q mod p]; d = 1 is a NOT gate), in the model in which
// a resampled gate reads a one with probability p, independently of everything else.
//   mode 0: Kleene-prefix count (every non-NOT gate of a window may be the step's gate)
//   mode 1: Dlast count (only relevant gates and the last reset; bounds the relevance-pruned and
//           prefix-Jacobi iterations)
// With plo < phi every transition is the entrywise minimum over p in [plo, phi] (a lower bound).
// usage: first_moment mode eps plo phi p d0 d1 ...     (plo = phi = 0: p = varpi(eps))
// The rate is the ratio of total masses after 70 rounds (power iteration).
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
static int W, p, d[8], NS, mode; static double eps, plo, phi;
// kind 0 = any outcome, 1 = not a reset, 2 = kept (the step's gate)
static void step(int f, const double *v, double *out, int kind) {
  memset(out, 0, sizeof(double) * NS); int dd = d[f];
  for (int s = 0; s < NS; s++) { double m = v[s]; if (m == 0) continue; int b0 = s & 1;
    int vv = (dd == 1) ? b0 : ((s >> (dd - 1)) & 1); int reset = (dd >= 2) && vv;
    int xq = !(b0 | vv); int ns = ((s << 1) | xq) & (NS - 1);
    if (kind == 2) { if (mode == 0 || reset || !b0) out[ns] += m * (1 - eps); continue; }
    if (kind == 0 || !reset) out[ns] += m * (1 - eps);
    int n0 = ((s << 1) | 0) & (NS - 1), n1 = ((s << 1) | 1) & (NS - 1);
    if (b0) out[n0] += (kind == 0) ? m * eps : m * eps * (1 - phi);
    else { if (kind == 0) out[n0] += m * eps * plo; out[n1] += m * eps * (1 - phi); }
  }
}
int main(int argc, char **argv) {
  mode = atoi(argv[1]); eps = atof(argv[2]); plo = atof(argv[3]); phi = atof(argv[4]);
  if (plo <= 0 && phi <= 0) plo = phi = ((2 * eps + 1) - sqrt(4 * eps + 1)) / (2 * eps);
  p = atoi(argv[5]); W = 1; for (int i = 0; i < p; i++) { d[i] = atoi(argv[6 + i]); if (d[i] > W) W = d[i]; }
  NS = 1 << W;
  double *V[8], *Vn[8], *a[8], *tmp = malloc(8 * NS), *tmp2 = malloc(8 * NS);
  for (int f = 0; f < p; f++) { V[f] = calloc(NS, 8); Vn[f] = calloc(NS, 8); a[f] = calloc(NS, 8);
    for (int s = 0; s < NS; s++) V[f][s] = 1; }
  double mu = 0;
  for (int t = 0; t < 70; t++) {
    for (int f = 0; f < p; f++) { memset(a[f], 0, 8 * NS); if (d[f] < 2) continue;
      int src = ((f - d[f]) % p + p) % p; memcpy(tmp, V[src], 8 * NS);
      for (int k = d[f] - 1; k >= 1; k--) { int g = ((f - k) % p + p) % p; step(g, tmp, tmp2, 0); memcpy(tmp, tmp2, 8 * NS); }
      step(f, tmp, a[f], 2); }
    for (int f = 0; f < p; f++) memcpy(Vn[f], a[f], 8 * NS);
    for (int it = 0; it < 5000; it++) { double diff = 0;
      for (int f = 0; f < p; f++) { int g = (f - 1 + p) % p; step(f, Vn[g], tmp, 1);
        for (int s = 0; s < NS; s++) { double nv = a[f][s] + tmp[s]; diff += fabs(nv - Vn[f][s]); Vn[f][s] = nv; } }
      if (diff < 1e-12) break; }
    double sn = 0, so = 0; for (int f = 0; f < p; f++) for (int s = 0; s < NS; s++) { sn += Vn[f][s]; so += V[f][s]; }
    mu = sn / so; for (int f = 0; f < p; f++) for (int s = 0; s < NS; s++) V[f][s] = Vn[f][s] / sn;
  }
  printf("%.6f\n", mu); return 0;
}
