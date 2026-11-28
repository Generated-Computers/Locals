/* embed2.c: simulator for the refined NOR-tree embedding (robust zero, pair leaves, double top)
   with zero padding.  usage: embed2 s m d rho eps seed
   Prints the density of ones over cell gates (min/max of the prefix density), and for each bundle
   value the largest fraction of bad cells. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static uint64_t S0, S1;
static inline uint64_t rnd(void){uint64_t a=S0,b=S1;S0=b;a^=a<<23;S1=a^b^(a>>17)^(b>>26);return S1+b;}
static inline double uni(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static uint32_t *J; static uint8_t *isCell; static uint32_t pos;
static int s, m, d, rho; static int *U, *V; static uint8_t *val; static uint32_t *O; /* O[t*m+l] */
static uint32_t copyOf(int w){ if (w == -1) return 0; if (w == -2) return 1; return O[w*m + (int)(uni()*m)]; }
static uint32_t put(uint32_t ptr){ J[pos] = ptr; isCell[pos] = 1; return pos++; }
static uint32_t block(int h, int w){
  if (h == 0){
    put(0); put(1); put(0);                     /* robust zero: z0, g (reads position 1), z */
    uint32_t p2 = copyOf(w), p1 = copyOf(w);
    put(p2); put(1); return put(p1);            /* l1 -> p2, l2 -> 1, l3 -> p1 */
  }
  uint32_t a = block(h-1, w); block(h-1, w); return put(a);
}
int main(int argc, char **argv){
  if (argc < 7){ fprintf(stderr, "usage: embed2 s m d rho eps seed\n"); return 1; }
  s = atoi(argv[1]); m = atoi(argv[2]); d = atoi(argv[3]); rho = atoi(argv[4]);
  double eps = atof(argv[5]); uint64_t seed = strtoull(argv[6], 0, 10);
  S0 = seed*0x9E3779B97F4A7C15ULL + 7; S1 = seed ^ 0xD1B54A32D192ED03ULL; for (int i = 0; i < 20; i++) rnd();
  uint64_t T = 7ULL*(1ULL<<d) - 1, c = 2*T + 3, pre = 2000;
  uint64_t n = pre + (uint64_t)(1+rho)*c*s*m;
  J = malloc(4*(n+1)); isCell = calloc(n+1, 1); uint8_t *x = malloc(n+1);
  U = malloc(sizeof(int)*s); V = malloc(sizeof(int)*s); val = malloc(s); O = malloc(4*(size_t)s*m);
  for (int t = 0; t < s; t++){            /* random NOR circuit; -1 = constant 1, -2 = constant 0 */
    U[t] = t < 2 ? -1 - (int)(uni()*2) : (int)(uni()*t); V[t] = t < 2 ? -1 - (int)(uni()*2) : (int)(uni()*t);
    if (t >= 2 && uni() < 0.15) V[t] = -1 - (int)(uni()*2);
    int a = U[t] == -1 ? 1 : U[t] == -2 ? 0 : val[U[t]], b = V[t] == -1 ? 1 : V[t] == -2 ? 0 : val[V[t]];
    val[t] = !(a | b);
  }
  J[0] = 0; pos = 1; while (pos <= pre){ J[pos] = (pos == 1) ? 0 : 0; pos++; }
  for (int t = 0; t < s; t++) for (int l = 0; l < m; l++){
    for (uint64_t k = 0; k < (uint64_t)rho*c; k++){ J[pos] = 0; pos++; }
    uint32_t lu = block(d, U[t]); block(d, V[t]);
    put(lu); put(1); O[t*m + l] = put(lu);    /* double top: tau1 -> L_u, tau2 -> 1, tau3 -> L_u */
  }
  if (pos != n+1){ fprintf(stderr, "layout size mismatch %u %llu\n", pos, (unsigned long long)n+1); return 2; }
  x[0] = 1; uint64_t ones = 1; double dmin = 1, dmax = 0;
  for (uint64_t i = 1; i <= n; i++){
    uint32_t p = J[i]; if (i >= 2 && uni() < eps) p = (uint32_t)(uni()*i);
    x[i] = !(x[i-1] | x[p]);
    if (isCell[i] && i > 100000){ double psi = (double)ones/i; if (psi < dmin) dmin = psi; if (psi > dmax) dmax = psi; }
    ones += x[i];
  }
  double worst[2] = {0, 0}; int worstt[2] = {-1, -1}; double mean[2] = {0, 0}; int cnt[2] = {0, 0};
  for (int t = 0; t < s; t++){
    int bad = 0; for (int l = 0; l < m; l++) bad += (x[O[t*m+l]] != val[t]);
    double f = (double)bad/m; int b = val[t]; mean[b] += f; cnt[b]++;
    if (f > worst[b]){ worst[b] = f; worstt[b] = t; }
  }
  printf("s=%d m=%d d=%d rho=%d eps=%.3f n=%llu  density over cell gates in [%.4f, %.4f]  "
         "bad fraction: value 0 mean %.4f max %.4f | value 1 mean %.4f max %.4f | output %s\n",
         s, m, d, rho, eps, (unsigned long long)n, dmin, dmax,
         cnt[0] ? mean[0]/cnt[0] : 0, worst[0], cnt[1] ? mean[1]/cnt[1] : 0, worst[1],
         (double)0 == 0 ? (({int bad=0; for(int l=0;l<m;l++) bad += (x[O[(s-1)*m+l]] != val[s-1]); bad*2 < m;}) ? "majority correct" : "MAJORITY WRONG") : "");
  return 0;
}
