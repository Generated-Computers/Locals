// table_jacobi.c: rounds of the table-Jacobi iteration with reach R on the eps-smoothing of a
// random block base (blocks of m positions on top of zero gates; af = 1: no zero gates below).
// A gate whose pointer reaches at most R back reads the CURRENT iterate (in a real implementation
// one round is a parallel prefix over tables with 2^R entries; here it is simulated sequentially);
// every other gate reads the previous iterate.  R = 1 is the prefix-Jacobi iteration.
// Prints the rounds until the iterate is a fixed point (checked against the trace), the length of
// the greedy chain of the correct-prefix bound, and sqrt(eps*n).
// usage: table_jacobi n eps m af seed R maxrounds
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
static uint64_t s[2];
static inline uint64_t rnd(void){uint64_t a=s[0],b=s[1];s[0]=b;a^=a<<23;s[1]=a^b^(a>>17)^(b>>26);return s[1]+b;}
static inline double uni(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static inline uint32_t below(uint32_t k){return (uint32_t)(uni()*k);}
int main(int argc,char**argv){
  if(argc<8){fprintf(stderr,"usage: table_jacobi n eps m af seed R maxrounds\n");return 1;}
  uint32_t n=atoi(argv[1]);double eps=atof(argv[2]);uint32_t m=atoi(argv[3]);double af=atof(argv[4]);
  uint64_t seed=strtoull(argv[5],0,10);uint32_t R=atoi(argv[6]);int maxr=atoi(argv[7]);
  s[0]=seed*0x9E3779B97F4A7C15ULL+1;s[1]=seed^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();
  uint32_t n0=(uint32_t)((1.0-af)*n); if(n0<1)n0=1;
  uint32_t*j=malloc(4*(n+1));uint8_t*x=malloc(n+1),*y=calloc(n+1,1),*z=malloc(n+1);
  j[0]=0;
  for(uint32_t i=1;i<=n;i++){uint32_t b=0; if(i>=n0+m){uint32_t t=(i-n0)/m; b=n0+(t-1)*m+below(m);} j[i]=b;}
  for(uint32_t i=1;i<=n;i++) if(uni()<eps) j[i]=below(i);
  x[0]=1; for(uint32_t k=1;k<=n;k++) x[k]=!(x[k-1]|x[j[k]]);
  // greedy chain of the correct-prefix bound: c -> (least long gate g with j_g > c) - 1
  uint32_t c=0; int chain=0;
  while(c<n){ uint32_t g=c+1; while(g<=n && !(g-j[g]>R && j[g]>c)) g++; c=g-1; chain++; }
  y[0]=1; int rounds;
  for(rounds=1;rounds<=maxr;rounds++){ z[0]=1;
    for(uint32_t k=1;k<=n;k++){uint32_t t=j[k]; uint8_t r=(k-t<=R)?z[t]:y[t]; z[k]=!(z[k-1]|r);}
    if(memcmp(y,z,n+1)==0)break; uint8_t*w=y;y=z;z=w; }
  int ok=(memcmp(y,x,n+1)==0);
  printf("n=%u eps=%.2f m=%u af=%.2f R=%u: rounds=%d%s  greedy chain=%d  sqrt(eps n)=%.0f  trace %s\n",n,eps,m,af,R,rounds,rounds>maxr?"+":"",chain,sqrt(eps*n),ok?"ok":"NOT REACHED");
  return 0;}
