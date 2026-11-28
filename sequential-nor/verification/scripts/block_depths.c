// block_depths.c: depths on the eps-smoothing of a random block base as in Section 10 (a simplified form of the bases of 001d).  Zero gates below n0=(1-af)n
// (af = 1: no filler).  Above, layers of m consecutive positions; a gate of layer t >= 1 has a base
// pointer chosen uniformly in layer t-1 (fixed before the noise); layer 0 consists of zero gates.
// Prints max d (Kleene-prefix rounds), max Dlast, max D (relevance-pruned rounds), number of layers.
// usage: block_depths n eps m af seed jacmax   (jacmax = 0: skip the prefix-Jacobi iteration)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static uint64_t s[2];
static inline uint64_t rnd(void){uint64_t a=s[0],b=s[1];s[0]=b;a^=a<<23;s[1]=a^b^(a>>17)^(b>>26);return s[1]+b;}
static inline double uni(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static inline uint32_t below(uint32_t k){return (uint32_t)(uni()*k);}
int main(int argc,char**argv){
  uint32_t n=atoi(argv[1]);double eps=atof(argv[2]);uint32_t m=atoi(argv[3]);double af=atof(argv[4]);
  uint64_t seed=strtoull(argv[5],0,10);int jac=argc>6?atoi(argv[6]):0;
  s[0]=seed*0x9E3779B97F4A7C15ULL+1;s[1]=seed^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();
  uint32_t n0=(uint32_t)((1.0-af)*n); if(n0<1)n0=1;
  uint32_t*j=malloc(4*(n+1));uint8_t*x=malloc(n+1);int32_t*d=malloc(4*(n+1)),*Dl=malloc(4*(n+1)),*D=malloc(4*(n+1));
  j[0]=0;
  for(uint32_t i=1;i<=n;i++){uint32_t b=0; if(i>=n0+m){uint32_t t=(i-n0)/m; b=n0+(t-1)*m+below(m);} j[i]=b;}
  for(uint32_t i=1;i<=n;i++) if(uni()<eps) j[i]=below(i);
  x[0]=1;d[0]=Dl[0]=D[0]=0;int32_t md=0,mDl=0,mD=0;uint64_t ones=1,onesA=0;
  for(uint32_t k=1;k<=n;k++){
    uint32_t t=j[k];uint8_t xp=x[k-1],xj=x[t];x[k]=!(xp|xj);ones+=x[k];if(k>=n0)onesA+=x[k];
    int notg=(t==k-1),reset=(!notg)&&xj,rel=!xp;
    if(k==1){d[k]=Dl[k]=D[k]=1;continue;}
    if(reset)d[k]=1+d[t];else if(!notg)d[k]=d[k-1]>1+d[t]?d[k-1]:1+d[t];else d[k]=d[k-1];
    if(reset)Dl[k]=1+Dl[t];else if(rel&&!notg)Dl[k]=Dl[k-1]>1+Dl[t]?Dl[k-1]:1+Dl[t];else Dl[k]=Dl[k-1];
    if(reset&&rel)D[k]=1+D[t];else if(reset)D[k]=D[k-1]<1+D[t]?D[k-1]:1+D[t];
    else if(rel&&!notg)D[k]=D[k-1]>1+D[t]?D[k-1]:1+D[t];else D[k]=D[k-1];
    if(d[k]>md)md=d[k];if(Dl[k]>mDl)mDl=Dl[k];if(D[k]>mD)mD=D[k];
  }
  int rounds=-1;
  if(jac){uint8_t*y=calloc(n+1,1),*z=malloc(n+1);y[0]=1;
    for(rounds=1;rounds<=jac;rounds++){z[0]=1;
      for(uint32_t k=1;k<=n;k++){uint32_t t=j[k];uint8_t r=(t==k-1)?z[k-1]:y[t];z[k]=!(z[k-1]|r);}
      if(memcmp(y,z,n+1)==0)break;uint8_t*w=y;y=z;z=w;}}
  printf("n=%u eps=%.3f m=%u af=%.2f layers=%u dens=%.4f densA=%.4f  d=%d Dlast=%d D=%d jacobi=%d\n",n,eps,m,af,(n-n0)/m,(double)ones/(n+1),(double)onesA/(n-n0+1),md,mDl,mD,rounds);
  return 0;}
