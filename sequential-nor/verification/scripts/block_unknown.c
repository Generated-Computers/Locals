// block_unknown.c: on the eps-smoothing of a random block base over zero gates (as block_depths.c),
// the fraction of positions of the upper blocks that the relevance-pruned iteration has not
// determined after s rounds, i.e. with D(i) > s, split by the value they hold.
// These fractions are compared in the note with the iterates of the map Psi of the section on
// block bases (Proposition "the map of the unknown fractions").
// usage: block_unknown n eps m af seed tlo s1 s2 ...   (blocks tlo..last are counted; af as in block_depths)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static uint64_t s[2];
static inline uint64_t rnd(void){uint64_t a=s[0],b=s[1];s[0]=b;a^=a<<23;s[1]=a^b^(a>>17)^(b>>26);return s[1]+b;}
static inline double uni(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static inline uint32_t below(uint32_t k){return (uint32_t)(uni()*k);}
int main(int argc,char**argv){
  if(argc<8){fprintf(stderr,"usage: block_unknown n eps m af seed tlo s1 s2 ...\n");return 1;}
  uint32_t n=atoi(argv[1]);double eps=atof(argv[2]);uint32_t m=atoi(argv[3]);double af=atof(argv[4]);
  uint64_t seed=strtoull(argv[5],0,10);uint32_t tlo=atoi(argv[6]);
  s[0]=seed*0x9E3779B97F4A7C15ULL+1;s[1]=seed^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();
  uint32_t n0=(uint32_t)((1.0-af)*n); if(n0<1)n0=1;
  uint32_t*j=malloc(4*(size_t)(n+1));uint8_t*x=malloc(n+1);int32_t*D=malloc(4*(size_t)(n+1));
  j[0]=0;
  for(uint32_t i=1;i<=n;i++){uint32_t b=0; if(i>=n0+m){uint32_t t=(i-n0)/m; b=n0+(t-1)*m+below(m);} j[i]=b;}
  for(uint32_t i=1;i<=n;i++) if(uni()<eps) j[i]=below(i);
  x[0]=1;D[0]=0;int32_t mD=0;uint64_t ones=1;
  for(uint32_t k=1;k<=n;k++){
    uint32_t t=j[k];uint8_t xp=x[k-1],xj=x[t];x[k]=!(xp|xj);ones+=x[k];
    int notg=(t==k-1),reset=(!notg)&&xj,rel=!xp;
    if(k==1){D[k]=1;continue;}
    if(reset&&rel)D[k]=1+D[t];else if(reset)D[k]=D[k-1]<1+D[t]?D[k-1]:1+D[t];
    else if(rel&&!notg)D[k]=D[k-1]>1+D[t]?D[k-1]:1+D[t];else D[k]=D[k-1];
    if(D[k]>mD)mD=D[k];
  }
  uint32_t layers=(n-n0)/m, first=n0+tlo*m, last=n0+layers*m-1;
  uint64_t tot=last-first+1, o=0; for(uint32_t i=first;i<=last;i++)o+=x[i];
  printf("n=%u eps=%.3f m=%u layers=%u density=%.4f density in the counted blocks=%.4f max D=%d\n",n,eps,m,layers,(double)ones/(n+1),(double)o/tot,mD);
  for(int a=7;a<argc;a++){int32_t r=atoi(argv[a]);uint64_t u1=0,u0=0;
    for(uint32_t i=first;i<=last;i++) if(D[i]>r){if(x[i])u1++;else u0++;}
    printf("  rounds %4d: undetermined fraction %.4f (holding 1: %.4f, holding 0: %.4f)\n",r,(double)(u1+u0)/tot,(double)u1/tot,(double)u0/tot);}
  return 0;}
