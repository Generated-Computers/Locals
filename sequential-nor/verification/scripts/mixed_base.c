// mixed_base.c: a layered base that mixes one short pointer with pointers to the layer below.
// Zero gates below n0 = (1-af)n. Above, layers of m consecutive positions (m even). In a layer
// t >= 1 the gates at even offsets read the position 2 before them (j_i = i-2), and the gates at
// odd offsets read a uniformly random even-offset position of layer t-1 (fixed before the noise).
// Layer 0 consists of zero gates. With mode = 0 every gate of a layer t >= 1 reads a uniformly
// random position of layer t-1 instead (the block bases of the note, for comparison).
// Prints, for one eps-smoothing: max D (the number of rounds of the relevance-pruned iteration),
// the number of rounds of the prefix-Jacobi iteration, and of the table-Jacobi iteration with
// reach R (pointers of reach at most R read the current iterate).
// usage: mixed_base n eps m af seed mode R maxrounds      (maxrounds = 0: skip the two iterations)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static uint64_t s[2];
static inline uint64_t rnd(void){uint64_t a=s[0],b=s[1];s[0]=b;a^=a<<23;s[1]=a^b^(a>>17)^(b>>26);return s[1]+b;}
static inline double uni(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static inline uint32_t below(uint32_t k){return (uint32_t)(uni()*k);}
static uint32_t n,*j; static uint8_t *x,*y,*z;
static int jacobi(uint32_t R,int maxr){
  memset(y,0,n+1);y[0]=1;int rounds;
  for(rounds=1;rounds<=maxr;rounds++){z[0]=1;
    for(uint32_t k=1;k<=n;k++){uint32_t t=j[k];uint8_t r=(k-t<=R)?z[t]:y[t];z[k]=!(z[k-1]|r);}
    if(memcmp(y,z,n+1)==0)break;uint8_t*w=y;y=z;z=w;}
  if(rounds<=maxr&&memcmp(y,x,n+1)!=0){fprintf(stderr,"wrong fixed point\n");exit(1);}
  return rounds;
}
int main(int argc,char**argv){
  if(argc<9){fprintf(stderr,"usage: mixed_base n eps m af seed mode R maxrounds\n");return 1;}
  n=atoi(argv[1]);double eps=atof(argv[2]);uint32_t m=atoi(argv[3]);double af=atof(argv[4]);
  uint64_t seed=strtoull(argv[5],0,10);int mode=atoi(argv[6]);uint32_t R=atoi(argv[7]);int maxr=atoi(argv[8]);
  m-=m%2; s[0]=seed*0x9E3779B97F4A7C15ULL+1;s[1]=seed^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();
  uint32_t n0=(uint32_t)((1.0-af)*n); if(n0<1)n0=1;
  j=malloc(4*(size_t)(n+1));x=malloc(n+1);y=malloc(n+1);z=malloc(n+1);int32_t*D=malloc(4*(size_t)(n+1));
  j[0]=0;
  for(uint32_t i=1;i<=n;i++){uint32_t b=0;
    if(i>n0+m){uint32_t off=i-n0-1,t=off/m,o=off%m; uint32_t base=n0+1+(t-1)*m;
      if(mode==0) b=base+below(m); else if(o%2==0) b=i-2; else b=base+2*below(m/2);}
    j[i]=b;}
  for(uint32_t i=1;i<=n;i++) if(uni()<eps) j[i]=below(i);
  x[0]=1;D[0]=0;int32_t mD=0;uint64_t ones=1;
  for(uint32_t k=1;k<=n;k++){uint32_t t=j[k];uint8_t xp=x[k-1],xj=x[t];x[k]=!(xp|xj);ones+=x[k];
    int notg=(t==k-1),reset=(!notg)&&xj,rel=!xp;
    if(k==1){D[k]=1;continue;}
    if(reset&&rel)D[k]=1+D[t];else if(reset)D[k]=D[k-1]<1+D[t]?D[k-1]:1+D[t];
    else if(rel&&!notg)D[k]=D[k-1]>1+D[t]?D[k-1]:1+D[t];else D[k]=D[k-1];
    if(D[k]>mD)mD=D[k];}
  int r1=-1,r2=-1; if(maxr>0){r1=jacobi(1,maxr);r2=jacobi(R,maxr);}
  printf("n=%u eps=%.3f m=%u layers=%u mode=%d density=%.4f | max D=%d | prefix-Jacobi rounds=%d | table-Jacobi (reach %u) rounds=%d\n",
    n,eps,m,(n-n0)/m-1,mode,(double)ones/(n+1),mD,r1,R,r2);
  return 0;}
