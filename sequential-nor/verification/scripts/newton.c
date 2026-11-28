// newton.c: rounds of the GF(2) Newton iteration of 001a (standard start) on eps-smoothed bases,
// next to the depths d and D.   usage: newton n eps type L seed cap
// types: 0 j=i-L; 1 distance uniform in [L,2L); 2 j=floor(i/2); 3 j=0 (filler); 4 log-uniform distance
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
  if(argc<7){fprintf(stderr,"usage\n");return 1;}
  uint32_t n=atoi(argv[1]);double eps=atof(argv[2]);int type=atoi(argv[3]);uint32_t L=atoi(argv[4]);
  uint64_t seed=strtoull(argv[5],0,10);int cap=atoi(argv[6]);
  s[0]=seed*0x9E3779B97F4A7C15ULL+1;s[1]=seed^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();
  uint32_t*j=malloc(4*(n+1));uint8_t*x=malloc(n+1),*y=malloc(n+1),*z=malloc(n+1);
  int32_t*d=malloc(4*(n+1)),*D=malloc(4*(n+1));
  j[0]=0;
  for(uint32_t i=1;i<=n;i++){uint32_t b=0;
    if(type==0)b=(i>L)?i-L:0;
    else if(type==1){uint32_t dist=L+below(L);b=(i>dist)?i-dist:0;}
    else if(type==2)b=i/2;
    else if(type==3)b=0;
    else if(type==4){double u=uni();uint32_t dist=(uint32_t)exp(u*log((double)i));if(dist<2)dist=2;b=(i>dist)?i-dist:0;}
    if(b>=i)b=i-1; j[i]=b;}
  for(uint32_t i=1;i<=n;i++) if(uni()<eps) j[i]=below(i);
  x[0]=1;d[0]=D[0]=0;int32_t md=0,mD=0;
  for(uint32_t k=1;k<=n;k++){uint32_t t=j[k];uint8_t xp=x[k-1],xj=x[t];x[k]=!(xp|xj);
    int notg=(t==k-1),reset=(!notg)&&xj,rel=!xp;
    if(k==1){d[k]=D[k]=1;continue;}
    if(reset)d[k]=1+d[t];else if(!notg)d[k]=d[k-1]>1+d[t]?d[k-1]:1+d[t];else d[k]=d[k-1];
    if(reset&&rel)D[k]=1+D[t];else if(reset)D[k]=D[k-1]<1+D[t]?D[k-1]:1+D[t];
    else if(rel&&!notg)D[k]=D[k-1]>1+D[t]?D[k-1]:1+D[t];else D[k]=D[k-1];
    if(d[k]>md)md=d[k];if(D[k]>mD)mD=D[k];}
  memset(y,0,n+1);y[0]=1;int rounds=0;uint32_t front=0;
  for(;;){uint32_t f=0;for(uint32_t i=1;i<=n;i++)if(y[i]!=x[i]){f=i;break;}
    if(!f)break; front=f; if(rounds>=cap){rounds=-1;break;}
    z[0]=1;
    for(uint32_t i=1;i<=n;i++){uint32_t t=j[i];
      if(t==i-1){z[i]=1^z[i-1];continue;}
      uint8_t a=y[i-1],b=y[t];
      z[i]=((1^a)&(1^b)) ^ ((1^b)&(z[i-1]^a)) ^ ((1^a)&(z[t]^b));}
    uint8_t*w=y;y=z;z=w;rounds++;}
  printf("n=%u eps=%.2f type=%d L=%u seed=%llu  d=%d D=%d newton=%d lastfront=%u\n",n,eps,type,L,(unsigned long long)seed,md,mD,rounds,front);
  return 0;}
