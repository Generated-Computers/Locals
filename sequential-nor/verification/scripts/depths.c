// depths.c: Kleene-prefix depth d, last-anchor depth Dlast, pruned depth D, and
// rounds of the two-valued prefix-Jacobi iteration, on eps-smoothed oblivious bases.
// usage: depths n eps type L seed jac
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static uint64_t s[2];
static inline uint64_t rnd(void){uint64_t a=s[0],b=s[1];s[0]=b;a^=a<<23;s[1]=a^b^(a>>17)^(b>>26);return s[1]+b;}
static inline double uni(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static inline uint32_t below(uint32_t k){return (uint32_t)(uni()*k);}
int main(int argc,char**argv){
  if(argc<7){fprintf(stderr,"usage\n");return 1;}
  uint32_t n=atoi(argv[1]);double eps=atof(argv[2]);int type=atoi(argv[3]);uint32_t L=atoi(argv[4]);
  uint64_t seed=strtoull(argv[5],0,10);int jac=atoi(argv[6]);
  s[0]=seed*0x9E3779B97F4A7C15ULL+1;s[1]=seed^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();
  uint32_t*j=malloc(4*(n+1));uint8_t*x=malloc(n+1);int32_t*d=malloc(4*(n+1)),*Dl=malloc(4*(n+1)),*D=malloc(4*(n+1));
  // base
  j[0]=0;
  for(uint32_t i=1;i<=n;i++){
    uint32_t b;
    if(type==0){b=(i>L)?i-L:0;}
    else if(type==1){uint32_t dist=L+below(L);b=(i>dist)?i-dist:0;}
    else if(type==2){ // blocks of period L: offset 0 -> pointer 0; others read an even offset>=2 of a block far below
      uint32_t off=i%L; if(off==0||i<64*L){b=(off==0)?0:i-1;}
      else{uint32_t blk=i/L; uint32_t back=8+below(56); uint32_t tb=(blk-back)*L; uint32_t eo=2+2*below((L-1)/2); if(eo>=L)eo=2; b=tb+eo;}
    }
    else if(type==3){double u=uni();uint32_t dist=(uint32_t)(2.0+ (double)(i>4?i-2:2)*u*u*u*u); if(dist<2)dist=2; b=(i>dist)?i-dist:0;}
    else {b=i-1;}
    if(b>=i)b=i-1;
    j[i]=b;
  }
  // smoothing
  for(uint32_t i=1;i<=n;i++) if(uni()<eps) j[i]=below(i);
  x[0]=1;d[0]=Dl[0]=D[0]=0;
  int32_t md=0,mDl=0,mD=0;uint64_t ones=1;
  for(uint32_t k=1;k<=n;k++){
    uint32_t t=j[k];uint8_t xp=x[k-1],xj=x[t];x[k]=!(xp|xj);ones+=x[k];
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
    for(rounds=1;rounds<=jac;rounds++){z[0]=1;int same=1;
      for(uint32_t k=1;k<=n;k++){uint32_t t=j[k];uint8_t r=(t==k-1)?z[k-1]:y[t];z[k]=!(z[k-1]|r);}
      if(memcmp(y,z,n+1)==0)break;uint8_t*w=y;y=z;z=w;(void)same;}
    if(memcmp(y,x,n+1)!=0&&rounds<=jac){fprintf(stderr,"fixed point differs from trace\n");return 2;}
  }
  printf("n=%u eps=%.3f type=%d L=%u seed=%llu ones=%.4f d=%d Dlast=%d D=%d jacobi=%d\n",n,eps,type,L,(unsigned long long)seed,(double)ones/(n+1),md,mDl,mD,rounds);
  return 0;}
