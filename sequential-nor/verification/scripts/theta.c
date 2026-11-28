// theta.c: growth rate of the number of chains that use only base (non-resampled) pointers,
// for an active region embedded in filler (pointer 0).  Counts for the Kleene-prefix depth d
// (every non-NOT gate of the window) and for Dlast (relevant children plus the last reset).
// usage: theta n eps type L seed T afrac
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
  if(argc<8){fprintf(stderr,"usage\n");return 1;}
  uint32_t n=atoi(argv[1]);double eps=atof(argv[2]);int type=atoi(argv[3]);uint32_t L=atoi(argv[4]);
  uint64_t seed=strtoull(argv[5],0,10);int T=atoi(argv[6]);double af=atof(argv[7]);
  s[0]=seed*0x9E3779B97F4A7C15ULL+1;s[1]=seed^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();
  uint32_t n0=(uint32_t)((1.0-af)*n); n0-=n0%L;
  uint32_t*j=malloc(4*(n+1));uint8_t*x=malloc(n+1),*adv=malloc(n+1);
  j[0]=0;
  for(uint32_t i=1;i<=n;i++){
    uint32_t b=0;
    if(i>=n0+64*L){
      if(type==1){uint32_t dist=L+below(L);b=i-dist;}
      else if(type==0){b=i-L;}
      else if(type==2){uint32_t off=i%L; if(off==0)b=0; else{uint32_t blk=i/L,back=8+below(56);uint32_t eo=2+2*below((L-1)/2);if(eo>=L)eo=2;b=(blk-back)*L+eo;}}
      else if(type==3){ // period L: offset 0 pinned, odd offsets read even offsets of lower blocks, even offsets are NOT gates
        uint32_t off=i%L; if(off==0)b=0; else if(off%2==0)b=i-1; else{uint32_t blk=i/L,back=8+below(56);uint32_t eo=2+2*below((L-1)/2);if(eo>=L)eo=2;b=(blk-back)*L+eo;}}
      else if(type==4){ // spread, but only read positions two apart from a random anchor shared by a pair of gates
        uint32_t dist=L+below(L);b=i-dist; b-=b%2;}
      if(b<n0&&b!=0)b=i-1;
    }
    j[i]=b;
  }
  for(uint32_t i=1;i<=n;i++){ if(uni()<eps){j[i]=below(i);adv[i]=0;} else adv[i]=1; }
  x[0]=1;uint64_t ones=1,onesA=0;
  for(uint32_t k=1;k<=n;k++){x[k]=!(x[k-1]|x[j[k]]);ones+=x[k];if(k>=n0)onesA+=x[k];}
  uint32_t m=n-n0+1;
  double*A=malloc(8*m),*B=malloc(8*m),*A2=malloc(8*m),*B2=malloc(8*m);
  for(uint32_t u=0;u<m;u++){A[u]=1;A2[u]=1;}
  printf("n=%u eps=%.3f type=%d L=%u dens=%.4f densA=%.4f\n",n,eps,type,L,(double)ones/(n+1),(double)onesA/m);
  double prev=1,prev2=1,m4=1,m4b=1;uint32_t lo=m/2;
  for(int t=1;t<=T;t++){
    B[0]=0;B2[0]=0;
    for(uint32_t u=1;u<m;u++){uint32_t k=n0+u,tg=j[k];int notg=(tg==k-1),reset=(!notg)&&x[tg],rel=!x[k-1];
      int child=adv[k]&&!notg&&tg>=n0;
      double c=child?A[tg-n0]:0, c2=child?A2[tg-n0]:0;
      B[u]=c+(reset?0:B[u-1]);
      if(reset)B2[u]=c2;else B2[u]=B2[u-1]+(rel?c2:0);
    }
    double sum=0,sum2=0;for(uint32_t u=lo;u<m;u++){sum+=B[u];sum2+=B2[u];}
    sum/=(m-lo);sum2/=(m-lo);
    if(t==4){m4=sum;m4b=sum2;} if(t==12){printf("rate over t=4..12:  KP %.4f   Dlast %.4f   (means at 12: %.3e %.3e)\n",pow(sum/m4,0.125),pow(sum2/m4b,0.125),sum,sum2);}
    prev=sum;prev2=sum2;
    double*w=A;A=B;B=w;w=A2;A2=B2;B2=w;
  }
  return 0;}
