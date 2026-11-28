// parity_counts.c: SAMPLED numbers of base-pointer chains (Kleene-prefix count and Dlast count) on
// the eps-smoothing of the base with zero gates below (1-af)n and j_i = i-de (i even), i-do (i odd)
// above.  Prints, for every length t <= T, the mean number of chains of length t over the upper
// half of the active part, and the growth rate between lengths 4 and T.
// usage: parity_counts n eps de do seed T af      e.g.  ./parity_counts 8000000 0.70 2 3 1 6 0.1
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
static uint64_t s[2];
static inline uint64_t rnd(void){uint64_t a=s[0],b=s[1];s[0]=b;a^=a<<23;s[1]=a^b^(a>>17)^(b>>26);return s[1]+b;}
static inline double uni(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static inline uint32_t below(uint32_t k){return (uint32_t)(uni()*k);}
int main(int argc,char**argv){
  if(argc<8){fprintf(stderr,"usage: parity_counts n eps de do seed T af\n");return 1;}
  uint32_t n=atoi(argv[1]);double eps=atof(argv[2]);int de=atoi(argv[3]),dd=atoi(argv[4]);
  uint64_t seed=strtoull(argv[5],0,10);int T=atoi(argv[6]);double af=atof(argv[7]);
  s[0]=seed*0x9E3779B97F4A7C15ULL+1;s[1]=seed^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();
  uint32_t n0=(uint32_t)((1.0-af)*n);
  uint32_t*j=malloc(4*(n+1));uint8_t*x=malloc(n+1),*adv=malloc(n+1);
  j[0]=0;
  for(uint32_t i=1;i<=n;i++){ uint32_t b=0; if(i>=n0+64){int d=(i%2==0)?de:dd; b=i-d;} j[i]=b; }
  for(uint32_t i=1;i<=n;i++){ if(uni()<eps){j[i]=below(i);adv[i]=0;} else adv[i]=1; }
  x[0]=1;uint64_t ones=1;
  for(uint32_t k=1;k<=n;k++){x[k]=!(x[k-1]|x[j[k]]);ones+=x[k];}
  uint32_t m=n-n0+1;
  double*A=malloc(8*m),*B=malloc(8*m),*A2=malloc(8*m),*B2=malloc(8*m);
  for(uint32_t u=0;u<m;u++){A[u]=1;A2[u]=1;}
  printf("n=%u eps=%.3f base (%d,%d) density of ones %.4f\n",n,eps,de,dd,(double)ones/(n+1));
  double m4=1,m4b=1;uint32_t lo=m/2;
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
    printf("length %d: mean Kleene-prefix count %.5f, mean Dlast count %.5f\n",t,sum,sum2);
    if(t==4){m4=sum;m4b=sum2;}
    if(t==T&&T>4)printf("rate over lengths 4..%d: Kleene-prefix %.4f, Dlast %.4f\n",T,pow(sum/m4,1.0/(T-4)),pow(sum2/m4b,1.0/(T-4)));
    double*w=A;A=B;B=w;w=A2;A2=B2;B2=w;
  }
  return 0;}
