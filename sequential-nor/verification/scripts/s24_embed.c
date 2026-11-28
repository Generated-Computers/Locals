// s24_embed.c: builds Prog(C) of Section "Above 4/9" for a random NOR circuit C with s gates,
// smooths it at noise eps, evaluates the trace, and decodes every wire from the phase-8 rail of
// its stage J (vote: fraction of ones above thr means value 1). Small parameters only.
// Gate stage: a read of phase k draws from rail k of stage J of u or of v with probability (1-beta)/2
// each, and with probability beta lists position 0 (k even) or 1 (k odd). Set VERBOSE to list gates.
// usage: s24_embed s eps h J rho m beta thr pre seed
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static uint64_t S0,S1;
static inline uint64_t rnd(void){uint64_t a=S0,b=S1;S0=b;a^=a<<23;S1=a^b^(a>>17)^(b>>26);return S1+b;}
static inline double uni(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static inline uint32_t below(uint32_t k){return (uint32_t)(uni()*k);}
static const int KAPPA[24]={22,9,22,9,22,9,22,9,22,9,-1,3,8,9,8,9,8,9,8,11,8,11,8,9};
static int h,J,rho,m; static uint32_t *jp; static uint32_t pos;
// bundle start positions: chain c of bundle starts at bstart + c*24h(1+rho) + 24h*rho
static uint32_t copy_pos(uint32_t bstart,uint32_t chain,int phase){
  return bstart+chain*(uint32_t)(24*h*(1+rho))+(uint32_t)(24*h*rho)+(uint32_t)(24*(h-1)+phase); }
int main(int argc,char**argv){
  if(argc<11){fprintf(stderr,"usage: s24_embed s eps h J rho m beta thr pre seed\n");return 1;}
  int s=atoi(argv[1]);double eps=atof(argv[2]);h=atoi(argv[3]);J=atoi(argv[4]);rho=atoi(argv[5]);m=atoi(argv[6]);
  double beta=atof(argv[7]),thr=atof(argv[8]);uint32_t pre=atoi(argv[9]);uint64_t seed=strtoull(argv[10],0,10);
  S0=seed*0x9E3779B97F4A7C15ULL+1;S1=seed^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();
  // circuit: wires 0=c0, 1=c1, 2..s+1 gates
  int W=s+2; int *u=malloc(sizeof(int)*W),*v=malloc(sizeof(int)*W),*val=malloc(sizeof(int)*W);
  val[0]=0;val[1]=1;
  for(int t=2;t<W;t++){u[t]=below(t);v[t]=below(t);val[t]=!(val[u[t]]|val[v[t]]);}
  uint64_t bundle=(uint64_t)24*h*(1+rho)*m; uint64_t nb=(uint64_t)(J+1)*s+2*J;
  uint64_t n64=pre+bundle*nb; if(n64>400000000ULL){fprintf(stderr,"too large %llu\n",(unsigned long long)n64);return 1;}
  uint32_t n=(uint32_t)n64;
  jp=malloc(4*(size_t)(n+1)); uint8_t*x=malloc(n+1);
  // bundle start of (wire, stage); constants have stages 1..J only
  uint32_t *bs=malloc(sizeof(uint32_t)*W*(J+1));
  pos=pre+1;
  for(int w=0;w<2;w++)for(int k=1;k<=J;k++){bs[w*(J+1)+k]=pos;pos+=bundle;}
  for(int w=2;w<W;w++)for(int k=0;k<=J;k++){bs[w*(J+1)+k]=pos;pos+=bundle;}
  for(uint32_t i=0;i<=n;i++) jp[i]=0;
  // fill chain pointers
  for(int w=0;w<W;w++) for(int k=(w<2?1:0);k<=J;k++){
    uint32_t b=bs[w*(J+1)+k];
    for(int c=0;c<m;c++){ uint32_t g0=b+c*(uint32_t)(24*h*(1+rho))+(uint32_t)(24*h*rho);
      for(int r=0;r<24*h;r++){ int kap=KAPPA[r%24]; uint32_t g=g0+r; uint32_t t;
        if(kap<0) t=1;
        else if(k>=1){ int src=w; 
          if(w<2 && k==1){ // stage 0 of a constant: constant rails (top for c1, bottom for c0)
            int one=(w==1)?(kap%2==0):(kap%2==1); t=one?0:1; }
          else t=copy_pos(bs[src*(J+1)+k-1],below(m),kap); }
        else { double z=uni(); int src;
          if(z<beta){ t=(kap%2==0)?0:1; }
          else { src=(z<beta+(1-beta)/2)?u[w]:v[w];
            t=copy_pos(bs[src*(J+1)+J],below(m),kap); } }
        jp[g]=t; } } }
  for(uint32_t i=1;i<=n;i++) if(uni()<eps) jp[i]=below(i);
  x[0]=1; uint64_t ones=1; for(uint32_t i=1;i<=n;i++){x[i]=!(x[i-1]|x[jp[i]]);ones+=x[i];}
  int32_t *D=NULL; int32_t mD=0;
  if(getenv("DEPTH")){ D=malloc(4*(size_t)(n+1)); D[0]=0;
    for(uint32_t k=1;k<=n;k++){uint32_t t=jp[k];uint8_t xp=x[k-1],xj=x[t];
      int notg=(t==k-1),reset=(!notg)&&xj,rel=!xp; int32_t v;
      if(k==1) v=1;
      else if(reset&&rel) v=1+D[t];
      else if(reset) v=(D[k-1]<1+D[t])?D[k-1]:1+D[t];
      else if(rel&&!notg) v=(D[k-1]>1+D[t])?D[k-1]:1+D[t];
      else v=D[k-1];
      D[k]=v; if(v>mD)mD=v;}
    printf("max D (rounds of the relevance-pruned iteration) = %d\n",mD); }
  int wrong=0; double minone=1,maxzero=0;
  for(int w=2;w<W;w++){ uint32_t b=bs[w*(J+1)+J]; int cnt=0;
    for(int c=0;c<m;c++) cnt+=x[copy_pos(b,c,8)];
    double f=(double)cnt/m; int dec=f>thr; if(dec!=val[w])wrong++;
    if(getenv("VERBOSE")) printf("  gate %d = NOR(%d,%d): inputs %d%d value %d  dens8 %.3f %s\n",w,u[w],v[w],val[u[w]],val[v[w]],val[w],f,dec!=val[w]?"WRONG":"");
    if(val[w]) {if(f<minone)minone=f;} else {if(f>maxzero)maxzero=f;} }
  // also report the stage-J densities of the constants
  printf("s=%d eps=%.3f h=%d J=%d rho=%d m=%d beta=%.4f n=%u density=%.4f | wrong gates %d of %d | value-1 rails min %.3f, value-0 rails max %.3f\n",
    s,eps,h,J,rho,m,beta,n,(double)ones/(n+1),wrong,s,minone,maxzero);
  return 0;}
