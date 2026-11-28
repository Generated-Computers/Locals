/* The latch embedding Latch(C) of a NOR circuit (Definition "Latch(C)" of the note), simulated
   under eps-smoothing.
   usage: latch_embed s m h J rho eps theta seed [verbose]
     s circuit gates, m copies per rail, h pairs per chain, J restoring stages, rho padding ratio,
     theta the thinning of the gate stage.
   Layout: position 0 holds 1; positions 1..pre are zero gates (pre = 20000); then the rails of
   stages 1..J of the constants c0 and c1; then, for every gate t, the rails of stages 0..J, in the
   order P_0, N_0, P_1, N_1, ...  A rail is m blocks; a block is rho*2h zero gates followed by a
   chain of 2h gates (h pairs: set check, reset check); the copy is the last gate of the chain.
   Stage 0 of a constant is exact (position 0 or 1). Stage 0 of g = NOR(u,v): P_0 sets from
   theta*(N_J(u)|N_J(v)) and resets from (P_J(u)|P_J(v)); N_0 the other way round. Stage k >= 1 of
   every wire: a rail sets from the same rail of stage k-1 and resets from the other rail.
   The circuit: gate 1 = NOR(c0,c0); for t >= 2 the first input is gate t-1 (with probability 0.7)
   or a random earlier gate, the second is c0 (0.25), c1 (0.05) or a random earlier gate.
   The trace is stored as bits (n/8 bytes). */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static uint64_t st[4];
static inline uint64_t rotl(uint64_t x,int k){return (x<<k)|(x>>(64-k));}
static inline uint64_t rnd(void){uint64_t r=rotl(st[1]*5,7)*9,t=st[1]<<17;st[2]^=st[0];st[3]^=st[1];st[1]^=st[2];st[0]^=st[3];st[2]^=t;st[3]=rotl(st[3],45);return r;}
static inline double unif(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static inline uint64_t below(uint64_t g){return (uint64_t)(((__uint128_t)rnd()*g)>>64);}
static uint8_t *bits;
static inline int get(uint64_t i){return (bits[i>>3]>>(i&7))&1;}
static inline void set1(uint64_t i){bits[i>>3]|=(uint8_t)(1u<<(i&7));}
static int s,m,h,J,rho; static double eps,theta; static uint64_t blk,pre,preC;
/* a wire is -1 (constant 1), 0 (constant 0) or a gate t >= 1; rail 0 = P, rail 1 = N */
static uint64_t railbase(int w,int k,int rail){
  if(w<=0){int c=(w==-1); return pre+((uint64_t)(c*J+(k-1))*2+rail)*m*blk;}
  return preC+(((uint64_t)(w-1)*(J+1)+k)*2+rail)*m*blk;
}
/* a position drawn from rail `rail` of wire w at stage k */
static uint64_t draw(int w,int k,int rail){
  if(w<=0&&k==0){int one=(w==-1)?(rail==0):(rail==1); return one?0:1;}
  return railbase(w,k,rail)+(below(m)+1)*blk;
}
static uint64_t g; static int prev; static uint64_t ones;
static inline void zero_gate(void){int rd; if(unif()<eps) rd=get(below(g)); else rd=1; int x=!(prev|rd); if(x){set1(g);ones++;} prev=x; g++;}
static inline void chain_gate(uint64_t tgt){if(unif()<eps) tgt=below(g); int x=!(prev|get(tgt)); if(x){set1(g);ones++;} prev=x; g++;}
/* one rail of stage k of wire w; for a gate stage (k == 0) the inputs are u and v */
static double build_rail(int w,int k,int rail,int u,int v){
  uint64_t cnt=0;
  if(g!=railbase(w,k,rail)+1){fprintf(stderr,"layout error\n");exit(1);}
  for(int l=0;l<m;l++){
    for(int i=0;i<2*h*rho;i++) zero_gate();
    for(int i=0;i<2*h;i++){
      int isset=(i%2==0); uint64_t tgt;
      if(k==0){
        int wantneg=(rail==0)?isset:!isset;      /* this check reads theta*(N(u)|N(v)); the other (P(u)|P(v)) */
        int in=(rnd()&1)?u:v;
        if(wantneg) tgt=(unif()<theta)?draw(in,J,1):1; else tgt=draw(in,J,0);
      }else tgt=draw(w,k-1,isset?rail:1-rail);
      chain_gate(tgt);
    }
    cnt+=prev;
  }
  return (double)cnt/m;
}
int main(int argc,char**argv){
  if(argc<9){fprintf(stderr,"usage: latch_embed s m h J rho eps theta seed [verbose]\n");return 1;}
  s=atoi(argv[1]);m=atoi(argv[2]);h=atoi(argv[3]);J=atoi(argv[4]);rho=atoi(argv[5]);
  eps=atof(argv[6]);theta=atof(argv[7]);uint64_t seed=strtoull(argv[8],0,10);int verbose=argc>9?atoi(argv[9]):0;
  uint64_t z=seed*0x9E3779B97F4A7C15ULL+1;for(int i=0;i<4;i++){z+=0x9E3779B97F4A7C15ULL;uint64_t x=z;x=(x^(x>>30))*0xBF58476D1CE4E5B9ULL;x=(x^(x>>27))*0x94D049BB133111EBULL;st[i]=x^(x>>31);}
  blk=(uint64_t)(1+rho)*2*h; pre=20000; preC=pre+(uint64_t)4*J*m*blk;
  uint64_t n=preC+(uint64_t)s*(J+1)*2*m*blk;
  if(n>6000000000ULL){fprintf(stderr,"n=%llu too large\n",(unsigned long long)n);return 1;}
  bits=calloc((n>>3)+2,1);if(!bits){fprintf(stderr,"no memory\n");return 1;}
  int *u=malloc((s+1)*sizeof(int)),*v=malloc((s+1)*sizeof(int)),*val=malloc((s+1)*sizeof(int));
  for(int t=1;t<=s;t++){
    if(t==1){u[t]=0;v[t]=0;}
    else{ u[t]=t-1; double q=unif(); v[t]= q<0.25?0 : q<0.30?-1 : 1+(int)below(t-1); if(unif()<0.3)u[t]=1+(int)below(t-1);}
    int a=u[t]==-1?1:u[t]==0?0:val[u[t]], b=v[t]==-1?1:v[t]==0?0:val[v[t]]; val[t]=!(a|b);
  }
  set1(0); ones=1; prev=1; g=1;
  while(g<=pre) zero_gate();
  double chi=0,clo=0;
  for(int c=0;c<2;c++){int w=c?-1:0; for(int k=1;k<=J;k++){double p=build_rail(w,k,0,0,0),q=build_rail(w,k,1,0,0); if(k==J){chi+=(c?p:q)/2; clo+=(c?q:p)/2;}}}
  double *fr=malloc((size_t)(s+1)*(J+1)*2*sizeof(double));
  for(int t=1;t<=s;t++)for(int k=0;k<=J;k++)for(int rail=0;rail<2;rail++)
    fr[(((size_t)t*(J+1))+k)*2+rail]=build_rail(t,k,rail,u[t],v[t]);
  int bad=0,n1=0; double minsep=1,sumhi=0,sumlo=0;
  for(int t=1;t<=s;t++){
    double a=fr[(((size_t)t*(J+1))+J)*2+0], b=fr[(((size_t)t*(J+1))+J)*2+1];
    double hi=val[t]?a:b, lo=val[t]?b:a; sumhi+=hi;sumlo+=lo; n1+=val[t]; if(hi-lo<minsep)minsep=hi-lo; if(hi<=lo)bad++;
    if(verbose>=2||(verbose==1&&t>s-3)){printf("gate %d = NOR(%d,%d) val %d:",t,u[t],v[t],val[t]);for(int k=0;k<=J;k+=(J>12?J/12:1))printf(" [%d] %.3f/%.3f",k,fr[(((size_t)t*(J+1))+k)*2+(val[t]?0:1)],fr[(((size_t)t*(J+1))+k)*2+(val[t]?1:0)]);printf(" [J] %.3f/%.3f\n",hi,lo);}
  }
  printf("s=%d m=%d h=%d J=%d rho=%d eps=%.4f theta=%.4f seed=%llu n=%llu density=%.4f | constants %.3f/%.3f | gates with value 1: %d | wrong gates: %d | mean high %.4f low %.4f | min separation %.4f\n",
    s,m,h,J,rho,eps,theta,(unsigned long long)seed,(unsigned long long)n,(double)ones/(n+1),chi,clo,n1,bad,sumhi/s,sumlo/s,minsep);
  return 0;
}
