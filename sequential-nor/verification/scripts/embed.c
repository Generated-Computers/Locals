/* NOR-tree embedding of a NOR circuit into Sequential NOR, with smoothing (my own code).
   usage: embed s d m rho eps seed nsmooth
   cell(t,l) = T_d(u_t) T_d(v_t) tau ; T_0(w) = [ptr 0][ptr copy of w] ; T_{h+1} = T_h T_h [ptr last(left)]
   rho = number of zero-padding gates before each cell, as a multiple of the cell length. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static uint64_t st[2];
static inline uint64_t rnd(void){uint64_t a=st[0],b=st[1];st[0]=b;a^=a<<23;st[1]=a^b^(a>>17)^(b>>26);return st[1]+b;}
static inline double urnd(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static uint32_t *ptr; static uint8_t *x; static uint64_t n; static uint32_t *out; static int m;
static int *U,*V; /* inputs: >=1 gate index, 0 = const c0 (value 0), -1 = const c1 (value 1) */
static uint32_t copy_of(int w){ if(w==0) return 1; if(w==-1) return 0; return out[(uint64_t)(w-1)*m + rnd()%m]; }
static uint32_t build(int h,int w){
  if(h==0){ ptr[++n]=0; ptr[++n]=copy_of(w); return (uint32_t)n; }
  uint32_t l=build(h-1,w); build(h-1,w); ptr[++n]=l; return (uint32_t)n;
}
int main(int argc,char**argv){
  int s=atoi(argv[1]),d=atoi(argv[2]); m=atoi(argv[3]); int rho=atoi(argv[4]); double eps=atof(argv[5]);
  uint64_t seed=strtoull(argv[6],0,10); int ns=atoi(argv[7]);
  st[0]=seed*0x9E3779B97F4A7C15ULL+1; st[1]=seed^0xD1B54A32D192ED03ULL; for(int i=0;i<20;i++) rnd();
  uint64_t cell=6ULL*(1ULL<<d)-1, pre=1000, tot=1+pre+(uint64_t)s*m*cell*(1+rho);
  ptr=malloc((tot+2)*sizeof(uint32_t)); x=malloc(tot+2); out=malloc((uint64_t)s*m*sizeof(uint32_t));
  uint32_t *base=malloc((tot+2)*sizeof(uint32_t));
  U=malloc(s*sizeof(int)); V=malloc(s*sizeof(int)); int *val=malloc((s+1)*sizeof(int));
  /* random circuit: one input is the previous gate with prob 1/2, else a uniform earlier gate or a constant */
  for(int t=1;t<=s;t++){
    int a,b;
    a=(t>1 && urnd()<0.5)? t-1 : (urnd()<0.15? (urnd()<0.5?0:-1) : (t>1? 1+(int)(rnd()%(t-1)) : 0));
    b=(urnd()<0.15)? (urnd()<0.5?0:-1) : (t>1? 1+(int)(rnd()%(t-1)) : -1+(int)(rnd()%2));
    if(urnd()<0.5){int c=a;a=b;b=c;}
    U[t-1]=a; V[t-1]=b;
    int va=a==0?0:(a==-1?1:val[a]), vb=b==0?0:(b==-1?1:val[b]); val[t]=!(va|vb);
  }
  n=0; ptr[0]=0; ptr[++n]=0; while(n<pre) ptr[++n]=0;
  for(int t=1;t<=s;t++) for(int l=0;l<m;l++){
    for(uint64_t i=0;i<cell*rho;i++) ptr[++n]=0;
    uint32_t a=build(d,U[t-1]); build(d,V[t-1]); ptr[++n]=a; out[(uint64_t)(t-1)*m+l]=(uint32_t)n;
  }
  memcpy(base,ptr,(n+1)*sizeof(uint32_t));
  int ones=0; for(int t=1;t<=s;t++) ones+=val[t];
  printf("s=%d d=%d m=%d rho=%d eps=%.4f  n=%llu  cell=%llu  gates with value 1: %d/%d\n",s,d,m,rho,eps,(unsigned long long)n,(unsigned long long)cell,ones,s);
  for(int r=0;r<=ns;r++){
    double e=(r==0)?0.0:eps;
    memcpy(ptr,base,(n+1)*sizeof(uint32_t));
    if(e>0) for(uint64_t i=1;i<=n;i++) if(urnd()<e) ptr[i]=(uint32_t)(rnd()%i);
    x[0]=1; uint64_t no=1; for(uint64_t i=1;i<=n;i++){ x[i]=!(x[i-1]|x[ptr[i]]); no+=x[i]; }
    double sum[2]={0,0},mx[2]={0,0}; int cnt[2]={0,0},lost=0;
    for(int t=1;t<=s;t++){ int bad=0; for(int l=0;l<m;l++) bad+= x[out[(uint64_t)(t-1)*m+l]]!=val[t];
      double f=(double)bad/m; int v=val[t]; sum[v]+=f; cnt[v]++; if(f>mx[v]) mx[v]=f; if(2*bad>=m) lost++; }
    printf("  %s  bad cells: value-0 gates mean %.4f max %.3f | value-1 gates mean %.4f max %.3f | bundles without correct majority: %d | density of ones %.4f\n",
      r==0?"noise 0  ":"smoothed ", cnt[0]?sum[0]/cnt[0]:0,mx[0], cnt[1]?sum[1]/cnt[1]:0,mx[1], lost,(double)no/(n+1));
  }
  return 0;
}
