/* hunt.c: rounds of the table-Jacobi iteration with reach R (and max D of the relevance-pruned
   iteration, and optionally the sound set-valued table iteration) on eps-smoothings of a base
   given by a periodic "spec".
   usage: hunt n eps R pf m nseeds spec [maxr] [sv] [seed0]
     n      length; positions 1..n0 (n0 = pf*n) are zero gates (j=0)
     m      layer width above the preamble (0: no layers); layer 0 consists of zero gates
     spec   comma separated tokens, token o%P is used at offset o (offset in the layer, or i if m=0):
            L<d>      j = i-d
            E<c>      uniformly random position of the layer below with offset = c mod P
            A         uniformly random position of the layer below
            Q<w>[-<c>] position of the layer below given by a table indexed by (o mod w), fresh per layer
                      (with -<c>: moved to offset = c mod P); offsets o and o+w are twins
            P<k>      the same position as gate i-k reads in the base (a repeated read)
            Z  O      j = 0, j = 1
            U<a>-<b>  uniform distance in [a,b]
            F<c>      j = floor(i*c/1000)
            H<d>      distance d*2^v2(i)
            S<d>      distance d*(1+(number of trailing ones of floor(o/P)))
     maxr   cap on rounds (default 100000); sv=1: also run the set-valued iteration (R<=62, sets capped)
   output: one line per call: mean/max over seeds of TJ (least s with y^(s) = trace), max D, density. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static uint64_t s[2];
static inline uint64_t rnd(void){uint64_t a=s[0],b=s[1];s[0]=b;a^=a<<23;s[1]=a^b^(a>>17)^(b>>26);return s[1]+b;}
static inline double uni(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static inline uint32_t below(uint32_t k){return (uint32_t)(uni()*k);}
typedef struct{char t;long a,b;}tok_t;
static tok_t tk[4096]; static int P;
static uint32_t n,*j; static uint8_t *x,*y,*z,*kn,*kn2; static int32_t *D;
static int cmp64(const void*a,const void*b){uint64_t u=*(const uint64_t*)a,v=*(const uint64_t*)b;return u<v?-1:u>v;}
#define CAP (1<<16)
/* set-valued iteration; returns rounds, -1 on overflow of the cap, -2 if unsound */
static int setval(int R,int maxr){
  static uint64_t *cur,*nxt; if(!cur){cur=malloc(8*(size_t)CAP*2+16);nxt=malloc(8*(size_t)CAP*2+16);}
  uint64_t msk=(R>=64)?~0ULL:((1ULL<<R)-1);
  memset(kn,0,n+1);kn[0]=1;int rounds=0;uint32_t unknown=n;
  while(unknown>0&&rounds<maxr){rounds++;size_t nc=1;cur[0]=1;kn2[0]=1;unknown=0;
    for(uint32_t i=1;i<=n;i++){uint32_t p=j[i],d=i-p;size_t nn=0;int any0=0,any1=0;
      for(size_t q=0;q<nc;q++){uint64_t sg=cur[q];int a=sg&1;
        if(d<=(uint32_t)R){int r=(sg>>(d-1))&1;int v=!(a|r);nxt[nn++]=((sg<<1)|v)&msk;if(v)any1=1;else any0=1;}
        else if(kn[p]){int v=!(a|x[p]);nxt[nn++]=((sg<<1)|v)&msk;if(v)any1=1;else any0=1;}
        else{int v0=!a;nxt[nn++]=((sg<<1)|v0)&msk;any0=1;if(v0){any1=1;nxt[nn++]=(sg<<1)&msk;}}}
      if(nn>1){qsort(nxt,nn,8,cmp64);size_t w=1;for(size_t q=1;q<nn;q++)if(nxt[q]!=nxt[w-1])nxt[w++]=nxt[q];nn=w;}
      if(nn>CAP)return -1;
      kn2[i]=!(any0&&any1);if(!kn2[i])unknown++;else if((any1?1:0)!=x[i])return -2;
      uint64_t*w=cur;cur=nxt;nxt=w;nc=nn;}
    for(uint32_t i=0;i<=n;i++)if(kn[i]&&!kn2[i])return -3; /* monotone */
    uint8_t*w=kn;kn=kn2;kn2=w;}
  return unknown?maxr+1:rounds;
}
static int jacobi(uint32_t R,int maxr){
  memset(y,0,n+1);y[0]=1;int r;
  for(r=0;r<=maxr;r++){if(memcmp(y,x,n+1)==0)return r;z[0]=1;
    for(uint32_t k=1;k<=n;k++){uint32_t t=j[k];uint8_t v=(k-t<=R)?z[t]:y[t];z[k]=!(z[k-1]|v);}
    uint8_t*w=y;y=z;z=w;}
  return r;
}
int main(int argc,char**argv){
  if(argc<8){fprintf(stderr,"usage: hunt n eps R pf m nseeds spec [maxr] [sv] [seed0]\n");return 1;}
  n=atoi(argv[1]);double eps=atof(argv[2]);uint32_t R=atoi(argv[3]);double pf=atof(argv[4]);uint32_t m=atoi(argv[5]);
  int ns=atoi(argv[6]);char*spec=strdup(argv[7]);int maxr=argc>8?atoi(argv[8]):100000;int sv=argc>9?atoi(argv[9]):0;int seed0=argc>10?atoi(argv[10]):1;
  P=0;for(char*t=strtok(spec,",");t;t=strtok(0,",")){tk[P].t=t[0];tk[P].a=0;tk[P].b=-1;
    if(t[1]){char*e;tk[P].a=strtol(t+1,&e,10);if(*e=='-')tk[P].b=strtol(e+1,0,10);}P++;}
  if(m)m-=m%P; uint32_t n0=(uint32_t)(pf*n);
  j=malloc(4*(size_t)(n+1));x=malloc(n+1);y=malloc(n+1);z=malloc(n+1);if(sv){kn=malloc(n+1);kn2=malloc(n+1);}D=malloc(4*(size_t)(n+1));
  uint32_t *qt=malloc(4*(size_t)(m+1));
  double sT=0,sD=0,sS=0,dens=0;int mT=0,mDm=0,mS=0,svbad=0;
  for(int sd=seed0;sd<seed0+ns;sd++){
    s[0]=(uint64_t)sd*0x9E3779B97F4A7C15ULL+1;s[1]=(uint64_t)sd^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();
    j[0]=0;uint32_t curlayer=0xffffffff;
    for(uint32_t i=1;i<=n;i++){long b=0;
      if(i>n0){uint32_t off=i-n0-1,t=m?off/m:1,o=m?off%m:i;tok_t*q=&tk[o%P];
        if(m&&t!=curlayer){curlayer=t;for(uint32_t u=0;u<m;u++)qt[u]=below(m);}
        uint32_t lb=m?n0+1+(t-1)*m:0;
        if(t==0)b=0;else switch(q->t){
          case 'L':b=(long)i-q->a;break;
          case 'E':b=lb+(long)P*below(m/P)+q->a;break;
          case 'A':b=lb+below(m);break;
          case 'Q':b=q->b<0?lb+qt[o%q->a]:lb+(long)(qt[o%q->a]/P)*P+q->b;break;
          case 'P':b=(i>(uint32_t)q->a)?(long)j[i-q->a]:0;break;
          case 'Z':b=0;break; case 'O':b=1;break;
          case 'U':b=(long)i-(q->a+below(q->b-q->a+1));break;
          case 'F':b=(long)((double)i*q->a/1000.0);break;
          case 'H':{int v=__builtin_ctz(i);b=(long)i-((long)q->a<<v);}break;
          case 'S':{uint32_t c=o/P;int v=0;while(c&1){v++;c>>=1;}b=(long)i-q->a*(1+v);}break;
          default:fprintf(stderr,"bad token\n");return 1;}
        if(b<0)b=0;if(b>=(long)i)b=i-1;}
      j[i]=b;}
    for(uint32_t i=1;i<=n;i++)if(uni()<eps)j[i]=below(i);
    x[0]=1;D[0]=0;int32_t mD=0;uint64_t ones=1;
    for(uint32_t k=1;k<=n;k++){uint32_t t=j[k];uint8_t xp=x[k-1],xj=x[t];x[k]=!(xp|xj);ones+=x[k];
      int notg=(t==k-1),reset=(!notg)&&xj,rel=!xp;
      if(k==1){D[k]=1;continue;}
      if(reset&&rel)D[k]=1+D[t];else if(reset)D[k]=D[k-1]<1+D[t]?D[k-1]:1+D[t];
      else if(rel&&!notg)D[k]=D[k-1]>1+D[t]?D[k-1]:1+D[t];else D[k]=D[k-1];
      if(D[k]>mD)mD=D[k];}
    int r=jacobi(R,maxr);
    if(getenv("HUNT_DUMP")){FILE*f=fopen(getenv("HUNT_DUMP"),"w");fprintf(f,"%d %d %d\n",r,mD,sv?setval(R,maxr):-9);for(uint32_t i=1;i<=n;i++)fprintf(f,"%u\n",j[i]);fclose(f);}
    if(r>mD){fprintf(stderr,"VIOLATION: TJ %d > max D %d\n",r,mD);}
    sT+=r;if(r>mT)mT=r;sD+=mD;if(mD>mDm)mDm=mD;dens+=(double)ones/(n+1);
    if(sv){int q=setval(R,maxr);if(q<0){svbad=q;}else{sS+=q;if(q>mS)mS=q;if(r>q)fprintf(stderr,"VIOLATION: TJ %d > SV %d\n",r,q);if(q>mD)fprintf(stderr,"VIOLATION: SV %d > max D %d\n",q,mD);}}
  }
  printf("%s n=%u eps=%.3f R=%u pf=%.2f m=%u seeds=%d | TJ mean %.1f max %d%s | maxD mean %.1f max %d | dens %.3f",argv[7],n,eps,R,pf,m,ns,sT/ns,mT,mT>maxr?"+":"",sD/ns,mDm,dens/ns);
  if(sv){if(svbad)printf(" | SV err %d",svbad);else printf(" | SV mean %.1f max %d",sS/ns,mS);}
  printf("\n");return 0;}
