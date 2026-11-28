/* refsim.c -- referee's own simulator for Remark "dependent reads; numerical".
   Written from the description in the remark, not from dep_design.c.

   Chain: x_i = NOR(x_{i-1}, read_i).  Layers of M positions; a layer continues the chain of the
   layer below.  With probability eps a read is resampled: a fresh bit, 1 with probability varpi
   (varpi = eps (1-varpi)^2), the same in both copies.  Otherwise it is the value of the position
   that the rule of the gate's phase lists.

   Rules (phase = offset mod T):
     U      uniform position of the layer below               (sets the anchor)
     Ic     uniform position of phase c of the layer below    (sets the anchor)
     Fd     layer below, own offset + d                       (sets the anchor)
     Hd     two layers below, own offset + d
     Rd     layer below, anchor + d                           (moves the anchor there)
     Ad     layer below, anchor + d                           (anchor stays)            [referee]
     Gd     the position listed by gate (anchor + d) of the layer below (anchor stays)  [referee]
     Qd     draw a new uniform anchor (not read) and read what gate (anchor+d) lists    [referee]
     V      uniform position two layers below                                           [referee]
     X      uniform position of the layer below, anchor stays                           [referee]
     Yc     uniform position of phase c of the layer below, anchor stays                [referee]
     Pk     the position listed by the gate k before in the chain (sets the anchor)     [referee]
     Jc     uniform position of phase c two layers below                                [referee]
     Lk     k positions back in the chain
     N      NOT gate (reads the position before it)
     Z      reads 1         O   reads 0
   A rule may be written X|Y@p : with probability p percent the gate has rule Y instead of X
   (a fixed choice of the base, the same in both copies).

   Two copies, same choices; after layer T0 is evaluated every position of copy b in it is flipped
   with probability DELTA.  Output per run:
     layer growth  (S2/S1)^(1/K), S1 = differing positions summed over layers T0+BURN..T0+BURN+K-1,
                   S2 over the next K layers;
     generation growth: every differing position gets a generation (flipped positions 0; a
                   difference that continues along the chain or is caused through a local pointer
                   keeps the generation; one that is born from a differing read of a lower layer
                   gets that generation + 1).  Zn = number of differing positions of generation n;
                   growth = (sum_{n=GB+GK..GB+2GK-1} Zn / sum_{n=GB..GB+GK-1} Zn)^(1/GK),
                   only reported if those generations are complete within the L layers.

   usage: refsim eps design M L nseeds [seed0]
   env:   DELTA (0.003) T0 (8) BURN (2) K (default (L-T0-BURN)/2) GB (2) GK (K) VERB
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static uint64_t xs[4];
static inline uint64_t rotl(uint64_t x,int k){return (x<<k)|(x>>(64-k));}
static inline uint64_t nxt(void){
  uint64_t r=rotl(xs[1]*5,7)*9,t=xs[1]<<17;
  xs[2]^=xs[0];xs[3]^=xs[1];xs[1]^=xs[2];xs[0]^=xs[3];xs[2]^=t;xs[3]=rotl(xs[3],45);return r;}
static uint64_t smx;
static uint64_t splitmix(void){uint64_t z=(smx+=0x9E3779B97F4A7C15ULL);z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
  z=(z^(z>>27))*0x94D049BB133111EBULL;return z^(z>>31);}
static void seedrng(uint64_t x){smx=x*0x2545F4914F6CDD1DULL+12345;for(int i=0;i<4;i++)xs[i]=splitmix();}
static inline uint32_t below(uint32_t n){return (uint32_t)(((__uint128_t)nxt()*n)>>64);}
static inline double u01(void){return (nxt()>>11)*(1.0/9007199254740992.0);}

typedef struct{char t;int p;char t2;int p2;double mix;}rule;
#define TMAX 256
static int parse1(const char*s,char*t,int*p){*t=s[0];*p=s[1]?atoi(s+1):0;
  if(!strchr("UIFHRAGQVJLNZOXYP",*t)){fprintf(stderr,"bad rule %s\n",s);exit(2);}return 0;}
static int parse(const char*s,rule*r){int T=0;char buf[4096];strncpy(buf,s,4095);buf[4095]=0;
  char*save;for(char*tok=strtok_r(buf,",",&save);tok;tok=strtok_r(NULL,",",&save)){
    r[T].mix=0;r[T].t2=0;r[T].p2=0;char*bar=strchr(tok,'|');
    if(bar){*bar=0;char*at=strchr(bar+1,'@');if(!at){fprintf(stderr,"mix needs @\n");exit(2);}
      *at=0;parse1(bar+1,&r[T].t2,&r[T].p2);r[T].mix=atof(at+1)/100.0;}
    parse1(tok,&r[T].t,&r[T].p);T++;if(T>=TMAX){fprintf(stderr,"too long\n");exit(2);}}
  return T;}

static double varpi(double e){return (2*e+1-sqrt(4*e+1))/(2*e);}

typedef struct{uint8_t*a,*b,*gen,*bd;int32_t*ptr;int8_t*dep;}layer;
#define NLMAX 12
static int NL=4;
#define GMAX 256

static int T0=8,BURN=2,KK=-1,GB=2,GK=-1,VERB=0;static double DELTA=0.003;

typedef struct{double glayer,ggen,gslope,r0hat,runlen,deep;int genok;long flipped;}result;

static result run(double eps,const rule*r,int T,int M,int L,uint64_t seed){
  seedrng(seed);
  double w=varpi(eps);
  uint64_t thr=(uint64_t)(eps*18446744073709551616.0),thw=(uint64_t)(w*18446744073709551616.0);
  int needptr=0;for(int i=0;i<T;i++){if(strchr("GQP",r[i].t))needptr=1;if(r[i].mix>0&&strchr("GQP",r[i].t2))needptr=1;}
  NL=needptr?NLMAX:4;
  layer lay[NLMAX];
  for(int i=0;i<NL;i++){lay[i].a=malloc(M);lay[i].b=malloc(M);lay[i].gen=calloc(M,1);lay[i].bd=calloc(M,1);
    lay[i].ptr=needptr?malloc(sizeof(int32_t)*(size_t)M):NULL;lay[i].dep=needptr?malloc(M):NULL;
    if(!lay[i].a||!lay[i].b||!lay[i].gen||(needptr&&(!lay[i].ptr||!lay[i].dep))){fprintf(stderr,"oom\n");exit(3);}
    for(int o=0;o<M;o++){uint8_t v=u01()<0.4;lay[i].a[o]=lay[i].b[o]=v;}
    if(needptr)for(int o=0;o<M;o++){lay[i].ptr[o]=0;lay[i].dep[o]=-2;}}
  double*D=calloc(L,sizeof(double));double Z[GMAX];for(int i=0;i<GMAX;i++)Z[i]=0;
  int maxdep=1;long flipped=0;double CB[NLMAX+1];for(int i=0;i<=NLMAX;i++)CB[i]=0;double births=0,dwin=0,fallback=0,gates=0;
  int K=KK>0?KK:(L-T0-BURN)/2;
  for(int t=0;t<L;t++){
    layer*cur=&lay[t%NL];layer*b1=&lay[(t+NL-1)%NL];
    int pa=b1->a[M-1],pb=b1->b[M-1],pg=b1->gen[M-1],pd=b1->bd[M-1];
    int inwin=(t>=T0+BURN&&t<T0+BURN+2*K);
    long d=0;int32_t anchor=0;int ph=0;
    for(int o=0;o<M;o++){
      const rule*q=&r[ph];if(++ph==T)ph=0;
      char ty=q->t;int par=q->p;
      if(q->mix>0&&u01()<q->mix){ty=q->t2;par=q->p2;}
      int kind=1;int32_t idx=0;
      switch(ty){
        case 'U':idx=(int32_t)below((uint32_t)M);anchor=idx;break;
        case 'X':idx=(int32_t)below((uint32_t)M);break;
        case 'Y':idx=(int32_t)below((uint32_t)(M/T))*T+par;break;
        case 'P':if(o-par>=0){kind=cur->dep[o-par];idx=cur->ptr[o-par];if(kind==1)anchor=idx;}
                 else{idx=(int32_t)below((uint32_t)M);anchor=idx;}break;
        case 'I':idx=(int32_t)below((uint32_t)(M/T))*T+par;anchor=idx;break;
        case 'F':idx=(int32_t)((((long)o+par)%M+M)%M);anchor=idx;break;
        case 'H':idx=(int32_t)((((long)o+par)%M+M)%M);kind=2;break;
        case 'V':idx=(int32_t)below((uint32_t)M);kind=2;break;
        case 'J':idx=(int32_t)below((uint32_t)(M/T))*T+par;kind=2;break;
        case 'R':idx=(int32_t)((((long)anchor+par)%M+M)%M);anchor=idx;break;
        case 'A':idx=(int32_t)((((long)anchor+par)%M+M)%M);break;
        case 'Q':anchor=(int32_t)below((uint32_t)M); /* fall through */
        case 'G':{int32_t g=(int32_t)((((long)anchor+par)%M+M)%M);int dp=b1->dep[g];
          if(dp>=0){kind=1+dp;idx=b1->ptr[g];}else kind=dp;}break;
        case 'N':par=1; /* fall through */
        case 'L':if(o-par>=0){kind=0;idx=o-par;}else{kind=1;idx=M+o-par;}break;
        case 'Z':kind=-1;break;
        case 'O':kind=-2;break;
        default:fprintf(stderr,"rule?\n");exit(2);}
      if(kind>=NL-1){idx=(int32_t)below((uint32_t)M);kind=2;fallback+=1;}
      gates+=1;
      if(kind>maxdep)maxdep=kind;
      if(needptr){cur->ptr[o]=idx;cur->dep[o]=(int8_t)kind;}
      int ra,rb,sg=0,sd=0;
      if(nxt()<thr){ra=rb=(nxt()<thw);}
      else if(kind==-1){ra=rb=1;}
      else if(kind==-2){ra=rb=0;}
      else{const layer*S=&lay[(t+NL-kind)%NL];ra=S->a[idx];rb=S->b[idx];
        if(ra!=rb){sg=S->gen[idx]+(kind>0);sd=kind>0?kind:S->bd[idx];}}
      int na=!(pa|ra),nb=!(pb|rb);int g=0,bdv=0;
      if(na!=nb){d++;if(pa!=pb){g=pg;bdv=pd;}else{g=sg;bdv=sd;if(inwin)births+=1;}if(g>=GMAX)g=GMAX-1;Z[g]+=1;if(inwin)CB[bdv]+=1;}
      cur->a[o]=(uint8_t)na;cur->b[o]=(uint8_t)nb;cur->gen[o]=(uint8_t)g;cur->bd[o]=(uint8_t)bdv;pa=na;pb=nb;pg=g;pd=bdv;
    }
    if(t==T0){for(int o=0;o<M;o++)if(u01()<DELTA){cur->b[o]^=1;cur->gen[o]=0;cur->bd[o]=1;flipped++;}Z[0]+=flipped;}
    D[t]=(double)d;if(inwin)dwin+=d;
  }
  double s1=0,s2=0;
  for(int t=T0+BURN;t<T0+BURN+K&&t<L;t++)s1+=D[t];
  for(int t=T0+BURN+K;t<T0+BURN+2*K&&t<L;t++)s2+=D[t];
  result R;R.flipped=flipped;
  R.glayer=(s1>0&&s2>0)?pow(s2/s1,1.0/K):0;
  /* least-squares slope of log D over the same 2K layers */
  {double sx=0,sy=0,sxx=0,sxy=0;int n=0;for(int t=T0+BURN;t<T0+BURN+2*K&&t<L;t++)if(D[t]>0){double x=t,y=log(D[t]);sx+=x;sy+=y;sxx+=x*x;sxy+=x*y;n++;}
   R.gslope=n>2?exp((n*sxy-sx*sy)/(n*sxx-sx*sx)):0;}
  int gk=GK>0?GK:K;int lastcomplete=(L-1-T0)/maxdep; /* generation n lies in layers T0+n..T0+n*maxdep */
  R.genok=(GB+2*gk-1<=lastcomplete);
  double z1=0,z2=0;for(int n=GB;n<GB+gk&&n<GMAX;n++)z1+=Z[n];for(int n=GB+gk;n<GB+2*gk&&n<GMAX;n++)z2+=Z[n];
  R.ggen=(z1>0&&z2>0)?pow(z2/z1,1.0/gk):0;
  {double tot=0,acc=0;for(int k=1;k<=NLMAX;k++)tot+=CB[k];for(int k=1;k<=NLMAX;k++)acc+=CB[k]/(tot>0?tot:1)*pow(R.glayer,k);R.r0hat=acc;}
  R.runlen=births>0?dwin/births:0;R.deep=gates>0?fallback/gates:0;
  if(VERB){printf("  seed %llu flipped %ld maxdep %d\n   D:",(unsigned long long)seed,flipped,maxdep);
    for(int t=T0;t<L;t++)printf(" %.0f",D[t]);printf("\n   Z:");
    for(int n=0;n<=lastcomplete&&n<GMAX;n++)printf(" %.0f",Z[n]);printf("\n");}
  for(int i=0;i<NL;i++){free(lay[i].a);free(lay[i].b);free(lay[i].gen);free(lay[i].bd);free(lay[i].ptr);free(lay[i].dep);}
  free(D);return R;
}

int main(int argc,char**argv){
  if(argc<6){fprintf(stderr,"usage: refsim eps design M L nseeds [seed0]\n");return 1;}
  if(getenv("DELTA"))DELTA=atof(getenv("DELTA"));
  if(getenv("T0"))T0=atoi(getenv("T0"));
  if(getenv("BURN"))BURN=atoi(getenv("BURN"));
  if(getenv("K"))KK=atoi(getenv("K"));
  if(getenv("GB"))GB=atoi(getenv("GB"));
  if(getenv("GK"))GK=atoi(getenv("GK"));
  if(getenv("VERB"))VERB=1;
  double eps=atof(argv[1]);rule r[TMAX];int T=parse(argv[2],r);
  int M=atoi(argv[3]),L=atoi(argv[4]),ns=atoi(argv[5]);uint64_t seed0=argc>6?strtoull(argv[6],0,10):7000;
  M-=M%T;
  double s=0,ss=0,g=0,gg=0,sl=0,sll=0,r0=0,r00=0,rl=0,dp=0;int genok=1;
  for(int k=0;k<ns;k++){result R=run(eps,r,T,M,L,seed0+k);
    s+=R.glayer;ss+=R.glayer*R.glayer;g+=R.ggen;gg+=R.ggen*R.ggen;sl+=R.gslope;sll+=R.gslope*R.gslope;genok&=R.genok;r0+=R.r0hat;r00+=R.r0hat*R.r0hat;rl+=R.runlen;dp+=R.deep;
    if(VERB)printf("  run %d: layer %.5f slope %.5f gen %.5f%s\n",k,R.glayer,R.gslope,R.ggen,R.genok?"":" (incomplete)");}
  double m=s/ns,se=ns>1?sqrt((ss/ns-m*m)/(ns-1)):0;
  double mg=g/ns,seg=ns>1?sqrt(fabs(gg/ns-mg*mg)/(ns-1)):0;
  double ml=sl/ns,sel=ns>1?sqrt(fabs(sll/ns-ml*ml)/(ns-1)):0;
  double mr=r0/ns,ser=ns>1?sqrt(fabs(r00/ns-mr*mr)/(ns-1)):0;
  printf("eps %.4f %-26s layer %.5f +- %.5f | pergen: R0hat %.5f +- %.5f  gentrack %.5f +- %.5f%s | slope %.5f runlen %.3f%s (M=%d L=%d n=%d delta=%g)\n",
    eps,argv[2],m,se,mr,ser,mg,seg,genok?"":"(inc)",ml,rl/ns,dp>0?" deepfallback":"",M,L,ns,DELTA);
  (void)sel;
  return 0;
}
