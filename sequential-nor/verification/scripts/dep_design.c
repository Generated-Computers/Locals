/* dep_design.c -- growth per layer of a flipped position in layered designs whose reads may
   depend on each other (ideal model: a resampled read is a fresh bit that is 1 with probability
   varpi and is the same in both copies).

   A layer has M positions and continues the chain of the layer below.  A design of period T
   gives every phase a rule:
     U        a uniformly random position of the layer below
     Ic       a uniformly random position of phase c of the layer below
     Fd       the position of the layer below at the own offset plus d  (d may be negative)
     Hd       the position two layers below at the own offset plus d
     Rd       the position of the layer below at distance d from the last position listed by a
              rule U, I, F or R (position 0 at the start of a layer)
     Lk       the position k before the gate (k>=2; k=1 is N)
     N        a NOT gate
     Z        position 0 (reads a one)
     O        position 1 (reads a zero)
   Two copies are evaluated with the same choices; in layer T0 a fraction DELTA of the second
   copy is flipped.  Output: (S2/S1)^(1/K), where S1 and S2 are the numbers of differing positions summed
   over the layers T0+BURN .. T0+BURN+K-1 and the K layers after them, K=(L-T0-BURN)/2.

   usage: dep_design eval eps design M L seeds
          dep_design search eps T nsamples seed M L ruleset [climb]
   design: comma separated rules, e.g.  F0,F-1   or  I0,N,Z
   environment: DELTA (fraction flipped, default 0.004), T0 (layer that is flipped, default 8),
   BURN (layers skipped after it, default 2), W (largest |d| and reach in a search, default 4),
   CUT (a search prints designs above it, default 0.99), VERB (print every run).

   The runs of the remark on dependent reads (noise 0.46):
     DELTA=0.003 dep_design eval 0.46 DESIGN 2000000 40 100     (the table)
       with DESIGN one of  U   U,R8   U,R3   U,U,R1   U,R2   U,R-1   U,R1   F0   F0,F0,H-2   F1,F-1
       and  I11,I12,I12,I14,I12,I14,I12,I14,I12,I14,I12,I14,I12,I0,I11,O   (the design D16)
     DELTA=0.02 CUT=0.945 W=5 dep_design search 0.46 T n T+10 200000 26 UIFNZOLH 0
       for T=2,...,6 with n=4000, 10000, 15000, 15000, 15000                      (the screen)
     DELTA=0.02 CUT=0.93 W=4 dep_design search 0.46 T 3000 T+50 200000 26 UIFRNZOLHRRFF 250
       for T=3,4,6,8                                                         (hill climbing)
   A search prints the mean of three runs for every design whose first run is above CUT.     */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static uint64_t S[2];
static inline uint64_t rnd(void){uint64_t s1=S[0],s0=S[1];S[0]=s0;s1^=s1<<23;S[1]=s1^s0^(s1>>17)^(s0>>26);return S[1]+s0;}
static inline double unif(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static void seed(uint64_t s){S[0]=s*0x9E3779B97F4A7C15ULL+1;S[1]=s^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();}

typedef struct{char t;int p;}rule;
#define TMAX 64
static int parse(const char*s,rule*r){int T=0;char buf[1024];strncpy(buf,s,1023);buf[1023]=0;
  for(char*tok=strtok(buf,",");tok;tok=strtok(NULL,",")){r[T].t=tok[0];r[T].p=tok[1]?atoi(tok+1):0;T++;if(T>=TMAX)break;}return T;}
static void show(const rule*r,int T,char*out){out[0]=0;for(int i=0;i<T;i++){char b[32];
  if(r[i].t=='U'||r[i].t=='N'||r[i].t=='Z'||r[i].t=='O')sprintf(b,"%c",r[i].t);else sprintf(b,"%c%d",r[i].t,r[i].p);
  strcat(out,b);if(i+1<T)strcat(out,",");}}

static double varpi(double e){double lo=0,hi=1;for(int i=0;i<200;i++){double m=(lo+hi)/2;if(e*(1-m)*(1-m)-m>0)lo=m;else hi=m;}return lo;}

static int T0=8,BURN=2;static double DELTA=0.004;

/* returns growth factor per layer; *last = number of differing positions in the last layer */
static double run(double eps,const rule*r,int T,int M,int L,long*last){
  double w=varpi(eps);
  uint8_t *a0=malloc(M),*a1=malloc(M),*a2=malloc(M),*b0=malloc(M),*b1=malloc(M),*b2=malloc(M);
  for(int o=0;o<M;o++){a1[o]=b1[o]=unif()<0.35;a2[o]=b2[o]=unif()<0.35;}
  for(int o=1;o<M;o++){if(a1[o-1])a1[o]=b1[o]=0;if(a2[o-1])a2[o]=b2[o]=0;}
  long d=0;int K=(L-T0-BURN)/2;double s1=0,s2=0;
  uint64_t thr=(uint64_t)(eps*18446744073709551615.0),thw=(uint64_t)(w*18446744073709551615.0);
  for(int t=0;t<L;t++){
    int pa=a1[M-1],pb=b1[M-1];d=0;int lastidx=0;
    for(int o=0;o<M;o++){
      int ra,rb;const rule q=r[o%T];int idx=-1;
      switch(q.t){
        case 'U':idx=(int)(rnd()%(uint64_t)M);break;
        case 'I':idx=(int)(rnd()%(uint64_t)(M/T))*T+q.p;break;
        case 'F':case 'H':idx=((o+q.p)%M+M)%M;break;
        case 'R':idx=((lastidx+q.p)%M+M)%M;break;
        default:break;}
      if(idx>=0&&q.t!='H')lastidx=idx;
      if(rnd()<thr){ra=rb=(rnd()<thw);}
      else{
        switch(q.t){
          case 'U':case 'I':case 'F':case 'R':ra=a1[idx];rb=b1[idx];break;
          case 'H':ra=a2[idx];rb=b2[idx];break;
          case 'L':idx=o-q.p;if(idx>=0){ra=a0[idx];rb=b0[idx];}else{ra=a1[M+idx];rb=b1[M+idx];}break;
          case 'N':ra=pa;rb=pb;break;
          case 'Z':ra=rb=1;break;
          default:ra=rb=0;break;}
      }
      pa=!(pa|ra);pb=!(pb|rb);a0[o]=pa;b0[o]=pb;d+=(pa!=pb);
    }
    if(t==T0){for(int o=0;o<M;o++)if(unif()<DELTA)b0[o]^=1;}
    if(t>=T0+BURN&&t<T0+BURN+K)s1+=d;else if(t>=T0+BURN+K&&t<T0+BURN+2*K)s2+=d;
    uint8_t*tmp=a2;a2=a1;a1=a0;a0=tmp;tmp=b2;b2=b1;b1=b0;b0=tmp;
  }
  if(last)*last=d;
  double g=(s1>0&&K>0)?pow(s2/s1,1.0/K):0;
  free(a0);free(a1);free(a2);free(b0);free(b1);free(b2);return g;
}

static const char*RS="UIFNZOL";
static void randrule(rule*q,int T,const char*rs,int W){
  char c=rs[rnd()%strlen(rs)];q->t=c;q->p=0;
  if(c=='I')q->p=(int)(rnd()%(uint64_t)T);
  else if(c=='F'||c=='H')q->p=(int)(rnd()%(uint64_t)(2*W+1))-W;
  else if(c=='R'){do{q->p=(int)(rnd()%(uint64_t)(2*W+1))-W;}while(q->p==0);}
  else if(c=='L')q->p=2+(int)(rnd()%(uint64_t)(W>1?W-1:1));
}
static int valid(const rule*r,int T){int far=0;for(int i=0;i<T;i++)if(strchr("UIFH",r[i].t))far=1;if(r[0].t=='R')return 0;return far;}

int main(int argc,char**argv){
  if(argc<2){fprintf(stderr,"usage\n");return 1;}
  if(getenv("DELTA"))DELTA=atof(getenv("DELTA"));
  if(getenv("T0"))T0=atoi(getenv("T0"));
  if(getenv("BURN"))BURN=atoi(getenv("BURN"));
  if(!strcmp(argv[1],"eval")){
    double eps=atof(argv[2]);rule r[TMAX];int T=parse(argv[3],r);int M=atoi(argv[4]),L=atoi(argv[5]),ns=atoi(argv[6]);
    M-=M%T;double s=0,s2=0;
    for(int k=1;k<=ns;k++){seed(1000+k);long last;double g=run(eps,r,T,M,L,&last);s+=g;s2+=g*g;
      if(getenv("VERB"))printf("  seed %d: %.5f (last layer %ld)\n",k,g,last);}
    double m=s/ns,sd=ns>1?sqrt((s2/ns-m*m)/(ns-1)):0;
    printf("eps %.4f  %s  growth %.5f +- %.5f  (M=%d L=%d seeds=%d)\n",eps,argv[3],m,sd,M,L,ns);
    return 0;}
  if(!strcmp(argv[1],"search")){
    double eps=atof(argv[2]);int T=atoi(argv[3]);long ns=atol(argv[4]);uint64_t sd=atoll(argv[5]);
    int M=atoi(argv[6]),L=atoi(argv[7]);const char*rs=argc>8?argv[8]:RS;int climb=argc>9?atoi(argv[9]):0;
    int W=getenv("W")?atoi(getenv("W")):4;double cut=getenv("CUT")?atof(getenv("CUT")):0.99;
    M-=M%T;seed(sd);rule r[TMAX],best[TMAX];double gb=0;char buf[2048];
    for(long k=0;k<ns;k++){
      do{for(int i=0;i<T;i++)randrule(&r[i],T,rs,W);}while(!valid(r,T));
      double g=run(eps,r,T,M,L,NULL);
      if(g>cut){double g2=(g+run(eps,r,T,M,L,NULL)+run(eps,r,T,M,L,NULL))/3;show(r,T,buf);
        if(g2>cut){printf("%.4f %s\n",g2,buf);fflush(stdout);}
        if(g2>gb){gb=g2;memcpy(best,r,sizeof(rule)*T);}}
    }
    if(gb>0){show(best,T,buf);printf("best %.4f %s\n",gb,buf);}
    /* hill climbing from the best */
    for(int it=0;it<climb&&gb>0;it++){
      memcpy(r,best,sizeof(rule)*T);int i=(int)(rnd()%(uint64_t)T);
      do{randrule(&r[i],T,rs,W);}while(!valid(r,T));
      double g=(run(eps,r,T,M,L,NULL)+run(eps,r,T,M,L,NULL)+run(eps,r,T,M,L,NULL))/3;
      if(g>gb){double g2=(run(eps,r,T,M,L,NULL)+run(eps,r,T,M,L,NULL)+run(eps,r,T,M,L,NULL))/3;
        if(g2>gb){gb=(g+g2)/2;memcpy(best,r,sizeof(rule)*T);show(best,T,buf);printf("climb %.4f %s\n",gb,buf);fflush(stdout);}}
    }
    return 0;}
  return 1;
}
