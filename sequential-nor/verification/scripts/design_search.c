// design_search.c: search for layered periodic designs on which the table-Jacobi iteration needs
// many rounds although its reach R = HMAX covers every short pointer of the design (numerical;
// used in the remark after the theorem on layered bases).
// Zero gates below, then layers of m positions. A design has period P; a gate at offset o of a
// layer t >= 1, with class c = o mod P, follows the rule of its class:
//   S(h): j = i-h, 1 <= h <= HMAX         F(d): a random position of layer t-1 with offset = d mod P
//   A(d): j = i-m-d (a far read at a fixed distance)   Z: j = 0
// The program draws the smoothing, computes the trace and prints the least s with y^(s) equal to
// the trace (one less than the number of applications of G_R in the definition); 'applications'
// below means this number.
// usage: design_search eps m af L1 L2 ndesigns seed HMAX [design]
//   af = fraction of the positions in the layers; L = number of layers.
//   ndesigns > 0: random designs; prints those with at least 25 applications at L1 layers and at
//                 least 12 more (and at least 40) at L2 layers.
//   ndesigns < 0: hill climbing, six restarts of -ndesigns steps, objective = mean number of
//                 applications at L2 layers over two smoothings.
//   with a design string such as "S2,F0": that design at L1, 2 L1, ..., L2 layers, three smoothings.
// Environment: RR=r sets the reach of the iteration to r (default HMAX). DR=1 prints, instead of
//   the number of applications, the largest three-valued depth when pointers of reach <= R cost no
//   round: the recursion of the lemma on the pruned depth with "1+D(p)" replaced by "D(p)" for them.
// The paper: ./design_search 0.55 600 0.5 40 160 6000 1 6 ; ./design_search 0.50 600 0.5 40 160 6000 2 6 ;
//   ./design_search 0.50 600 0.5 40 160 -250 3 8 ; and e.g. ./design_search 0.50 600 0.5 160 1280 0 5 6 "F0,F1" ;
//   RR=2 ./design_search 0.50 600 0.5 40 320 0 1 2 "S2,F0" and the same with DR=1 (the mixed base).
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static uint64_t s[2];
static inline uint64_t rnd(void){uint64_t a=s[0],b=s[1];s[0]=b;a^=a<<23;s[1]=a^b^(a>>17)^(b>>26);return s[1]+b;}
static inline double uni(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static inline uint32_t below(uint32_t k){return (uint32_t)(uni()*k);}
static void seed(uint64_t v){s[0]=v*0x9E3779B97F4A7C15ULL+1;s[1]=v^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();}
static uint32_t *j; static uint8_t *x,*y,*z;
typedef struct{int P; char ty[16]; int par[16];} design;
static int run(const design*d,double eps,uint32_t m,double af,uint32_t L,uint32_t R,uint64_t sd,int cap){
  uint32_t top=(L+1)*m, n0=(uint32_t)(top*(1.0-af)/af); uint32_t n=n0+top;
  seed(sd); j[0]=0;
  for(uint32_t i=1;i<=n;i++){uint32_t b=0;
    if(i>n0+m){uint32_t off=i-n0-1,t=off/m,o=off%m,base=n0+1+(t-1)*m; int c=o%d->P;
      switch(d->ty[c]){case 'S': b=i-d->par[c]; break;
        case 'F': b=base+d->P*below(m/d->P)+d->par[c]; break;
        case 'A': b=i-m-d->par[c]; break; default: b=0;}}
    j[i]=b;}
  for(uint32_t i=1;i<=n;i++) if(uni()<eps) j[i]=below(i);
  x[0]=1; for(uint32_t k=1;k<=n;k++) x[k]=!(x[k-1]|x[j[k]]);
  if(getenv("DR")){ static uint32_t *D=0; static uint32_t cap2=0; if(cap2<n+1){D=realloc(D,4*(size_t)(n+1));cap2=n+1;}
    uint32_t mx=0; D[0]=0;
    for(uint32_t k=1;k<=n;k++){uint32_t p=j[k]; int notg=(p==k-1), reset=(!notg)&&x[p], rel=(x[k-1]==0); uint32_t c=(k-p<=R)?0:1, v;
      if(k==1) v=1; else if(reset&&rel) v=c+D[p]; else if(reset) v=(D[k-1]<c+D[p])?D[k-1]:c+D[p];
      else if(rel&&!notg) v=(D[k-1]>c+D[p])?D[k-1]:c+D[p]; else v=D[k-1];
      D[k]=v; if(v>mx) mx=v;}
    return (int)mx;}
  memset(y,0,n+1);y[0]=1;
  for(int r=0;r<cap;r++){ if(memcmp(y,x,n+1)==0) return r; z[0]=1;
    for(uint32_t k=1;k<=n;k++){uint32_t t=j[k];uint8_t v=(k-t<=R)?z[t]:y[t];z[k]=!(z[k-1]|v);}
    uint8_t*w=y;y=z;z=w;}
  return cap;}
static void show(const design*d,char*buf){buf[0]=0;for(int c=0;c<d->P;c++){char t[16];sprintf(t,"%s%c%d",c?",":"",d->ty[c],d->par[c]);strcat(buf,t);}}
int main(int argc,char**argv){
  if(argc<9){fprintf(stderr,"usage: design_search eps m af L1 L2 ndesigns seed HMAX [design]\n");return 1;}
  double eps=atof(argv[1]);uint32_t m=atoi(argv[2]);double af=atof(argv[3]);uint32_t L1=atoi(argv[4]),L2=atoi(argv[5]);
  int nd=atoi(argv[6]);uint64_t sd=strtoull(argv[7],0,10);int HMAX=atoi(argv[8]); int RR=getenv("RR")?atoi(getenv("RR")):HMAX;
  uint32_t nmax=(uint32_t)((L2+1)*m/af)+8; j=malloc(4*(size_t)(nmax+1));x=malloc(nmax+1);y=malloc(nmax+1);z=malloc(nmax+1);
  char buf[256]; design d;
  if(argc>9){ d.P=0; char*tok=strtok(argv[9],","); while(tok){d.ty[d.P]=tok[0];d.par[d.P]=atoi(tok+1);d.P++;tok=strtok(0,",");}
    m-=m%d.P; show(&d,buf);
    for(uint32_t L=L1;L<=L2;L*=2){ printf("%s eps=%.3f m=%u L=%u:",buf,eps,m,L);
      for(int q=0;q<3;q++) printf(" %d",run(&d,eps,m,af,L,RR,sd+q,20000)); printf("\n"); fflush(stdout);}
    return 0;}
  if(nd<0){ // hill climbing: -nd steps per restart, objective = mean rounds at L2 over 2 seeds
    uint64_t hs[2]; seed(sd*104729+7); hs[0]=s[0];hs[1]=s[1];
    #define HR() (s[0]=hs[0],s[1]=hs[1])
    #define HS() (hs[0]=s[0],hs[1]=s[1])
    for(int rs=0;rs<6;rs++){
      HR(); int Ps[]={2,3,4,6,8,12}; d.P=Ps[below(6)];
      for(int c=0;c<d.P;c++){double u=uni(); if(u<0.4){d.ty[c]='S';d.par[c]=1+below(HMAX);} else if(u<0.75){d.ty[c]='F';d.par[c]=below(d.P);} else {d.ty[c]='A';d.par[c]=below(d.P);} }
      d.ty[0]='F'; d.par[0]=0; HS();
      uint32_t mm=m-m%d.P; double best=(run(&d,eps,mm,af,L2,RR,11,1500)+run(&d,eps,mm,af,L2,RR,12,1500))/2.0;
      for(int it=0;it<-nd;it++){ design e=d; HR(); int c=below(e.P); double u=uni();
        if(u<0.35){e.ty[c]='S';e.par[c]=1+below(HMAX);} else if(u<0.65){e.ty[c]='F';e.par[c]=below(e.P);} else if(u<0.9){e.ty[c]='A';e.par[c]=below(e.P);} else {e.ty[c]='Z';e.par[c]=0;}
        HS(); int hasF=0; for(int q=0;q<e.P;q++) if(e.ty[q]=='F'||e.ty[q]=='A') hasF=1; if(!hasF) continue;
        double v=(run(&e,eps,mm,af,L2,RR,11,1500)+run(&e,eps,mm,af,L2,RR,12,1500))/2.0;
        if(v>best){best=v;d=e;show(&d,buf);printf("restart %d step %d: %.1f  %s\n",rs,it,best,buf);fflush(stdout);} }
      show(&d,buf); printf("RESULT restart %d: %.1f  %s\n",rs,best,buf); fflush(stdout);}
    return 0;}
  uint64_t ds[2]; seed(sd*7919+13); ds[0]=s[0];ds[1]=s[1];
  for(int it=0;it<nd;it++){
    s[0]=ds[0];s[1]=ds[1];
    int Ps[]={2,3,4,4,5,6,6,8}; d.P=Ps[below(8)];
    int hasF=0;
    for(int c=0;c<d.P;c++){double u=uni();
      if(u<0.45){d.ty[c]='S';d.par[c]=1+below(HMAX);}
      else if(u<0.75){d.ty[c]='F';d.par[c]=below(d.P);hasF=1;}
      else if(u<0.95){d.ty[c]='A';d.par[c]=below(d.P);hasF=1;}
      else {d.ty[c]='Z';d.par[c]=0;}}
    ds[0]=s[0];ds[1]=s[1];
    if(!hasF) continue;
    uint32_t mm=m-m%d.P;
    int r1=run(&d,eps,mm,af,L1,RR,1000+it,400); if(r1<25) continue;
    int r2=run(&d,eps,mm,af,L2,RR,2000+it,1500);
    if(r2>=r1+12 && r2>=40){show(&d,buf);printf("%s  P=%d  L=%u:%d  L=%u:%d\n",buf,d.P,L1,r1,L2,r2);fflush(stdout);}
  }
  return 0;}
