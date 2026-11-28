/* Referee's single-flip Monte Carlo for the window designs (infinite-width limit, no flipped fraction).
   One plain gate (position 0 of the plain layer) lists the flipped position: its target is 0 in copy a, 1 in copy b.
   The plain chain around it is simulated; for every window i of the program and every start s for which the window
   meets a differing position, one period of the scanning chain that draws this window at s is simulated (weight 1/T,
   the expected number of such periods), after a warm-up of clean periods and followed by clean periods until the
   copies agree.  The density d of the scanning layer two below is an input (take it from the tower or the fixed point).
   usage: local_mc eps design d samples seed */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static unsigned long long s;
static inline unsigned long long nxt(void){ unsigned long long z=(s+=0x9E3779B97F4A7C15ULL); z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
static inline double U01(void){ return (nxt()>>11)*(1.0/9007199254740992.0); }
static double eps,vp,d,alpha;
static int nw,T,wd[4]; static char kind[16]; static int win[16],off[16];
/* a clean stationary window: values v[], targets t[] */
static void clean_win(int w,int*v,int*t){ int x=!(U01()<alpha);
  for(int i=0;i<w;i++){ int tg=U01()<d; int y=(U01()<eps)?(U01()<vp):tg; x=!(x||y); v[i]=x; t[i]=tg; } }
#define WMAX 4
#define LMAX 400
int main(int argc,char**argv){
  eps=atof(argv[1]); d=atof(argv[3]); long n=atol(argv[4]); s=strtoull(argv[5],0,10);
  char buf[256]; strncpy(buf,argv[2],255); char*col=strchr(buf,':'); *col=0;
  for(char*t=strtok(buf,",");t;t=strtok(NULL,",")) wd[nw++]=atoi(t);
  for(char*t=strtok(col+1,",");t;t=strtok(NULL,",")){ kind[T]=t[0]; win[T]=t[1]-'0'; off[T]=t[2]-'0'; T++; }
  vp=((2*eps+1)-sqrt(4*eps+1))/(2*eps); double A=eps*vp,B=eps*(1-vp),c=1-eps; double q=B+c*(1-d); alpha=1/(1+q);
  double sum=0,sum2=0;
  for(long it=0;it<n;it++){
    /* plain chain, positions -WMAX .. ; index p+WMAX */
    static int va[LMAX],vb[LMAX],ta[LMAX],tb[LMAX];
    int xa=!(U01()<alpha), xb=xa, L=0, len=0, tail=0;
    for(int p=-WMAX; p<LMAX-WMAX; p++){ int i=p+WMAX; int tg=U01()<d, t1=tg,t2=tg; if(p==0){ t1=0; t2=1; }
      int ya,yb; if(U01()<eps){ ya=yb=(U01()<vp); } else { ya=t1; yb=t2; }
      xa=!(xa||ya); xb=!(xb||yb); va[i]=xa; vb[i]=xb; ta[i]=t1; tb[i]=t2; len=i+1;
      if(p>=0){ if(xa!=xb && tail==0) L=p+1; else { tail++; if(tail>=WMAX) break; } } }
    /* differing positions: 0..L-1 (values), and the target at 0 */
    int last=L>0?L-1:0; double tot=0;
    for(int iw=0;iw<nw;iw++) for(int st=-(wd[iw]-1); st<=last; st++){
      int x=0,y=0;   /* chain copies */
      int cv[4][WMAX],ct[4][WMAX];
      /* warm-up: 12 clean periods */
      for(int per=0;per<12;per++){ for(int i=0;i<nw;i++) clean_win(wd[i],cv[i],ct[i]);
        for(int g=0;g<T;g++){ int r=kind[g]=='P'?cv[win[g]][off[g]]:ct[win[g]][off[g]]; if(U01()<eps) r=(U01()<vp); x=!(x||r); } }
      y=x; double cnt=0;
      /* the dirty period */
      for(int i=0;i<nw;i++) if(i!=iw) clean_win(wd[i],cv[i],ct[i]);
      for(int g=0;g<T;g++){ int ra,rb;
        if(win[g]==iw){ int i=st+off[g]+WMAX; if(kind[g]=='P'){ ra=va[i]; rb=vb[i]; } else { ra=ta[i]; rb=tb[i]; } }
        else { ra=rb=(kind[g]=='P'?cv[win[g]][off[g]]:ct[win[g]][off[g]]); }
        if(U01()<eps){ ra=rb=(U01()<vp); }
        x=!(x||ra); y=!(y||rb); cnt+=(x!=y); }
      while(x!=y){ for(int i=0;i<nw;i++) clean_win(wd[i],cv[i],ct[i]);
        for(int g=0;g<T&&x!=y;g++){ int r=kind[g]=='P'?cv[win[g]][off[g]]:ct[win[g]][off[g]]; if(U01()<eps) r=(U01()<vp); x=!(x||r); y=!(y||r); cnt+=(x!=y); } }
      tot+=cnt; }
    tot/=T; sum+=tot; sum2+=tot*tot; (void)len; }
  double m=sum/n, se=sqrt((sum2/n-m*m)/(n-1));
  printf("local MC eps %.3f design %s d %.6f samples %ld: G2 = %.5f +- %.5f\n",eps,argv[2],d,n,m,se);
  return 0; }
