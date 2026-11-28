/* Referee's Monte Carlo of a real alternating tower: scanning layers (word W) and plain layers,
   M positions each, two copies with the same choices; a fraction delta of a scanning layer is
   flipped and the differing positions of the scanning layer two above are counted.
   usage: tower_mc eps word M delta trials seed */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static unsigned long long s;
static inline unsigned long long nxt(void){ unsigned long long z=(s+=0x9E3779B97F4A7C15ULL); z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
static inline double U01(void){ return (nxt()>>11)*(1.0/9007199254740992.0); }
int main(int argc,char**argv){
  double eps=atof(argv[1]); const char*W=argv[2]; int T=strlen(W); long M=atol(argv[3]); double delta=atof(argv[4]); int trials=atoi(argv[5]); s=strtoull(argv[6],0,10);
  double vp=((2*eps+1)-sqrt(4*eps+1))/(2*eps);
  unsigned char *E=malloc(M),*E2=malloc(M),*P=malloc(M),*P2=malloc(M),*N=malloc(M),*N2=malloc(M);
  for(long i=0;i<M;i++) E[i]=U01()<0.4;
  double sum=0,sum2=0,dens=0; long nt=0;
  for(int tr=-30;tr<trials;tr++){
    long flips=0; memcpy(E2,E,M);
    if(tr>=0){ long nf=(long)(delta*M); for(long f=0;f<nf;f++){ long i=(long)(U01()*M); if(E2[i]==E[i]){ E2[i]=!E[i]; flips++; } } }
    /* plain layer */
    int x=0,x2=0;
    for(long g=0;g<M;g++){ double u=U01(); int y,y2;
      if(u<eps){ y=y2=(U01()<vp); } else { long i=(long)(U01()*M); y=E[i]; y2=E2[i]; }
      x=!(x||y); x2=!(x2||y2); P[g]=x; P2[g]=x2; }
    /* scanning layer */
    x=0; x2=0; long pos=0, diff=0, ones=0;
    for(long g=0;g<M;g++){ if(W[g%T]=='U') pos=(long)(U01()*M); else { pos++; if(pos>=M) pos=0; }
      double u=U01(); int y,y2;
      if(u<eps){ y=y2=(U01()<vp); } else { y=P[pos]; y2=P2[pos]; }
      x=!(x||y); x2=!(x2||y2); N[g]=x; N2[g]=x2; diff+=(x!=x2); ones+=x; }
    if(tr>=0){ double r=(double)diff/flips; sum+=r; sum2+=r*r; dens+=(double)ones/M; nt++; }
    unsigned char*t=E; E=N; N=t;
  }
  double m=sum/nt, sd=sqrt((sum2/nt-m*m)/(nt-1));
  printf("eps %.3f word %s M %ld delta %g trials %ld: growth %.5f +- %.5f, density %.5f\n",eps,W,M,delta,nt,m,sd,dens/nt);
  return 0; }
