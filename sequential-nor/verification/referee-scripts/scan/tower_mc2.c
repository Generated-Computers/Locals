/* Referee's Monte Carlo of a real alternating tower for the "window designs" of Remark rem:scan(iii).
   Plain layer: gate h lists a uniformly random position L[h] of the scanning layer below.
   Scanning layer: every period draws a uniformly random start for each window; a gate P(i,o) reads the
   value at position start_i+o of the plain layer below, a gate T(i,o) reads the position L[start_i+o] of the
   scanning layer two below.  Reads are resampled with probability eps.  Two copies, same choices; a fraction
   delta of a scanning layer is flipped; differing positions of the scanning layer two above are counted.
   usage: tower_mc2 eps design M delta trials seed       design like 2,1:P01,T10,T00 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static unsigned long long s;
static inline unsigned long long nxt(void){ unsigned long long z=(s+=0x9E3779B97F4A7C15ULL); z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
static inline double U01(void){ return (nxt()>>11)*(1.0/9007199254740992.0); }
int main(int argc,char**argv){
  double eps=atof(argv[1]); long M=atol(argv[3]); double delta=atof(argv[4]); int trials=atoi(argv[5]); s=strtoull(argv[6],0,10);
  char buf[256]; strncpy(buf,argv[2],255); char*col=strchr(buf,':'); *col=0;
  int nw=0,T=0; char kind[16]; int win[16],off[16];
  for(char*t=strtok(buf,",");t;t=strtok(NULL,",")) nw++;
  for(char*t=strtok(col+1,",");t;t=strtok(NULL,",")){ kind[T]=t[0]; win[T]=t[1]-'0'; off[T]=t[2]-'0'; T++; }
  double vp=((2*eps+1)-sqrt(4*eps+1))/(2*eps);
  unsigned char *E=malloc(M),*E2=malloc(M),*P=malloc(M),*P2=malloc(M),*N=malloc(M); int*L=malloc(M*sizeof(int));
  for(long i=0;i<M;i++) E[i]=U01()<0.4;
  double sum=0,sum2=0,dens=0; long nt=0;
  for(int tr=-30;tr<trials;tr++){
    long flips=0; memcpy(E2,E,M);
    if(tr>=0){ long nf=(long)(delta*M); for(long f=0;f<nf;f++){ long i=(long)(U01()*M); if(E2[i]==E[i]){ E2[i]=!E[i]; flips++; } } }
    int x=0,x2=0;
    for(long g=0;g<M;g++){ double u=U01(); int y,y2; long i=(long)(U01()*M); L[g]=(int)i;
      if(u<eps){ y=y2=(U01()<vp); } else { y=E[i]; y2=E2[i]; }
      x=!(x||y); x2=!(x2||y2); P[g]=x; P2[g]=x2; }
    x=0; x2=0; long st[4]={0,0,0,0}, diff=0, ones=0;
    for(long g=0;g<M;g++){ int ph=g%T; if(ph==0) for(int i=0;i<nw;i++) st[i]=(long)(U01()*M);
      long pos=st[win[ph]]+off[ph]; if(pos>=M) pos-=M;
      double u=U01(); int y,y2;
      if(u<eps){ y=y2=(U01()<vp); } else if(kind[ph]=='P'){ y=P[pos]; y2=P2[pos]; } else { y=E[L[pos]]; y2=E2[L[pos]]; }
      x=!(x||y); x2=!(x2||y2); N[g]=x; diff+=(x!=x2); ones+=x; }
    if(tr>=0){ double r=(double)diff/flips; sum+=r; sum2+=r*r; dens+=(double)ones/M; nt++; }
    unsigned char*t=E; E=N; N=t;
  }
  double m=sum/nt, sd=sqrt((sum2/nt-m*m)/(nt-1));
  printf("eps %.3f design %s M %ld delta %g trials %ld seed %s: growth %.5f +- %.5f, density %.5f\n",eps,argv[2],M,delta,nt,argv[6],m,sd,dens/nt);
  return 0; }
