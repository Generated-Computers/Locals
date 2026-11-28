/* One layered periodic design (as in slowsearch.c): pruned depth D and the number of rounds of the
   prefix-Jacobi iteration (two-valued; every pointer read uses the previous iterate) and of the
   table-Jacobi iteration with reach R (pointers of reach <= R read the current iterate).
   usage: layered eps T layers N seed R d0 d1 ...   (digits as in slowsearch.c) */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
static uint64_t st[2];
static inline uint64_t rnd(void){uint64_t a=st[0],b=st[1];st[0]=b;a^=a<<23;st[1]=a^b^(a>>17)^(b>>26);return st[1]+b;}
static inline double uni(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
int main(int argc,char**argv){
  double eps=atof(argv[1]);int T=atoi(argv[2]),LY=atoi(argv[3]),N=atoi(argv[4]);uint64_t sd=strtoull(argv[5],0,10);int R=atoi(argv[6]);
  int des[8];for(int i=0;i<T;i++)des[i]=atoi(argv[7+i]); N-=N%T;
  double vp=((2*eps+1)-sqrt(4*eps+1))/(2*eps);
  st[0]=sd*0x9E3779B97F4A7C15ULL+1;st[1]=sd^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();
  long n=(long)(LY+1)*N; int32_t *tg=malloc(4*n),*D=malloc(4*n); uint8_t *x=malloc(n),*y=calloc(n,1),*z=malloc(n);
  /* tg[k] >= 0: position read; -1: constant 0; -2: constant 1 */
  int prevx=0;int32_t prevD=0,mD=0;
  for(int l=0;l<=LY;l++)for(int o=0;o<N;o++){long k=(long)l*N+o;int ty=(l==0)?T+1:des[o%T];int32_t t;
    if(uni()<eps) t=(uni()<vp)?-2:-1;
    else if(ty<T){int per=(int)(uni()*(N/T)); t=(l-1)*N+per*T+ty;}
    else if(ty==T) t=-1; else if(ty==T+1) t=-2; else {int d=ty-T; t=(o>=d)?(int32_t)(k-d):-1;}
    tg[k]=t; int xj=t>=0?x[t]:(t==-2); int32_t Dj=t>=0?D[t]:0;
    int rel=!prevx,reset=xj; int xv=!(prevx|xj); int32_t Dv;
    if(reset&&rel)Dv=1+Dj;else if(reset)Dv=prevD<1+Dj?prevD:1+Dj;else if(rel)Dv=prevD>1+Dj?prevD:1+Dj;else Dv=prevD;
    x[k]=xv;D[k]=Dv;prevx=xv;prevD=Dv;if(Dv>mD)mD=Dv;}
  int rounds;
  for(rounds=1;rounds<=100000;rounds++){int pz=0;
    for(long k=0;k<n;k++){int32_t t=tg[k];int r; if(t<0)r=(t==-2); else if(k-t<=R)r=z[t]; else r=y[t]; int v=!(pz|r); z[k]=v; pz=v;}
    if(memcmp(y,z,n)==0)break; uint8_t*w=y;y=z;z=w;}
  int ok=(memcmp(y,x,n)==0);
  printf("eps=%.3f layers=%d N=%d design:",eps,LY,N);for(int i=0;i<T;i++){int ty=des[i];if(ty<T)printf(" E%d",ty);else if(ty==T)printf(" Z0");else if(ty==T+1)printf(" Z1");else printf(" L%d",ty-T);}
  printf(" | max D=%d | Jacobi with reach %d: %d rounds%s\n",mD,R,rounds,ok?"":" (WRONG FIXED POINT)");
  return 0;}
