/* Like slowsearch.c, but the score of a design is the number of rounds of the table-Jacobi
   iteration with reach 5 (all local reads are exact; only the reads of the layer below are
   iterated), capped at the number of layers.
   usage: tjsearch eps T layers N seed top */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
static uint64_t st[2];
static inline uint64_t rnd(void){uint64_t a=st[0],b=st[1];st[0]=b;a^=a<<23;st[1]=a^b^(a>>17)^(b>>26);return st[1]+b;}
static inline double uni(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static double eps,vp; static int T,LY,N; static long n;
static int32_t *tg; static uint8_t *y,*z;
static int run(const int*des){
  for(int l=0;l<=LY;l++)for(int o=0;o<N;o++){long k=(long)l*N+o;int ty=(l==0)?T+1:des[o%T];int32_t t;
    if(uni()<eps) t=(uni()<vp)?-2:-1;
    else if(ty<T){int per=(int)(uni()*(N/T)); t=(l-1)*N+per*T+ty;}
    else if(ty==T) t=-1; else if(ty==T+1) t=-2; else {int d=ty-T; t=(o>=d)?(int32_t)(k-d):-1;}
    tg[k]=t;}
  memset(y,0,n); int rounds;
  for(rounds=1;rounds<=LY;rounds++){int pz=0;
    for(long k=0;k<n;k++){int32_t t=tg[k];int r; if(t<0)r=(t==-2); else if(k-t<=5)r=z[t]; else r=y[t]; int v=!(pz|r); z[k]=v; pz=v;}
    if(memcmp(y,z,n)==0)break; uint8_t*w=y;y=z;z=w;}
  return rounds;
}
typedef struct{int sc;long code;}res_t;
static int cmp(const void*a,const void*b){return ((const res_t*)b)->sc-((const res_t*)a)->sc;}
int main(int argc,char**argv){
  eps=atof(argv[1]);T=atoi(argv[2]);LY=atoi(argv[3]);N=atoi(argv[4]);uint64_t sd=strtoull(argv[5],0,10);int top=atoi(argv[6]);
  N-=N%T; n=(long)(LY+1)*N; vp=((2*eps+1)-sqrt(4*eps+1))/(2*eps);
  tg=malloc(4*n);y=malloc(n);z=malloc(n);
  int B=T+6; long tot=1;for(int i=0;i<T;i++)tot*=B; res_t*res=malloc(sizeof(res_t)*tot); long cnt=0; int des[8];
  for(long code=0;code<tot;code++){long c=code;int hasE=0;for(int i=0;i<T;i++){des[i]=c%B;c/=B;if(des[i]<T)hasE=1;} if(!hasE)continue;
    st[0]=sd*0x9E3779B97F4A7C15ULL+1;st[1]=sd^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();
    res[cnt].sc=run(des);res[cnt].code=code;cnt++;}
  qsort(res,cnt,sizeof(res_t),cmp);
  printf("eps=%.3f T=%d layers=%d N=%d: %ld designs; most rounds of table-Jacobi with reach 5 (cap %d)\n",eps,T,LY,N,cnt,LY);
  for(int i=0;i<top&&i<cnt;i++){long c=res[i].code;printf("%4d  ",res[i].sc);for(int k=0;k<T;k++){int ty=c%B;c/=B;if(ty<T)printf("E%d ",ty);else if(ty==T)printf("Z0 ");else if(ty==T+1)printf("Z1 ");else printf("L%d ",ty-T);}printf("\n");}
  return 0;}
