/* Search for a layered base on which the relevance-pruned iteration is slow.
   A base consists of layers; every layer is a chain of N gates with a periodic program of period T.
   The gate at slot o of a period has one of the read types
     E_t  : a uniformly random position of slot t in the layer below (t = 0..T-1),
     Z0   : position 1 (holds 0), Z1 : position 0 (holds 1),
     L_d  : the position d before it in the same layer (d = 2..5).
   Every read is resampled with probability eps; a resampled read returns a one with probability
   varpi(eps) and its target has depth 0 (as if it lay in a large region of zero gates).
   Layer 0 consists of zero gates. The depth D is computed by its recursion.
   Score of a design: the fraction of positions of the top layers (the top sixth) with D > layers/2.
   usage: slowsearch eps T layers N seed top          (exhaustive over the (T+6)^T designs)
          slowsearch eps T layers N seed 0 d0 d1 ...   (one design, digits: 0..T-1 = E_t, T = Z0, T+1 = Z1, T+2.. = L2..L5) */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
static uint64_t st[2];
static inline uint64_t rnd(void){uint64_t a=st[0],b=st[1];st[0]=b;a^=a<<23;st[1]=a^b^(a>>17)^(b>>26);return st[1]+b;}
static inline double uni(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static double eps,vp; static int T,LY,N;
static uint8_t *x; static int32_t *D;
static void seed(uint64_t s){st[0]=s*0x9E3779B97F4A7C15ULL+1;st[1]=s^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();}
static double run(const int *des,double *meanD,double *dens){
  /* positions: layer l occupies [l*N, (l+1)*N); position before layer 0 holds 0, depth 0 */
  int prevx=0; int32_t prevD=0;
  for(int l=0;l<=LY;l++){
    for(int o=0;o<N;o++){
      int k=l*N+o; int ty = (l==0)? T+1 : des[o%T];
      int xj; int32_t Dj;
      if(uni()<eps){ xj=(uni()<vp); Dj=0; }
      else if(ty<T){ int per=(int)(uni()*(N/T)); int p=(l-1)*N+per*T+ty; xj=x[p]; Dj=D[p]; }
      else if(ty==T){ xj=0; Dj=0; }
      else if(ty==T+1){ xj=1; Dj=0; }
      else { int d=ty-T; if(o>=d){ xj=x[k-d]; Dj=D[k-d]; } else { xj=0; Dj=0; } }
      int rel=!prevx, reset=xj; int xv=!(prevx|xj); int32_t Dv;
      if(reset&&rel) Dv=1+Dj; else if(reset) Dv=prevD<1+Dj?prevD:1+Dj;
      else if(rel) Dv=prevD>1+Dj?prevD:1+Dj; else Dv=prevD;
      x[k]=xv; D[k]=Dv; prevx=xv; prevD=Dv;
    }
  }
  int top=LY/6; if(top<1)top=1; long cnt=0,tot=0; double sD=0,so=0;
  for(int l=LY-top+1;l<=LY;l++)for(int o=0;o<N;o++){int k=l*N+o; tot++; if(D[k]>LY/2)cnt++; sD+=D[k]; so+=x[k];}
  if(meanD)*meanD=sD/tot; if(dens)*dens=so/tot;
  return (double)cnt/tot;
}
typedef struct{double sc,mD;long code;}res_t;
static int cmp(const void*a,const void*b){double p=((const res_t*)a)->sc,q=((const res_t*)b)->sc;if(p!=q)return p<q?1:-1;double u=((const res_t*)a)->mD,v=((const res_t*)b)->mD;return u<v?1:u>v?-1:0;}
static void show(const int*des){for(int i=0;i<T;i++){int ty=des[i]; if(ty<T)printf("E%d ",ty);else if(ty==T)printf("Z0 ");else if(ty==T+1)printf("Z1 ");else printf("L%d ",ty-T);} }
int main(int argc,char**argv){
  if(argc<7){fprintf(stderr,"usage: slowsearch eps T layers N seed top [design digits]\n");return 1;}
  eps=atof(argv[1]);T=atoi(argv[2]);LY=atoi(argv[3]);N=atoi(argv[4]);uint64_t sd=strtoull(argv[5],0,10);int top=atoi(argv[6]);
  N-=N%T; vp=((2*eps+1)-sqrt(4*eps+1))/(2*eps);
  x=malloc((size_t)(LY+1)*N);D=malloc(sizeof(int32_t)*(size_t)(LY+1)*N);
  int des[8];
  if(top==0){for(int i=0;i<T;i++)des[i]=atoi(argv[7+i]); seed(sd); double mD,dn; double sc=run(des,&mD,&dn);
    printf("eps=%.3f layers=%d N=%d design: ",eps,LY,N);show(des);printf(" open fraction=%.4f mean D=%.1f density=%.3f\n",sc,mD,dn);return 0;}
  int B=T+6; long tot=1;for(int i=0;i<T;i++)tot*=B;
  res_t *res=malloc(sizeof(res_t)*tot); long n=0;
  for(long code=0;code<tot;code++){long c=code;int hasE=0;for(int i=0;i<T;i++){des[i]=c%B;c/=B;if(des[i]<T)hasE=1;}
    if(!hasE)continue; seed(sd); double mD; double sc=run(des,&mD,0); res[n].sc=sc;res[n].mD=mD;res[n].code=code;n++;}
  qsort(res,n,sizeof(res_t),cmp);
  printf("eps=%.3f T=%d layers=%d N=%d: %ld designs; top %d by open fraction (then mean D of the top layers)\n",eps,T,LY,N,n,top);
  for(int i=0;i<top&&i<n;i++){long c=res[i].code;for(int k=0;k<T;k++){des[k]=c%B;c/=B;} printf("%.4f  meanD=%6.1f  ",res[i].sc,res[i].mD);show(des);printf("\n");}
  return 0;
}
