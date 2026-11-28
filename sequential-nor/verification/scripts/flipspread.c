/* flipspread.c: how a flipped position spreads from layer to layer on a real smoothed instance.
   Zero gates, then G layers of H positions in the last 1/F of the instance. A gate of a layer
   lists a position of the layer below: the gates come in groups of m consecutive gates, and a
   group lists m consecutive positions p, p+1, ..., p+m-1 with p uniform in the layer below
   (m = 1: independent reads, a block base; m = H: the whole layer is a shifted copy).
   Two copies are evaluated with the same pointers; in layer W0 a fraction f of the positions of
   the second copy is flipped. Prints the number of positions that differ in every layer.
   usage: flipspread eps H G F m f seed */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static uint64_t st[2];
static inline uint64_t rnd(void){uint64_t a=st[0],b=st[1];st[0]=b;a^=a<<23;st[1]=a^b^(a>>17)^(b>>26);return st[1]+b;}
static inline double uni(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
int main(int argc,char**argv){
  if(argc<8){fprintf(stderr,"usage: flipspread eps H G F m f seed\n");return 1;}
  double eps=atof(argv[1]); long H=atol(argv[2]); int G=atoi(argv[3]); int F=atoi(argv[4]); long m=atol(argv[5]);
  double f=atof(argv[6]); uint64_t sd=strtoull(argv[7],0,10);
  long reg=H*G, n=reg*F, n0=n-reg; int W0=3;
  st[0]=sd*0x9E3779B97F4A7C15ULL+1; st[1]=sd^0xD1B54A32D192ED03ULL; for(int i=0;i<20;i++) rnd();
  uint8_t *x=malloc(n), *y=malloc(n); if(!x||!y){fprintf(stderr,"memory\n");return 1;}
  x[0]=y[0]=1; long *D=calloc(G,sizeof(long)); long ones=0; long p=0;
  for(long i=1;i<n;i++){
    long t;
    if(i<n0) t=0;
    else { long o=(i-n0)%H, w=(i-n0)/H; long lo=n0+(w-1)*H;
      if(o%m==0){ long span=H-m+1; if(span<1) span=1; p=lo+(long)(uni()*span); }
      t=p+(o%m); if(t>=lo+H) t=lo+H-1; }
    if(uni()<eps) t=(long)(uni()*i);
    x[i]=!(x[i-1]|x[t]); y[i]=!(y[i-1]|y[t]);
    if(i>=n0){ long w=(i-n0)/H; if(w==W0 && uni()<f) y[i]^=1; if(x[i]!=y[i]) D[w]++; ones+=x[i]; }
  }
  printf("eps=%.3f H=%ld G=%d m=%ld f=%.4f seed=%llu density of ones in the layers %.4f\n",eps,H,G,m,f,(unsigned long long)sd,(double)ones/reg);
  for(int w=W0;w<G;w++) printf("%ld ",D[w]); printf("\n");
  return 0;
}
