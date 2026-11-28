// two class-preserving distances h0,h1 (period 2): upper row U reads lower row L0 at even steps and L1 at odd steps.
// L0 reads row A at even steps and row B at odd steps; L1 reads row B at even steps and row C at odd steps.
// ratio = E[#open in U] / E[#open at the lower positions that U reads], stationary, per period.
// usage: per2 eps d T mode   (mode 0: exhaustive over periodic inputs of period T; mode 1: hill climbing, random restarts)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
static double eps,d; static int T;
static void gate(int pv,int po,int b,int ob,int*x,int*o){ *x=!(pv|b); if(!pv&&b)*o=ob; else if(pv&&b)*o=po&&ob; else if(!pv&&!b)*o=po||ob; else *o=po; }
// state: bits 0,1 = L0 (x,o); 2,3 = L1; 4,5 = U
static int nxt[2][4][4][64][27]; static double pr[27];
static void init(void){ double p[3]={1-eps,eps*d,eps*(1-d)};
  for(int a=0;a<3;a++)for(int b=0;b<3;b++)for(int c=0;c<3;c++) pr[a*9+b*3+c]=p[a]*p[b]*p[c];
  for(int par=0;par<2;par++)for(int s0=0;s0<4;s0++)for(int s1=0;s1<4;s1++)for(int s=0;s<64;s++){
    int x0=s&1,o0=(s>>1)&1,x1=(s>>2)&1,o1=(s>>3)&1,xu=(s>>4)&1,ou=(s>>5)&1;
    for(int a=0;a<3;a++)for(int b=0;b<3;b++)for(int c=0;c<3;c++){
      int r0[3]={s0&1,1,0},q0[3]={s0>>1,0,0}, r1[3]={s1&1,1,0},q1[3]={s1>>1,0,0};
      int nx0,no0,nx1,no1; gate(x0,o0,r0[a],q0[a],&nx0,&no0); gate(x1,o1,r1[b],q1[b],&nx1,&no1);
      int lx=par?nx1:nx0, lo=par?no1:no0; int ru[3]={lx,1,0},qu[3]={lo,0,0}; int nxu,nou; gate(xu,ou,ru[c],qu[c],&nxu,&nou);
      nxt[par][s0][s1][s][a*9+b*3+c]=nx0|no0<<1|nx1<<2|no1<<3|nxu<<4|nou<<5; } } }
// word: w[j] = symbol of L0 (low 2 bits) and of L1 (high 2 bits) at step j
static double ratio(const int*w,double*N1){ double p[64]={0},q[64]; p[0]=1; double n1=0,n2=0;
  for(int rep=0;rep<30;rep++){ n1=n2=0; for(int j=0;j<T;j++){ int par=j&1,s0=w[j]&3,s1=w[j]>>2; memset(q,0,sizeof q);
      for(int s=0;s<64;s++) if(p[s]>0){ const int*nx=nxt[par][s0][s1][s]; for(int k=0;k<27;k++) q[nx[k]]+=p[s]*pr[k]; }
      memcpy(p,q,sizeof p); for(int s=0;s<64;s++){ if(s&32) n2+=p[s]; if(par?(s&8):(s&2)) n1+=p[s]; } } }
  if(N1)*N1=n1; return n1>1e-12?n2/n1:0; }
// row B: at even j the symbol of L1, at odd j the symbol of L0; consecutive positions: no two ones
static int valid(const int*w){ for(int j=0;j<T;j++){ int bj=(j&1)?(w[j]&3):(w[j]>>2); int k=(j+1)%T; int bk=(k&1)?(w[k]&3):(w[k]>>2); if((bj&1)&&(bk&1)) return 0; } return 1; }
static void show(const int*w){ const char*c="01oI"; for(int j=0;j<T;j++) printf("%c%c ",c[w[j]&3],c[w[j]>>2]); }
int main(int argc,char**argv){ eps=atof(argv[1]); d=atof(argv[2]); T=atoi(argv[3]); int mode=atoi(argv[4]); init(); int w[64],bw[64]; double best=0;
  if(mode==0){ long N=1; for(int j=0;j<T;j++)N*=16; for(long c=0;c<N;c++){ long cc=c; int any=0; for(int j=0;j<T;j++){ w[j]=cc&15; cc>>=4; if((w[j]&2)||(w[j]&8)) any=1; } if(!any||!valid(w)) continue; double r=ratio(w,NULL); if(r>best){best=r;memcpy(bw,w,sizeof w);} } }
  else { srand(atoi(argv[5])); for(int rs=0;rs<mode;rs++){ do{ for(int j=0;j<T;j++) w[j]=rand()&15; }while(!valid(w)); double cur=ratio(w,NULL);
      for(int it=0;it<400;it++){ int j=rand()%T, old=w[j]; w[j]=rand()&15; double n1; double r=valid(w)?ratio(w,&n1):0; if(r>cur&&n1>1e-6) cur=r; else w[j]=old; }
      if(cur>best){best=cur;memcpy(bw,w,sizeof w);} } }
  printf("eps %.3f d %.3f T %d: max ratio %.5f  word (L0 L1 per step): ",eps,d,T,best); show(bw); printf("\n"); return 0; }
