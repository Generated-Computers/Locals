/* twoback_layered.c -- rounds of the relevance-pruned iteration (max D) and of the iteration with
   lower and upper traces for pointers two back, on one smoothing of a layered base with wide bundles.

   The base: n0 zero gates, then N bundles of m chains; a chain has T*h gates and runs a program
   given as a string of T letters:  E = a uniformly random copy of the output class of the bundle
   below (the first bundle reads position 1),  L = the position two before,  N = the position before,
   Z = position 0,  O = position 1.  The output class of a bundle is the gates of the last letter's
   phase in the last period of its chains.  Smoothing: every pointer is replaced with probability
   eps by a uniformly random earlier position.

   usage: twoback_layered eps program h m N prefix_fraction seed
   prints n, max D, the number of rounds of the two-back traces, and the number of rounds with the
   rule for pointers two back alone (without the standard rule).                              */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static uint64_t S[2];
static inline uint64_t rnd(void){uint64_t s1=S[0],s0=S[1];S[0]=s0;s1^=s1<<23;S[1]=s1^s0^(s1>>17)^(s0>>26);return S[1]+s0;}
static inline double unif(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static long n; static int *j; static unsigned char *x,*L,*U,*Lp,*Up;
static int rounds(int meet){
  for(long i=1;i<=n;i++){L[i]=0;U[i]=1;} L[0]=U[0]=1; int s=0;
  for(;;){ long open=0; for(long i=1;i<=n;i++) open+=(L[i]!=U[i]); if(!open) return s;
    memcpy(Lp,L,n+1); memcpy(Up,U,n+1); s++;
    for(long k=1;k<=n;k++){ long p=j[k]; int l,u;
      if(p==k-1){ l=1-U[k-1]; u=1-L[k-1]; } else { l=(1-U[k-1])&(1-Up[p]); u=(1-L[k-1])&(1-Lp[p]); }
      if(p==k-2){ long q=j[k-1]; int lt,ut;
        if(q==k-2){ lt=0; ut=0; }
        else { lt=(1-U[k-2])&Lp[q]; ut=(1-L[k-2])&Up[q];
               if(q==k-3){ if(L[k-3]>lt) lt=L[k-3]; if(U[k-3]<ut) ut=U[k-3]; } }
        if(meet){ if(lt>l) l=lt; if(ut<u) u=ut; } else { l=lt; u=ut; } }
      if(l>x[k]||u<x[k]){ fprintf(stderr,"unsound at %ld\n",k); exit(2); }
      L[k]=l; U[k]=u; }
    if(s>100000) return -1; }
}
int main(int argc,char**argv){
  if(argc<8){ fprintf(stderr,"usage: twoback_layered eps program h m N prefix_fraction seed\n"); return 1; }
  double eps=atof(argv[1]); const char*prog=argv[2]; int T=strlen(prog),h=atoi(argv[3]),m=atoi(argv[4]),N=atoi(argv[5]);
  double pf=atof(argv[6]); uint64_t sd=atoll(argv[7]); S[0]=sd*0x9E3779B97F4A7C15ULL+1;S[1]=sd^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();
  long body=(long)N*m*T*h; long n0=(long)(pf/(1-pf)*body); if(n0<4) n0=4; n=n0+body;
  j=malloc((n+1)*sizeof(int)); x=malloc(n+1); L=malloc(n+1); U=malloc(n+1); Lp=malloc(n+1); Up=malloc(n+1);
  long *out=malloc(sizeof(long)*m), *outp=malloc(sizeof(long)*m);
  j[0]=0; for(long i=1;i<=n0;i++) j[i]=0;
  long pos=n0;
  for(int b=0;b<N;b++){
    for(int c=0;c<m;c++) for(int r=0;r<T*h;r++){ pos++; char t=prog[r%T];
      if(t=='E') j[pos]= b? (int)outp[rnd()%m] : 1;
      else if(t=='L') j[pos]=pos-2; else if(t=='N') j[pos]=pos-1; else if(t=='Z') j[pos]=0; else j[pos]=1;
      if(r==T*h-1) out[c]=pos; }
    memcpy(outp,out,sizeof(long)*m); }
  for(long i=1;i<=n;i++) if(unif()<eps) j[i]=(int)(rnd()%i);
  x[0]=1; for(long i=1;i<=n;i++) x[i]=!(x[i-1]||x[j[i]]);
  /* pruned depth */
  int *D=malloc((n+1)*sizeof(int)); D[0]=0; int maxD=0;
  for(long k=1;k<=n;k++){ long p=j[k]; int notg=(p==k-1), reset=(!notg)&&x[p]==1, rel=(x[k-1]==0);
    if(k==1) D[k]=1; else if(reset&&rel) D[k]=1+D[p]; else if(reset) D[k]=D[k-1]<1+D[p]?D[k-1]:1+D[p];
    else if(rel&&!notg) D[k]=D[k-1]>1+D[p]?D[k-1]:1+D[p]; else D[k]=D[k-1];
    if(D[k]>maxD) maxD=D[k]; }
  int r1=rounds(1), r2=rounds(0);
  printf("eps %.3f program %s h %d m %d N %d n %ld : max D %d   two-back rounds %d   (rewritten rule alone %d)\n",eps,prog,h,m,N,n,maxD,r1,r2);
  return 0;
}
