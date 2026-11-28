/* latch_local_search.c: the search of Remark "what this says about the range between 4/9 and 3/4".
   In the ideal model of the section on the latch embedding (density of ones varpi(eps), independent
   reads, infinitely long chains) a chain is periodic with period T, and every gate has one of the
   read types P (own rail), Q (other rail), 0 (position 1, which holds 0), 1 (position 0, which holds 1),
   L2..L5 (the position 2..5 before it in the same chain). Every read is resampled with probability
   eps and then returns a one with probability varpi. The copy is the last gate of a period. For
   every pattern with at least one P and one Q the program finds the diagonal fixed point pi = p(pi,pi)
   and the multiplier of the difference of the two rails there, and prints the patterns with the
   largest multipliers. The plain latch P Q has the largest one.
   usage: latch_local_search eps Tmax top      (Tmax = 7 takes about two minutes) */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define R 5
#define NS (1<<R)
static double eps, vp;
static const char *names[8]={"P","Q","0","1","L2","L3","L4","L5"};
static double copyprob(const int *pat,int T,double xP,double xQ){
  double a[NS],b[NS]; for(int i=0;i<NS;i++)a[i]=1.0/NS;
  double A=eps*vp,c=1-eps,out=0;
  for(int per=0;per<60;per++){
    for(int t=0;t<T;t++){
      for(int i=0;i<NS;i++)b[i]=0;
      double p1=0;
      for(int st=0;st<NS;st++){ double w=a[st]; if(w==0)continue;
        double rho; int ty=pat[t];
        if(ty==0)rho=A+c*xP; else if(ty==1)rho=A+c*xQ; else if(ty==2)rho=A; else if(ty==3)rho=A+c;
        else { int d=ty-2; rho=A+c*((st>>(d-1))&1); }
        double q1=((st&1)?0.0:(1-rho));
        int s1=((st<<1)|1)&(NS-1), s0=(st<<1)&(NS-1);
        b[s1]+=w*q1; b[s0]+=w*(1-q1); p1+=w*q1;
      }
      memcpy(a,b,sizeof a);
      if(t==T-1) out=p1;
    }
  }
  return out;
}
static double diag(const int *pat,int T){double p=0.4;for(int it=0;it<80;it++){double q=copyprob(pat,T,p,p); if(fabs(q-p)<1e-10){p=q;break;} p=q;} return p;}
typedef struct{double lam,pi;int pat[8];int T;}res_t;
static int cmp(const void*a,const void*b){double x=((const res_t*)a)->lam,y=((const res_t*)b)->lam;return x<y?1:x>y?-1:0;}
int main(int argc,char**argv){
  eps=atof(argv[1]); int Tmax=atoi(argv[2]); int top=atoi(argv[3]);
  vp=((2*eps+1)-sqrt(4*eps+1))/(2*eps);
  size_t cap=1<<20,n=0; res_t *res=malloc(cap*sizeof(res_t));
  for(int T=1;T<=Tmax;T++){
    long tot=1; for(int i=0;i<T;i++)tot*=8;
    for(long code=0;code<tot;code++){
      int pat[8]; long cc=code; int hasP=0,hasQ=0;
      for(int i=0;i<T;i++){pat[i]=cc%8;cc/=8; if(pat[i]==0)hasP=1; if(pat[i]==1)hasQ=1;}
      if(!hasP||!hasQ)continue;
      double pi=diag(pat,T); double h=1e-5;
      double lam=(copyprob(pat,T,pi+h,pi-h)-copyprob(pat,T,pi-h,pi+h))/(2*h);
      if(n==cap){cap*=2;res=realloc(res,cap*sizeof(res_t));}
      res[n].lam=lam;res[n].pi=pi;res[n].T=T;memcpy(res[n].pat,pat,sizeof pat);n++;
    }
  }
  qsort(res,n,sizeof(res_t),cmp);
  printf("eps=%.4f varpi=%.4f: %zu patterns; top %d by multiplier\n",eps,vp,n,top);
  for(int i=0;i<top&&i<(int)n;i++){printf("%8.4f  pi=%.3f  ",res[i].lam,res[i].pi);for(int t=0;t<res[i].T;t++)printf("%s ",names[res[i].pat[t]]);printf("\n");}
  /* reference: plain latch */
  int pl[2]={0,1}; double pi=diag(pl,2),h=1e-5; printf("plain latch P Q: lambda=%.4f pi=%.3f\n",(copyprob(pl,2,pi+h,pi-h)-copyprob(pl,2,pi-h,pi+h))/(2*h),pi);
  return 0;
}
