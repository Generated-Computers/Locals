/* rows_periodic.c: the two-row chain with a fixed density d, for all periodic rows below.
   Symbols of the row below: 0 and 1 closed, o an open 0, I an open 1; no two adjacent ones.  For
   every period T <= Tmax and every such word with at least one open symbol it computes, in the
   stationary regime, the ratio
       E[number of open positions of the upper row] / E[number of open positions of the lower row]
   per period, and prints the largest ratio and its word.  Floating point; an exploration, not a proof.
   Compile with:  cc -O2 -o rows_periodic rows_periodic.c -lm
   usage: rows_periodic eps d Tmax                      quoted in the paper: rows_periodic 0.455 0.3333333333 9 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static double eps,d;
static void gate(int pv,int po,int b,int ob,int*x,int*o){ *x=!(pv|b); if(!pv&&b)*o=ob; else if(pv&&b)*o=po&&ob; else if(!pv&&!b)*o=po||ob; else *o=po; }
static double M[4][16][16];
int main(int c,char**v){ eps=atof(v[1]); d=atof(v[2]); int Tmax=atoi(v[3]);
 for(int sym=0;sym<4;sym++){ int b=sym&1, ob=sym>>1; for(int s=0;s<16;s++){ int x1=s&1,o1=(s>>1)&1,x2=(s>>2)&1,o2=(s>>3)&1;
   double p1[3]={1-eps,eps*d,eps*(1-d)}; int r1[3]={b,1,0}, q1[3]={ob,0,0};
   for(int i=0;i<3;i++){ int nx1,no1; gate(x1,o1,r1[i],q1[i],&nx1,&no1); int r2[3]={nx1,1,0}, q2[3]={no1,0,0};
     for(int k=0;k<3;k++){ int nx2,no2; gate(x2,o2,r2[k],q2[k],&nx2,&no2); M[sym][s][nx1|no1<<1|nx2<<2|no2<<3]+=p1[i]*p1[k]; } } } }
 double best=0; for(int T=1;T<=Tmax;T++){ long N=1L<<(2*T); double bt=0; long bw=0;
  for(long w=0;w<N;w++){ int ok=1,any=0; for(int i=0;i<T;i++){ int a=(w>>(2*i))&3, bb=(w>>(2*((i+1)%T)))&3; if((a&1)&&(bb&1)&&T>1) ok=0; if(T==1&&(a&1)) ok=0; if(a>>1) any=1; } if(!ok||!any) continue;
   double p[16]={0},q[16]; p[0]=1; double n1=0,n2=0;
   for(int rep=0;rep<40;rep++){ n1=n2=0; for(int i=0;i<T;i++){ int sym=(w>>(2*i))&3; for(int s=0;s<16;s++)q[s]=0; for(int s=0;s<16;s++) if(p[s]>0) for(int u=0;u<16;u++) q[u]+=p[s]*M[sym][s][u];
      for(int s=0;s<16;s++){ p[s]=q[s]; if(s&2) n1+=p[s]; if(s&8) n2+=p[s]; } } }
   double r=n2/n1; if(r>bt){bt=r;bw=w;} }
  printf("eps %.3f d %.2f period %d  max ratio %.5f  word ",eps,d,T,bt); for(int i=0;i<T;i++){ int a=(bw>>(2*i))&3; printf("%s",a==0?"0":a==1?"1":a==2?"o":"I"); } printf("\n"); fflush(stdout); if(bt>best)best=bt; }
}
