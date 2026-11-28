// referee: numeric evaluation of Lemma traceineq (a),(b): grid + local refinement, long double
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
typedef long double R;
static R f(R x){ if(x<=0||x>=1) return 0; return powl(x,2.15L)*powl(1-x,1.65L);}
static R fh(R y){ R t1=23.0L/56,t2=43.0L/56; if(y<=t1) return f(t1)/t1*y; if(y>=t2) return f(t2)/(1-t2)*(1-y); return f(y);}
static R e,w,A,B,c,r,lo,hi;
static void setc(R ee){ e=ee; R a=0,b=1; for(int i=0;i<200;i++){R m=(a+b)/2; if(e*(1-m)*(1-m)-m>0)a=m; else b=m;} w=a; A=e*w;B=e*(1-w);c=1-e;r=1-A;hi=(1-B)/(1-B*r);lo=1-r*hi;}
static R ra(R X,R Xp,R y,int*inr){ R q=B+c*y; R arg=1-B*(1-X*q)-c*(1-q)*Xp; if(arg<lo-1e-15L||arg>hi+1e-15L)*inr=0; return (B*q*f(X)+c*(1-q)*f(Xp)+c*(B*X+c*Xp)*fh(y))/f(arg);}
static R rb(R a1,R a2,R a3,int*inr){ R arg=1-B*a1-c*A*a2-c*c*(1-a3); if(arg<lo-1e-15L||arg>hi+1e-15L)*inr=0; return (B*f(a1)+c*A*f(a2)+c*c*f(a3))/f(arg);}
static R cl(R x,R a,R b){return x<a?a:x>b?b:x;}
int main(int argc,char**argv){ int N=argc>1?atoi(argv[1]):120;
  R es[]={0.455L,0.4551L,0.456L,0.46L,0.47L,0.5L,0.55L,0.6L,0.7L,0.8L,0.9L,0.99L,0.999999L};
  for(int k=0;k<13;k++){ setc(es[k]); int inr=1; R ba=0,bb=0; R pa[3]={0},pb[3]={0};
    for(int i=0;i<=N;i++)for(int j=0;j<=N;j++)for(int l=0;l<=N;l++){ R X=lo+(hi-lo)*i/N, Y=lo+(hi-lo)*j/N, y=(R)l/N, z=lo+(hi-lo)*l/N;
      R v=ra(X,Y,y,&inr); if(v>ba){ba=v;pa[0]=X;pa[1]=Y;pa[2]=y;} v=rb(X,Y,z,&inr); if(v>bb){bb=v;pb[0]=X;pb[1]=Y;pb[2]=z;} }
    // local refinement by coordinate search
    R st=(hi-lo)/N; for(int it=0;it<200;it++){ for(int d=0;d<3;d++) for(int s=-1;s<=1;s+=2){ R p[3]={pa[0],pa[1],pa[2]}; p[d]+=s*st; p[0]=cl(p[0],lo,hi);p[1]=cl(p[1],lo,hi);p[2]=cl(p[2],0,1); R v=ra(p[0],p[1],p[2],&inr); if(v>ba){ba=v;pa[0]=p[0];pa[1]=p[1];pa[2]=p[2];}
        R q[3]={pb[0],pb[1],pb[2]}; q[d]+=s*st; for(int t=0;t<3;t++)q[t]=cl(q[t],lo,hi); v=rb(q[0],q[1],q[2],&inr); if(v>bb){bb=v;pb[0]=q[0];pb[1]=q[1];pb[2]=q[2];} } st*=0.93L; }
    printf("eps %.6Lf lo %.5Lf hi %.5Lf: max ratio (a) %.6Lf at (%.4Lf,%.4Lf,%.4Lf); (b) %.6Lf at (%.4Lf,%.4Lf,%.4Lf); args in [lo,hi]: %d\n",e,lo,hi,ba,pa[0],pa[1],pa[2],bb,pb[0],pb[1],pb[2],inr);
  } return 0; }
