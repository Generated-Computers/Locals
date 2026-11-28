/* scan.c: for a noise level, the best potential f(x) = x^a (1-x)^b (not rigorous; a grid).
   sup over a1, a2 in [lo,hi] and the dead reads of  [q f(a1) + c a1 f(a2)] / f(1 - a1 q).
   usage: scan eps amin amax bmin bmax step N */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static double A,B,c,lo,hi;
static double supratio(double a,double b,int N){
  double worst=0;
  for(int i=0;i<N;i++){ double a1=lo+(hi-lo)*i/(N-1), f1=pow(a1,a)*pow(1-a1,b);
    for(int j=-2;j<N;j++){ double a2=(j==-2)?0:(j==-1)?1:lo+(hi-lo)*j/(N-1);
      double q=B+c*a2, ap=1-a1*q, num=q*f1+((j>=0)?c*a1*pow(a2,a)*pow(1-a2,b):0);
      double r=num/(pow(ap,a)*pow(1-ap,b)); if(r>worst) worst=r; } }
  return worst;
}
int main(int argc,char**argv){
  double e=atof(argv[1]),amin=atof(argv[2]),amax=atof(argv[3]),bmin=atof(argv[4]),bmax=atof(argv[5]),st=atof(argv[6]); int N=atoi(argv[7]);
  double vp=((2*e+1)-sqrt(4*e+1))/(2*e); A=e*vp; B=e*(1-vp); c=1-e;
  hi=(1-B)/(1-B*(1-A)); lo=1-hi*(1-A);
  double best=9,ba=0,bb=0;
  for(double a=amin;a<=amax+1e-9;a+=st) for(double b=bmin;b<=bmax+1e-9;b+=st){ double r=supratio(a,b,N); if(r<best){best=r;ba=a;bb=b;} }
  printf("eps=%.4f lo=%.4f hi=%.4f: best a=%.2f b=%.2f sup ratio %.5f (grid %d); finer grid: %.5f\n",e,lo,hi,ba,bb,best,N,supratio(ba,bb,4*N));
  return 0;
}
