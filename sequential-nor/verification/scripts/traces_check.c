/* traces_check.c: rigorous check of the two inequalities behind the contraction of the lower and
   upper traces of chains with pointers two back.

   f, fhat, varpi, A, B, c = 1-eps, lo, hi are as in potential_check.c.  The program proves, for
   every eps in [e0,e1]:
   (Q)  for all X, X' in [lo,hi], y in [0,1], with q = B + c y:
          B q f(X) + c (1-q) f(X') + c (B X + c X') fhat(y)  <=  theta f( 1 - B (1 - X q) - c (1-q) X' ),
   (R)  for all a1, a2, a3 in [lo,hi]:
          B f(a1) + c A f(a2) + c^2 f(a3)  <=  theta f( 1 - B a1 - c A a2 - c^2 (1 - a3) ).
   It uses that the arguments of f on the right lie in [lo,hi] (proved in the paper).

   Method: as in potential_check.c (IEEE doubles, every operation widened by one unit in the last
   place in the safe direction; roots accepted only after a check by multiplication; f and fhat
   increase up to 43/76 and decrease after it).  All quantities are nonnegative, so the left sides
   are bounded from above with upper bounds of the factors, and the argument on the right is
   enclosed with lower and upper bounds of its terms.

   Compile with:  cc -O2 -ffp-contract=off -o traces_check traces_check.c -lm
   usage: traces_check e0 e1 theta [width]
   The runs quoted in the paper (they prove the factor 9/10):
     traces_check 0.45499 0.50001 0.8999 0.0005
     traces_check 0.49999 1.0 0.8999 0.005
   Prints one line per eps interval and "PROVED" if the whole range is covered. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static inline double up(double x){ return nextafter(x, INFINITY); }
static inline double dn(double x){ return nextafter(x, -INFINITY); }
#define PA 43
#define PB 33
#define PD 20
static double pw_up(double x, int p){ double r=1; for(int i=0;i<p;i++) r=up(r*x); return r; }
static double pw_dn(double x, int p){ double r=1; for(int i=0;i<p;i++) r=dn(r*x); return r; }
/* upper and lower bounds of f at the real number x, a double in (0,1) */
static double f_up(double x){
  double y=up(1.0-x); double g=up(pw_up(x,PA)*pw_up(y,PB));
  double r=pow(g,1.0/PD)*(1+1e-13); int it=0;
  if(!(r>0)||!(g>0)){ fprintf(stderr,"root guess\n"); exit(2); }
  while(pw_dn(r,PD)<g){ r*=(1+1e-13); if(++it>100000){ fprintf(stderr,"root\n"); exit(2);} }
  return r;
}
static double f_dn(double x){
  double y=dn(1.0-x); double g=dn(pw_dn(x,PA)*pw_dn(y,PB));
  double r=pow(g,1.0/PD)*(1-1e-13); int it=0;
  if(!(r>0)||!(g>0)){ fprintf(stderr,"root guess\n"); exit(2); }
  while(pw_up(r,PD)>g){ r*=(1-1e-13); if(++it>100000){ fprintf(stderr,"root\n"); exit(2);} }
  return r;
}
static double mode_dn, mode_up, fmax_up;
static double fsup(double u, double v){ /* sup of f over [u,v] */
  if(v<=mode_dn) return f_up(v);
  if(u>=mode_up) return f_up(u);
  return fmax_up;
}
static double finf(double u, double v){ double a=f_dn(u), b=f_dn(v); return a<b?a:b; }
/* fhat: t1 = 23/56, t2 = 43/56; S1up >= max f(t)/t = t1^(23/20) (1-t1)^(33/20),
   S2up >= max f(t)/(1-t) = t2^(43/20) (1-t2)^(13/20) */
static double t1dn,t1up,t2dn,t2up,S1up,S2up;
static double rootup(double g){ double r=pow(g,1.0/PD)*(1+1e-13);
  if(!(r>0)||!(g>0)){ fprintf(stderr,"root guess\n"); exit(2); } while(pw_dn(r,PD)<g) r*=(1+1e-13); return r; }
static double fhat_up(double y){ /* upper bound of fhat at the double y in [0,1] */
  double r=0;
  if(y<=t1up){ double a=up(S1up*y); if(a>r) r=a; }
  if(y>=t2dn){ double a=up(S2up*up(1.0-y)); if(a>r) r=a; }
  if(y>=t1dn&&y<=t2up){ double a=f_up(y); if(a>r) r=a; }
  return r;
}
static double fhatsup(double u, double v){ /* sup of fhat over [u,v] */
  if(v<=mode_dn) return fhat_up(v);
  if(u>=mode_up) return fhat_up(u);
  return fmax_up;
}



static double Alo,Aup,Blo,Bup,clo,cup,LO,HI,theta;
static long nbox;
static double clampf(double alo,double aup){ if(alo<LO) alo=LO; if(aup>HI) aup=HI; if(!(alo<=aup)) return -1; return finf(alo,aup); }
static int boxQ(const double*b){ /* X in [b0,b1], X' in [b2,b3], y in [b4,b5] */
  double qlo=dn(Blo+dn(clo*b[4])), qup=up(Bup+up(cup*b[5]));
  double omq_up=up(1.0-qlo), omq_lo=dn(1.0-qup); if(omq_lo<0) omq_lo=0;
  double L=up(up(up(Bup*qup)*fsup(b[0],b[1]))+up(up(cup*omq_up)*fsup(b[2],b[3])));
  L=up(L+up(up(cup*up(up(Bup*b[1])+up(cup*b[3])))*fhatsup(b[4],b[5])));
  /* alpha = 1 - B + B X q - c (1-q) X' */
  double alo=dn(dn(dn(1.0-Bup)+dn(dn(Blo*b[0])*qlo))-up(up(cup*omq_up)*b[3]));
  double aup=up(up(up(1.0-Blo)+up(up(Bup*b[1])*qup))-dn(dn(clo*omq_lo)*b[2]));
  double fr=clampf(alo,aup); if(fr<0) return 0;
  return L<=dn(theta*fr);
}
static int boxR(const double*b){ /* a1 in [b0,b1], a2 in [b2,b3], a3 in [b4,b5] */
  double cA_up=up(cup*Aup), cA_lo=dn(clo*Alo), cc_up=up(cup*cup), cc_lo=dn(clo*clo);
  double L=up(up(up(Bup*fsup(b[0],b[1]))+up(cA_up*fsup(b[2],b[3])))+up(cc_up*fsup(b[4],b[5])));
  double alo=dn(dn(dn(1.0-up(Bup*b[1]))-up(cA_up*b[3]))-up(cc_up*up(1.0-b[4])));
  double aup=up(up(up(1.0-dn(Blo*b[0]))-dn(cA_lo*b[2]))-dn(cc_lo*dn(1.0-b[5])));
  double fr=clampf(alo,aup); if(fr<0) return 0;
  return L<=dn(theta*fr);
}
static int verify(int which,double*b,double minsize,long cap){
  nbox++; if(nbox>cap) return 0;
  if(which? boxR(b):boxQ(b)) return 1;
  int d=0; double w=b[1]-b[0];
  for(int i=1;i<3;i++){ double wi=b[2*i+1]-b[2*i]; if(wi>w){ w=wi; d=i; } }
  if(w<minsize) return 0;
  double m=0.5*(b[2*d]+b[2*d+1]), b1[6], b2[6];
  for(int i=0;i<6;i++){ b1[i]=b[i]; b2[i]=b[i]; }
  b1[2*d+1]=m; b2[2*d]=m;
  return verify(which,b1,minsize,cap)&&verify(which,b2,minsize,cap);
}
/* enclosures of the constants for eps in [e0,e1]; returns 0 if they could not be certified */
static int setup(double e0,double e1){
  double v0=((2*e0+1)-sqrt(4*e0+1))/(2*e0)*(1-1e-12), v1=((2*e1+1)-sqrt(4*e1+1))/(2*e1)*(1+1e-12);
  { double t=dn(1.0-v0), s=dn(t*t), u=dn(e0*s); if(!(u>=v0)) return 0; }   /* v0 <= varpi(e0) */
  { double t=up(1.0-v1), s=up(t*t), u=up(e1*s); if(!(u<=v1)) return 0; }   /* v1 >= varpi(e1) */
  /* varpi is increasing in eps, so varpi(eps) lies in [v0,v1] for eps in [e0,e1] */
  Alo=dn(e0*v0); Aup=up(e1*v1); clo=dn(1.0-e1); if(clo<0) clo=0; cup=up(1.0-e0);
  Blo=dn(e0*dn(1.0-v1)); Bup=up(e1*up(1.0-v0));
  double num=up(1.0-Blo), den=dn(1.0-up(Bup*up(1.0-Alo)));
  HI=up(num/den); LO=dn(1.0-up(HI*up(1.0-Alo)));
  if(!(0<v0&&v0<=v1&&v1<1)) return 0;
  return LO>0.01&&HI<0.99;
}
static long nQ,nR;
static int range(double e0,double e1,double minw){
  int ok=setup(e0,e1); long n0=nbox,n1=0;
  if(ok){ double b[6]={LO,HI,LO,HI,0.0,1.0}; ok=verify(0,b,(HI-LO)/1e4,nbox+40000000L); n1=nbox; }
  if(ok){ double b[6]={LO,HI,LO,HI,LO,HI}; ok=verify(1,b,(HI-LO)/1e4,nbox+40000000L); }
  if(ok){ nQ+=n1-n0; nR+=nbox-n1; printf("eps in [%.6f, %.6f]: proved, %ld + %ld boxes, [lo,hi] within [%.4f, %.4f]\n",e0,e1,n1-n0,nbox-n1,LO,HI); fflush(stdout); return 1; }
  if(e1-e0<minw){ printf("eps in [%.6f, %.6f]: FAILED\n",e0,e1); return 0; }
  double m=0.5*(e0+e1);
  return range(e0,m,minw)&&range(m,e1,minw);
}
int main(int argc,char**argv){
  if(argc<4){ fprintf(stderr,"usage: traces_check e0 e1 theta [width]\n"); return 1; }
  double e0=atof(argv[1]), e1=atof(argv[2]); theta=atof(argv[3]); double w=(argc>4)?atof(argv[4]):0.01;
  double m=(double)PA/(PA+PB); mode_dn=dn(m); mode_up=up(m);
  { double y=up(1.0-mode_dn); double g=up(pw_up(mode_up,PA)*pw_up(y,PB)); double r=pow(g,1.0/PD)*(1+1e-13);
    while(pw_dn(r,PD)<g) r*=(1+1e-13); fmax_up=r; }
  { double t=23.0/56.0; t1dn=dn(t); t1up=up(t); double y=up(1.0-t1dn);
    S1up=rootup(up(pw_up(t1up,PA-PD)*pw_up(y,PB)));
    t=43.0/56.0; t2dn=dn(t); t2up=up(t); y=up(1.0-t2dn);
    S2up=rootup(up(pw_up(t2up,PA)*pw_up(y,PB-PD))); }
  int all=1;
  for(double a=e0;a<e1;){ double b=a+w; if(b>e1||!(b>a)) b=e1; if(!range(a,b,1e-7)){ all=0; break; } a=b; }
  printf("%s for eps in [%.6f, %.6f] with theta = %.6f; %ld boxes for (Q), %ld for (R)\n",all?"PROVED":"NOT PROVED",e0,e1,theta,nQ,nR);
  return all?0:1;
}
