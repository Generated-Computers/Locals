/* potential_check.c: rigorous check of the inequality behind the theorem on designs with
   independent reads.

   f(x) = x^(43/20) (1-x)^(33/20) on [0,1], and fhat is its least concave majorant on [0,1]:
   fhat(y) = S1 y for y <= 23/56, fhat(y) = f(y) for 23/56 <= y <= 43/56, fhat(y) = S2 (1-y) for
   y >= 43/56, where S1 = max f(t)/t (at t = 23/56) and S2 = max f(t)/(1-t) (at t = 43/56).
   For a noise level eps let varpi be the root of varpi = eps (1-varpi)^2, A = eps varpi,
   B = eps (1-varpi), c = 1-eps, hi = (1-B)/(1-B(1-A)), lo = 1-hi(1-A).  The program proves, for
   every eps in [e0,e1]:

     (B + c y) f(x) + c x fhat(y)  <=  theta f(1 - x (B + c y))   for all x in [lo,hi], y in [0,1].

   Method: interval arithmetic in IEEE double precision, round to nearest, every operation widened
   by one unit in the last place in the safe direction (nextafter); apart from nextafter and the
   conversion of the arguments (atof), no library function is trusted: a root r of g is accepted as an upper (lower) bound only after r^20 >= g (<= g) has
   been checked with widened multiplications. f and fhat increase on [0,43/76] and decrease on
   [43/76,1], which gives suprema and infima over intervals. The (eps, x, y) region is cut into
   boxes; a box on which the bound fails is halved; an eps interval on which a box of the
   smallest size fails is halved.

   Compile with:  cc -O2 -ffp-contract=off -o potential_check potential_check.c -lm
   usage: potential_check e0 e1 theta [width]     (width: first width of the eps intervals)
   The runs quoted in the note start and end a little outside the intervals they are said to
   cover, and use theta a little below the stated one, because the decimal arguments are rounded:
     potential_check 0.45499 0.46001 0.99979 0.0001
     potential_check 0.45999 0.47001 0.9995 0.00025
     potential_check 0.46999 0.50001 0.9995 0.002
     potential_check 0.49999 1.0 0.9995 0.01
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
static int boxok(double x0,double x1,double y0,double y1){
  double qlo=dn(Blo+dn(clo*y0)), qup=up(Bup+up(cup*y1));
  double aplo=dn(1.0-up(x1*qup)), apup=up(1.0-dn(x0*qlo));
  if(aplo<=0.001||apup>=0.999) return 0;
  double L=up(up(qup*fsup(x0,x1))+up(up(cup*x1)*fhatsup(y0,y1)));
  double Rr=dn(theta*finf(aplo,apup));
  return L<=Rr;
}
static int verify(double x0,double x1,double y0,double y1,double minsize,long cap){
  nbox++; if(nbox>cap) return 0;
  if(boxok(x0,x1,y0,y1)) return 1;
  double wx=x1-x0, wy=y1-y0;
  if(wx<minsize&&wy<minsize) return 0;
  if(wx>=wy){ double m=0.5*(x0+x1); return verify(x0,m,y0,y1,minsize,cap)&&verify(m,x1,y0,y1,minsize,cap); }
  else { double m=0.5*(y0+y1); return verify(x0,x1,y0,m,minsize,cap)&&verify(x0,x1,m,y1,minsize,cap); }
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
static int range(double e0,double e1,double minw,int depth){
  int ok=setup(e0,e1); long n0=nbox;
  if(ok){ long cap=nbox+40000000L;
    ok=verify(LO,HI,0.0,1.0,(HI-LO)/4e6,cap); }
  if(ok){ printf("eps in [%.6f, %.6f]: proved, %ld boxes, x in [%.4f, %.4f], y in [0,1]\n",e0,e1,nbox-n0,LO,HI); fflush(stdout); return 1; }
  if(e1-e0<minw){ printf("eps in [%.6f, %.6f]: FAILED\n",e0,e1); return 0; }
  double m=0.5*(e0+e1);
  return range(e0,m,minw,depth+1)&&range(m,e1,minw,depth+1);
}
int main(int argc,char**argv){
  if(argc<4){ fprintf(stderr,"usage: potential_check e0 e1 theta [width]\n"); return 1; }
  double e0=atof(argv[1]), e1=atof(argv[2]); theta=atof(argv[3]); double w=(argc>4)?atof(argv[4]):0.01;
  double m=(double)PA/(PA+PB); mode_dn=dn(m); mode_up=up(m);
  { double y=up(1.0-mode_dn); double g=up(pw_up(mode_up,PA)*pw_up(y,PB)); double r=pow(g,1.0/PD)*(1+1e-13);
    while(pw_dn(r,PD)<g) r*=(1+1e-13); fmax_up=r; }
  { double t=23.0/56.0; t1dn=dn(t); t1up=up(t); double y=up(1.0-t1dn);
    S1up=rootup(up(pw_up(t1up,PA-PD)*pw_up(y,PB)));          /* t^(23/20) (1-t)^(33/20) at its maximum */
    t=43.0/56.0; t2dn=dn(t); t2up=up(t); y=up(1.0-t2dn);
    S2up=rootup(up(pw_up(t2up,PA)*pw_up(y,PB-PD))); }        /* t^(43/20) (1-t)^(13/20) at its maximum */
  int all=1; long tot0=nbox;
  /* the intervals [a,b] are consecutive, and the last one ends exactly at e1 */
  for(double a=e0;a<e1;){ double b=a+w; if(b>e1||!(b>a)) b=e1; if(!range(a,b,1e-7,0)){ all=0; break; } a=b; }
  printf("%s for eps in [%.6f, %.6f] with theta = %.6f; %ld boxes\n",all?"PROVED":"NOT PROVED",e0,e1,theta,nbox-tot0);
  return all?0:1;
}
