/* runs_check.c: rigorous check of the inequalities behind the theorem on designs with pointers
   two back (types L_2), for runs of such gates after a gate with an independent read.

   f, fhat, varpi, A, B, c = 1-eps, lo, hi are as in potential_check.c, and r = 1-A.  For
   x in [lo,hi], y in [0,1] and q = B + c y put
     alpha_{-1} = x,  alpha_0 = 1 - x q,  alpha_j = (2-eps) - r alpha_{j-1} - c alpha_{j-2}  (j >= 1),
     v^0 = ( c x fhat(y),  q f(x),  (1-q) f(x) ),
     v^j = ( c v3,  r v1 + B v2,  A v1 + (1-B) v2 )  with v = v^{j-1}                        (j >= 1).
   The program proves, for every eps in [e0,e1], x in [lo,hi], y in [0,1]:
     (i)   v^k_1 + v^k_2            <=  theta f(alpha_k)              for 1 <= k < K,
     (ii)  v^K_1 + v^K_2 + v^K_3    <=  theta min( f(lo), f(hi) ).
   It uses that every alpha_k lies in [lo,hi] (proved in the paper), and that
   alpha_k = C_k + x (P_k + Q_k q) with numbers C_k, P_k, Q_k that depend on eps only and satisfy
   the recursion of alpha (C: with the term 2-eps; P, Q: without), from (C,P,Q)_{-1} = (0,1,0) and
   (C,P,Q)_0 = (1,0,-1).

   Method: as in potential_check.c (IEEE doubles, every operation widened by one unit in the last
   place in the safe direction; roots accepted only after a check by multiplication; f and fhat
   increase up to 43/76 and decrease after it).  The components of v^j are sums and products of
   nonnegative numbers, so upper bounds of the inputs give upper bounds of v^j.

   Compile with:  cc -O2 -ffp-contract=off -o runs_check runs_check.c -lm
   usage: runs_check e0 e1 theta K [width]
   The runs quoted in the paper (they prove the lemma's factor 9/10; the argument is a little
   below it because decimal arguments are rounded):
     runs_check 0.45499 0.50001 0.8999 18 0.0005
     runs_check 0.49999 1.0 0.8999 18 0.005
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


typedef struct{double lo,hi;}iv;
static iv iadd(iv a,iv b){iv r={dn(a.lo+b.lo),up(a.hi+b.hi)};return r;}
static iv imul(iv a,iv b){double p[4]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};double l=p[0],h=p[0];
  for(int i=1;i<4;i++){if(p[i]<l)l=p[i];if(p[i]>h)h=p[i];} iv r={dn(l),up(h)};return r;}
#define KMAX 64
static double Alo,Aup,Blo,Bup,clo,cup,rlo,rup,LO,HI,theta,fminLH; static int K;
static iv Ck[KMAX+2],Pk[KMAX+2],Qk[KMAX+2];   /* index j+1 holds the coefficients of alpha_j */
static long nbox;
static int boxok(double x0,double x1,double y0,double y1){
  double qlo=dn(Blo+dn(clo*y0)), qup=up(Bup+up(cup*y1));
  double F=fsup(x0,x1);
  double v1=up(up(cup*x1)*fhatsup(y0,y1)), v2=up(qup*F), v3=up(up(1.0-qlo)*F);
  iv X={x0,x1}, Qv={qlo,qup};
  for(int k=1;k<=K;k++){
    double w1=up(cup*v3), w2=up(up(rup*v1)+up(Bup*v2)), w3=up(up(Aup*v1)+up(up(1.0-Blo)*v2));
    v1=w1; v2=w2; v3=w3;
    if(k<K){
      iv al=iadd(Ck[k+1],imul(X,iadd(Pk[k+1],imul(Qk[k+1],Qv))));
      double alo=al.lo<LO?LO:al.lo, aup=al.hi>HI?HI:al.hi;      /* alpha_k lies in [lo,hi] */
      if(!(alo<=aup)) return 0;
      if(!(up(v1+v2)<=dn(theta*finf(alo,aup)))) return 0;
    } else {
      if(!(up(up(v1+v2)+v3)<=dn(theta*fminLH))) return 0;
    }
  }
  return 1;
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
  rlo=dn(1.0-Aup); rup=up(1.0-Alo);
  double num=up(1.0-Blo), den=dn(1.0-up(Bup*up(1.0-Alo)));
  HI=up(num/den); LO=dn(1.0-up(HI*up(1.0-Alo)));
  if(!(0<v0&&v0<=v1&&v1<1)) return 0;
  if(!(LO>0.01&&HI<0.99)) return 0;
  fminLH=finf(LO,HI);
  iv two={dn(2.0-e1),up(2.0-e0)}, mr={-rup,-rlo}, mc={-cup,-clo};
  Ck[0]=(iv){0,0}; Pk[0]=(iv){1,1}; Qk[0]=(iv){0,0};
  Ck[1]=(iv){1,1}; Pk[1]=(iv){0,0}; Qk[1]=(iv){-1,-1};
  for(int j=2;j<=K+1;j++){
    Ck[j]=iadd(two,iadd(imul(mr,Ck[j-1]),imul(mc,Ck[j-2])));
    Pk[j]=iadd(imul(mr,Pk[j-1]),imul(mc,Pk[j-2]));
    Qk[j]=iadd(imul(mr,Qk[j-1]),imul(mc,Qk[j-2]));
  }
  return 1;
}
static int range(double e0,double e1,double minw,int depth){
  int ok=setup(e0,e1); long n0=nbox;
  if(ok){ long cap=nbox+4000000L;
    ok=verify(LO,HI,0.0,1.0,(HI-LO)/1e5,cap); }
  if(ok){ printf("eps in [%.6f, %.6f]: proved, %ld boxes, x in [%.4f, %.4f], y in [0,1]\n",e0,e1,nbox-n0,LO,HI); fflush(stdout); return 1; }
  if(e1-e0<minw){ printf("eps in [%.6f, %.6f]: FAILED\n",e0,e1); return 0; }
  double m=0.5*(e0+e1);
  return range(e0,m,minw,depth+1)&&range(m,e1,minw,depth+1);
}
int main(int argc,char**argv){
  if(argc<5){ fprintf(stderr,"usage: runs_check e0 e1 theta K [width]\n"); return 1; }
  double e0=atof(argv[1]), e1=atof(argv[2]); theta=atof(argv[3]); K=atoi(argv[4]); double w=(argc>5)?atof(argv[5]):0.01;
  if(K<1||K>KMAX){ fprintf(stderr,"K\n"); return 1; }
  double m=(double)PA/(PA+PB); mode_dn=dn(m); mode_up=up(m);
  { double y=up(1.0-mode_dn); double g=up(pw_up(mode_up,PA)*pw_up(y,PB)); double r=pow(g,1.0/PD)*(1+1e-13);
    while(pw_dn(r,PD)<g) r*=(1+1e-13); fmax_up=r; }
  { double t=23.0/56.0; t1dn=dn(t); t1up=up(t); double y=up(1.0-t1dn);
    S1up=rootup(up(pw_up(t1up,PA-PD)*pw_up(y,PB)));
    t=43.0/56.0; t2dn=dn(t); t2up=up(t); y=up(1.0-t2dn);
    S2up=rootup(up(pw_up(t2up,PA)*pw_up(y,PB-PD))); }
  int all=1; long tot0=nbox;
  for(double a=e0;a<e1;){ double b=a+w; if(b>e1||!(b>a)) b=e1; if(!range(a,b,1e-7,0)){ all=0; break; } a=b; }
  printf("%s for eps in [%.6f, %.6f] with theta = %.6f and K = %d; %ld boxes\n",all?"PROVED":"NOT PROVED",e0,e1,theta,K,nbox-tot0);
  return all?0:1;
}
