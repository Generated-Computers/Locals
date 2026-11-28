/* scan_check.c: rigorous bound on the two-layer spread G2 of scanning programs.

   Ideal model.  A plain chain P has gates whose targets are independent bits that are 1 with
   probability d; a read is resampled with probability eps to a bit that is 1 with probability varpi.
   A scanning chain E runs a word over the letters U and R: a gate U reads the current position of
   a new, independent stationary copy of P (a window); a gate R reads the next position of the
   window of the gate before it.  A block is U R^k with 0 <= k <= K.  One target of one gate of
   one window, at or before the position read, is 0 in one copy and 1 in the other.

   For p0 = P[x_E = 0] before a block, the program computes enclosures of
     cost_k(p0)  expected number of positions of the block at which the two copies of E differ,
                 added over the positions of the differing target (all those up to the last read),
     X_k(p0)     the same for the last position of the block alone,
     out_k(p0)   P[x_E = 0] after the block,   ones_k(p0)  expected number of ones in the block,
     S_k, Q_k    expected number of positions of a block through which a difference that enters it
                 persists, and the probability that it leaves the block,
   all affine in p0, and proves for every eps in [e0,e1] and every d in [0,1]:  either
     (X)  no word has density d: with I an interval that contains p0 before every block,
          d < min_{k, p0 in I} (ones_k + g (out_k - p0))/(k+1)  for some g, or
          d > max_{k, p0 in I} (ones_k + g (out_k - p0))/(k+1)  for some g;   or
     (B)  for some gamma:  cost_k(p0) + C X_k(p0) + gamma (out_k(p0) - p0) <= theta (k+1)
          for all k <= K and p0 in I, where C >= max_k S_k/(1-Q_k).
   The paper shows that (B) gives G2 <= theta for every word with that d, and that (X) excludes d.

   Arithmetic: IEEE doubles, every operation widened by one unit in the last place in the safe
   direction (nextafter); varpi is enclosed as in potential_check.c.
   Compile with:  cc -O2 -ffp-contract=off -o scan_check scan_check.c -lm
   usage: scan_check e0 e1 theta K [width]
   The runs quoted in the paper:
     scan_check 0.44599 0.45501 0.999 120 0.0005
     scan_check 0.45499 0.50001 0.99 120 0.001
     scan_check 0.49999 1.0 0.99 120 0.01                                                       */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static inline double up(double x){ return nextafter(x, INFINITY); }
static inline double dn(double x){ return nextafter(x, -INFINITY); }
typedef struct{double lo,hi;}iv;
static iv mk(double a,double b){iv r={a,b};return r;}
static iv iadd(iv a,iv b){return mk(dn(a.lo+b.lo),up(a.hi+b.hi));}
static iv isub(iv a,iv b){return mk(dn(a.lo-b.hi),up(a.hi-b.lo));}
static iv imul(iv a,iv b){double p[4]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};double l=p[0],h=p[0];
  for(int i=1;i<4;i++){if(p[i]<l)l=p[i];if(p[i]>h)h=p[i];} return mk(dn(l),up(h));}
static iv idiv(iv a,iv b){ if(!(b.lo>0)){fprintf(stderr,"division\n");exit(2);} return imul(a,mk(dn(1.0/b.hi),up(1.0/b.lo))); }
static const iv ZERO={0,0}, ONE={1,1};
#define KMAX 128
static int K; static double theta;
static iv A,B,C,Q,AL;                 /* A, B, c = 1-eps, q of the plain chain, alpha of the plain chain */
/* results for the two starts p0 = 0 (index 0) and p0 = 1 (index 1) */
static iv cost[2][KMAX+1], Xk[2][KMAX+1], outk[2][KMAX+1], ones[2][KMAX+1], Sk[KMAX+1], Qk[KMAX+1];
/* pair states: P pair 0 = both 0, 1 = both 1, 2 = (1,0), 3 = (0,1); E pair index 2a+b */
static const int PA[4]={0,1,1,0}, PB[4]={0,1,0,1};
static int pidx(int a,int b){ return a==b?(a?1:0):(a?2:3); }
static void eread(iv m,int pe,int pp,iv nd[4][4]){        /* add mass m of E pair pe after a read of the P pair pp */
  int a=pe>>1,b=pe&1,v=PA[pp],w=PB[pp];
  nd[pp][0]=iadd(nd[pp][0],imul(m,A));
  int s=2*(!a)+(!b); nd[pp][s]=iadd(nd[pp][s],imul(m,B));
  s=2*(!(a||v))+(!(b||w)); nd[pp][s]=iadd(nd[pp][s],imul(m,C)); }
static void run(void){
  iv omq=isub(ONE,Q), omal=isub(ONE,AL);
  /* a difference that enters a clean block */
  { iv w0=AL,w1=omal,S=ZERO;
    for(int j=0;j<=K;j++){
      if(j>0){ iv n1=imul(w0,Q), n0=iadd(imul(w0,omq),w1); w0=n0; w1=n1; }
      w0=imul(w0,iadd(B,C)); w1=imul(w1,B);             /* the read of E is 0 */
      iv z=iadd(w0,w1); S=iadd(S,z); Sk[j]=S; Qk[j]=z; } }
  for(int st=0;st<2;st++){
    iv px[2]; px[0]=st?ONE:ZERO; px[1]=st?ZERO:ONE;       /* P[x_E = x] before the block */
    iv mu[2][2], d[4][4], cs=ZERO, on=ZERO;
    for(int j=0;j<=K;j++){
      iv nd[4][4]; for(int a=0;a<4;a++) for(int b=0;b<4;b++) nd[a][b]=ZERO;
      iv nm[2][2]={{ZERO,ZERO},{ZERO,ZERO}};
      if(j==0){
        iv den=isub(ONE,imul(Q,Q)), base=idiv(imul(C,AL),den);
        for(int x=0;x<2;x++){ eread(imul(base,px[x]),3*x,2,nd); eread(imul(imul(base,Q),px[x]),3*x,3,nd); }
        for(int x=0;x<2;x++) for(int P2=0;P2<2;P2++){ iv m=imul(px[x],P2?omal:AL);
          nm[0][P2]=iadd(nm[0][P2],imul(m,A)); nm[!x][P2]=iadd(nm[!x][P2],imul(m,B));
          int x2=!(x||P2); nm[x2][P2]=iadd(nm[x2][P2],imul(m,C)); }
      } else {
        for(int pp=0;pp<4;pp++) for(int pe=0;pe<4;pe++){ iv m=d[pp][pe]; if(m.hi==0) continue;
          int a=PA[pp],b=PB[pp];
          eread(imul(m,Q),pe,pidx(!a,!b),nd);            /* the window reads a zero */
          eread(imul(m,omq),pe,0,nd); }                  /* the window reads a one */
        for(int x=0;x<2;x++) eread(imul(C,mu[x][0]),3*x,2,nd);     /* the new gate of the window is the differing one */
        for(int x=0;x<2;x++) for(int P=0;P<2;P++){ iv m=mu[x][P];
          for(int r=0;r<2;r++){ iv p1; int P2;
            if(P==0){ p1=r?omq:Q; P2=r?0:1; } else { if(r) continue; p1=ONE; P2=0; }
            iv mm=imul(m,p1);
            nm[0][P2]=iadd(nm[0][P2],imul(mm,A)); nm[!x][P2]=iadd(nm[!x][P2],imul(mm,B));
            int x2=!(x||P2); nm[x2][P2]=iadd(nm[x2][P2],imul(mm,C)); } }
      }
      iv X=ZERO;
      for(int pp=0;pp<4;pp++) for(int pe=0;pe<4;pe++){
        int pdiff=(pp>=2), ediff=(pe==1||pe==2);
        if(!pdiff&&!ediff) nd[pp][pe]=ZERO;              /* the copies agree from now on */
        d[pp][pe]=nd[pp][pe]; if(ediff) X=iadd(X,nd[pp][pe]); }
      for(int x=0;x<2;x++) for(int P=0;P<2;P++) mu[x][P]=nm[x][P];
      cs=iadd(cs,X); on=iadd(on,iadd(mu[1][0],mu[1][1]));
      cost[st][j]=cs; Xk[st][j]=X; outk[st][j]=iadd(mu[0][0],mu[0][1]); ones[st][j]=on; } }
}
static iv at(iv v0,iv v1,double t){ iv T=mk(t,t), O=mk(dn(1.0-t),up(1.0-t)); return iadd(imul(O,v0),imul(T,v1)); }
/* returns 1 if the box is excluded, 2 if the bound is proved, 0 otherwise */
static int box(double d0,double d1){
  iv D=mk(d0,d1); Q=iadd(B,imul(C,isub(ONE,D))); AL=idiv(ONE,iadd(ONE,Q));
  run();
  double a=0,b=1;
  for(int it=0;it<60;it++){ double lo=1,hi=0;
    for(int k=0;k<=K;k++) for(int e=0;e<2;e++){ iv v=at(outk[0][k],outk[1][k],e?b:a); if(v.lo<lo) lo=v.lo; if(v.hi>hi) hi=v.hi; }
    if(lo<0) lo=0; if(hi>1) hi=1; a=lo; b=hi; }
  /* (X): the density of ones */
  for(int gi=-50;gi<=50;gi++){ double g=gi/25.0; double mn=1e9,mx=-1e9;
    for(int k=0;k<=K;k++) for(int e=0;e<2;e++){ double t=e?b:a;
      iv v=iadd(at(ones[0][k],ones[1][k],t),imul(mk(g,g),isub(at(outk[0][k],outk[1][k],t),mk(t,t))));
      double l=dn(v.lo/(k+1)), h=up(v.hi/(k+1)); if(l<mn) mn=l; if(h>mx) mx=h; }
    if(d1<mn||d0>mx) return 1; }
  /* (B) */
  double Cs=0;
  for(int k=0;k<=K;k++){ iv v=idiv(Sk[k],isub(ONE,Qk[k])); if(v.hi>Cs) Cs=v.hi; }
  for(int gi=50;gi<=130;gi++){ double g=gi/100.0; int ok=1;
    for(int k=0;k<=K&&ok;k++) for(int e=0;e<2;e++){ double t=e?b:a;
      iv v=iadd(at(cost[0][k],cost[1][k],t),imul(mk(Cs,Cs),at(Xk[0][k],Xk[1][k],t)));
      v=iadd(v,imul(mk(g,g),isub(at(outk[0][k],outk[1][k],t),mk(t,t))));
      if(!(v.hi<=dn(theta*(k+1)))){ ok=0; break; } }
    if(ok) return 2; }
  return 0;
}
static long nx,nb;
static int drange(double d0,double d1){
  int r=box(d0,d1);
  if(r==1){ nx++; return 1; } if(r==2){ nb++; return 1; }
  if(d1-d0<1e-6) return 0;
  double m=0.5*(d0+d1); return drange(d0,m)&&drange(m,d1);
}
static int setup(double e0,double e1){
  double v0=((2*e0+1)-sqrt(4*e0+1))/(2*e0)*(1-1e-12), v1=((2*e1+1)-sqrt(4*e1+1))/(2*e1)*(1+1e-12);
  { double t=dn(1.0-v0), s=dn(t*t), u=dn(e0*s); if(!(u>=v0)) return 0; }
  { double t=up(1.0-v1), s=up(t*t), u=up(e1*s); if(!(u<=v1)) return 0; }
  if(!(0<v0&&v0<=v1&&v1<1)) return 0;
  A=mk(dn(e0*v0),up(e1*v1)); C=mk(dn(1.0-e1),up(1.0-e0)); if(C.lo<0) C.lo=0;
  B=mk(dn(e0*dn(1.0-v1)),up(e1*up(1.0-v0)));
  return 1;
}
static int erange(double e0,double e1){
  long x0=nx,b0=nb; int ok=setup(e0,e1);
  if(ok){ ok=1; for(int i=0;i<16&&ok;i++) ok=drange(i/16.0,(i+1)/16.0); }
  if(ok){ printf("eps in [%.6f, %.6f]: proved, %ld boxes of d excluded, %ld with the bound\n",e0,e1,nx-x0,nb-b0); fflush(stdout); return 1; }
  nx=x0; nb=b0;
  if(e1-e0<1e-6){ printf("eps in [%.6f, %.6f]: FAILED\n",e0,e1); return 0; }
  double m=0.5*(e0+e1); return erange(e0,m)&&erange(m,e1);
}
int main(int argc,char**argv){
  if(argc<5){ fprintf(stderr,"usage: scan_check e0 e1 theta K [width]\n"); return 1; }
  double e0=atof(argv[1]), e1=atof(argv[2]); theta=atof(argv[3]); K=atoi(argv[4]); double w=argc>5?atof(argv[5]):0.01;
  if(K<0||K>KMAX) return 1;
  int all=1;
  for(double a=e0;a<e1;){ double b=a+w; if(b>e1||!(b>a)) b=e1; if(!erange(a,b)){ all=0; break; } a=b; }
  printf("%s for eps in [%.6f, %.6f] with theta = %.6f and K = %d; %ld boxes excluded, %ld with the bound\n",all?"PROVED":"NOT PROVED",e0,e1,theta,K,nx,nb);
  return all?0:1;
}
