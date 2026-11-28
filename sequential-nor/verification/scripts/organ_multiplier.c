/* organ_multiplier.c: exact multipliers of periodic layered designs in the ideal model.

   Ideal model (as in the section on the latch embedding): the density of ones outside the layers
   is varpi(eps); a layer consists of long chains ("rails") of R kinds; a rail of kind r repeats a
   program of T gates. A class is a pair (rail, phase). The gate at phase f of rail r has one of
   the read types
     E(c)      a uniformly random position of class c of the layer below,
     Z0        position 0, which holds a one,
     Z1        position 1, which holds a zero,
     L1..LK    the position 1..K before it in the same chain (L1 is a NOT gate),
     U         a uniformly random position of the layer below (any class),
     S         the position that the last gate of type E or U before it in the chain reads.
   Every read is resampled with probability eps, independently, and then returns a one with
   probability varpi. The positions read by different gates of type E or U are independent.

   For a design the program finds fixed points d* of the map "class densities of one layer ->
   class densities of the next" (by iteration from five starting points), and at each of them
     rho_u  the spectral radius of the matrix Mbar[k][t] = mean number of positions of class k
            of the layer above that change when one position of class t of the layer below is
            flipped and both layers are evaluated with the same noise;
     rho_s  the spectral radius of the Jacobian of the density map.
   rho_s <= rho_u always (the Jacobian is bounded entrywise by Mbar).
   rho_u > 1: a flip survives from layer to layer with positive probability.
   rho_s > 1, or two fixed points: the densities can store information.
   The plain chain (R=1, T=1, E) has rho_u = (1-eps)/(1-q^2); the latch (two rails, program
   E(own,1) E(other,1)) has rho_s = rho_u = (1-eps)/(1-q^2); both are 1 at eps = 4/9.

   usage: organ_multiplier eps R T K mode ext thr a b [code]
     mode 0: all designs with R rails; a = shard, b = number of shards.
     mode 1: two rails, rail 1 is rail 0 with the two rails exchanged; a, b as in mode 0.
     mode 2: one rail of period T; the first a gates run through the types E(0..b-1), Z0, Z1,
             L1..LK (and S if ext = 2), all other gates are of type U.
     ext 0: types E, Z0, Z1, L1..LK; ext 1: also U; ext 2: also U and S.
   Prints every design that has rho_u > thr or rho_s > thr at some fixed point, or two or more
   fixed points, or a start without convergence. With a code, prints the details of that design. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define MAXR 3
#define MAXT 64
#define MAXC 64
#define KMAX 10
#define NSMAX 1024
static double eps, vp, A, c1, dbar;
static int R, T, K, NS, MASK, NC, TZ0, TZ1, TU, TS;
static int ty[MAXR][MAXT];
static double (*pi_)[MAXT][2*NSMAX]; /* law of (target value, window) before gate f of rail r */
static double dfix[MAXC];

/* one period of the chain of rail r; a is the law of (target value, window) */
static void period(int r, double *a, const double *d, double *dens, int store){
  double b[2*NSMAX]; dbar=0; for(int k=0;k<NC;k++) dbar+=d[k]/NC;
  for(int f=0;f<T;f++){
    if(store) memcpy(pi_[r][f],a,sizeof(double)*2*NS);
    int t=ty[r][f]; memset(b,0,sizeof(double)*2*NS); double p1=0;
    if(t<NC||t==TU){ double dd=(t<NC)?d[t]:dbar;
      for(int st=0;st<NS;st++){ double w=a[st]+a[NS+st]; if(w==0) continue;
        for(int tv=0;tv<2;tv++){ double wt=w*(tv?dd:1-dd); double q1=(st&1)?0.0:(1-A-c1*tv);
          b[tv*NS+(((st<<1)|1)&MASK)]+=wt*q1; b[tv*NS+((st<<1)&MASK)]+=wt*(1-q1); p1+=wt*q1; } } }
    else for(int tv=0;tv<2;tv++) for(int st=0;st<NS;st++){ double w=a[tv*NS+st]; if(w==0) continue;
        double rho=(t==TS)?A+c1*tv:(t==TZ0?A+c1:(t==TZ1?A:A+c1*((st>>(t-NC-2))&1)));
        double q1=(st&1)?0.0:(1-rho);
        b[tv*NS+(((st<<1)|1)&MASK)]+=w*q1; b[tv*NS+((st<<1)&MASK)]+=w*(1-q1); p1+=w*q1; }
    memcpy(a,b,sizeof(double)*2*NS); dens[r*T+f]=p1;
  }
}
/* fixed point by lagged iteration; returns the number of iterations, or -1 */
static int fixedpoint(const double *d0, double *d){
  static double a[MAXR][2*NSMAX], dn[MAXC];
  for(int r=0;r<R;r++) for(int s=0;s<2*NS;s++) a[r][s]=0.5/NS;
  memcpy(d,d0,sizeof(double)*NC);
  for(int it=0;it<6000;it++){
    for(int r=0;r<R;r++){ period(r,a[r],d,dn,0); period(r,a[r],d,dn,0); }
    double diff=0; for(int k=0;k<NC;k++){ double e=fabs(dn[k]-d[k]); if(e>diff) diff=e; d[k]=dn[k]; }
    if(diff<1e-13){
      for(int r=0;r<R;r++){ for(int rep=0;rep<40;rep++) period(r,a[r],d,dn,0); period(r,a[r],d,dn,1); }
      return it;
    }
  }
  return -1;
}
/* The position read by gate (r,f), of type E or U, has true value v and is flipped in the second
   copy. cnt[k][x]: mean number of positions of class k at which the copies differ and the first
   copy has value x. Both copies use the same noise. Mode 2: the target of the current group
   differs between the copies; mode 0, 1: common target with that value. */
static void disagreements(int r, int f, int v, double cnt[MAXC][2]){
  static double *P=0, *Q=0; size_t sz=(size_t)3*NS*NS;
  if(!P){ P=malloc(sizeof(double)*sz); Q=malloc(sizeof(double)*sz); }
  memset(P,0,sizeof(double)*sz); for(int k=0;k<NC;k++) cnt[k][0]=cnt[k][1]=0;
  for(int w=0;w<NS;w++) P[((size_t)2*NS+w)*NS+w]=pi_[r][f][w]+pi_[r][f][NS+w];
  for(int step=0;step<6000;step++){
    int ph=(f+step)%T, t=ty[r][ph]; double mass=0; memset(Q,0,sizeof(double)*sz);
    int fresh=(step>0)&&(t<NC||t==TU);
    double dd=(t<NC)?dfix[t]:dbar;
    for(int m=0;m<3;m++) for(int w=0;w<NS;w++) for(int u=0;u<NS;u++){ double ms=P[((size_t)m*NS+w)*NS+u]; if(ms==0) continue;
      double pr[4]; int ra[4], rb[4], nm[4]; int no=0;
      if(fresh){ /* new common target */
        for(int tv=0;tv<2;tv++){ double pt=tv?dd:1-dd, ro=A+c1*tv;
          pr[no]=pt*ro; ra[no]=rb[no]=1; nm[no]=tv; no++;
          pr[no]=pt*(1-ro); ra[no]=rb[no]=0; nm[no]=tv; no++; } }
      else if(step==0||t==TS){
        if(m==2){ pr[0]=A; ra[0]=rb[0]=1; pr[1]=eps-A; ra[1]=rb[1]=0; pr[2]=c1; ra[2]=v; rb[2]=1-v; no=3;
          nm[0]=nm[1]=nm[2]=2; }
        else { double ro=A+c1*m; pr[0]=ro; ra[0]=rb[0]=1; pr[1]=1-ro; ra[1]=rb[1]=0; no=2; nm[0]=nm[1]=m; } }
      else if(t==TZ0||t==TZ1){ double ro=(t==TZ0)?A+c1:A; pr[0]=ro; ra[0]=rb[0]=1; pr[1]=1-ro; ra[1]=rb[1]=0; no=2; nm[0]=nm[1]=m; }
      else { int dbit=t-NC-2; pr[0]=A; ra[0]=rb[0]=1; pr[1]=eps-A; ra[1]=rb[1]=0;
        pr[2]=c1; ra[2]=(w>>dbit)&1; rb[2]=(u>>dbit)&1; no=3; nm[0]=nm[1]=nm[2]=m; }
      for(int o=0;o<no;o++){ double p=ms*pr[o]; if(p<=0) continue;
        int x=!((w&1)|ra[o]), y=!((u&1)|rb[o]);
        int nw=((w<<1)|x)&MASK, nu=((u<<1)|y)&MASK;
        if(x!=y) cnt[r*T+ph][x]+=p;
        if(nw!=nu||nm[o]==2){ Q[((size_t)nm[o]*NS+nw)*NS+nu]+=p; mass+=p; } }
    }
    { double *tmp=P; P=Q; Q=tmp; } if(mass<1e-11) break;
  }
}
static double specrad(const double *M0, int n){
  /* ||M^(2^12)|| / ||M^(2^11)|| to the power 2^-11; an eigenvalue of smaller modulus mu changes
     the result by about (mu/rho)^2048 */
  static double X[MAXC*MAXC], Y[MAXC*MAXC]; memcpy(X,M0,sizeof(double)*n*n); double lg=0, lgp=0;
  for(int s=0;s<=12;s++){ /* X = M^(2^s) / exp(lg) */
    double nr=0; for(int i=0;i<n*n;i++) nr+=X[i]*X[i]; nr=sqrt(nr);
    if(nr<1e-280) return 0;
    for(int i=0;i<n*n;i++) X[i]/=nr; lg+=log(nr);
    if(s==11) lgp=lg;
    if(s==12) return exp((lg-lgp)/2048.0);
    for(int i=0;i<n;i++) for(int j=0;j<n;j++){ double a=0; for(int k=0;k<n;k++) a+=X[i*n+k]*X[k*n+j]; Y[i*n+j]=a; }
    memcpy(X,Y,sizeof(double)*n*n); lg*=2;
  }
  return 0;
}
static void multipliers(double *ru, double *rs, int verbose){
  static double M[MAXC*MAXC], J[MAXC*MAXC]; memset(M,0,sizeof M); memset(J,0,sizeof J);
  static double cnt[MAXC][2], cn1[MAXC][2]; dbar=0; for(int k=0;k<NC;k++) dbar+=dfix[k]/NC;
  double asym=0;
  for(int r=0;r<R;r++) for(int f=0;f<T;f++){ int t=ty[r][f]; if(t>=NC&&t!=TU) continue;
    int lo=(t<NC)?t:0, hi=(t<NC)?t:NC-1; double wt=(t<NC)?1.0:1.0/NC;
    disagreements(r,f,0,cnt);
    if(verbose){ disagreements(r,f,1,cn1); for(int k=0;k<NC;k++){ double e=fabs(cnt[k][0]+cnt[k][1]-cn1[k][0]-cn1[k][1]); if(e>asym) asym=e; } }
    for(int tt=lo;tt<=hi;tt++) for(int k=0;k<NC;k++){ M[k*NC+tt]+=wt*(cnt[k][0]+cnt[k][1]); J[k*NC+tt]+=wt*(cnt[k][0]-cnt[k][1]); } }
  *ru=specrad(M,NC); *rs=specrad(J,NC);
  if(verbose){ printf("  largest difference between the counts for true value 0 and 1: %.2e\n",asym);
    if(NC<=12){ printf("  Jacobian of the density map (row = class above, column = class below):\n");
      for(int i=0;i<NC;i++){ printf("   "); for(int j=0;j<NC;j++) printf(" %8.4f",J[i*NC+j]); printf("\n"); }
      printf("  mean numbers of changed positions:\n");
      for(int i=0;i<NC;i++){ printf("   "); for(int j=0;j<NC;j++) printf(" %8.4f",M[i*NC+j]); printf("\n"); } } }
}
static void tyname(int t, char *s){
  if(t<NC){ if(R>1) sprintf(s,"E%d.%d",t/T,t%T); else sprintf(s,"E%d",t); }
  else if(t==TZ0) sprintf(s,"Z0"); else if(t==TZ1) sprintf(s,"Z1"); else if(t==TU) sprintf(s,"U");
  else if(t==TS) sprintf(s,"S"); else sprintf(s,"L%d",t-NC-1);
}
static void printdesign(void){ char s[16];
  for(int r=0;r<R;r++){ if(r) printf("| "); for(int f=0;f<T;f++){ tyname(ty[r][f],s); printf("%s ",s);} } }
static int ext, blk, Fblk;
/* returns 1 if the design is admissible: it has a gate of type E or U, and no rail has an S
   without a gate of type E or U */
static int decode(long code, int mode){
  int hasE=0;
  if(mode==0){ int B=NC+2+K+ext; for(int r=0;r<R;r++) for(int f=0;f<T;f++){ ty[r][f]=code%B; code/=B; } }
  else if(mode==1){ int B=NC+2+K+ext; for(int f=0;f<T;f++){ int t=code%B; code/=B; ty[0][f]=t; ty[1][f]=(t<NC)?((t<T)?t+T:t-T):t; } }
  else { int B=Fblk+2+K+(ext==2); for(int f=0;f<T;f++) ty[0][f]=TU;
    for(int f=0;f<blk;f++){ int t=code%B; code/=B; ty[0][f]=(t<Fblk)?t:(t<Fblk+2+K?NC+(t-Fblk):TS); } }
  for(int r=0;r<R;r++){ int e=0, s=0; for(int f=0;f<T;f++){ int t=ty[r][f]; if(t<NC||t==TU){ e=1; hasE=1; } if(t==TS) s=1; }
    if(s&&!e) return 0; }
  return hasE;
}
int main(int argc, char **argv){
  if(argc<10){ fprintf(stderr,"usage: organ_multiplier eps R T K mode ext thr a b [code]\n"); return 1; }
  eps=atof(argv[1]); R=atoi(argv[2]); T=atoi(argv[3]); K=atoi(argv[4]); int mode=atoi(argv[5]); ext=atoi(argv[6]);
  double thr=atof(argv[7]); int shard=atoi(argv[8]), nsh=atoi(argv[9]); long only=(argc>10)?atol(argv[10]):-1;
  if(mode==1) R=2;
  if(mode==2){ R=1; if(ext<1) ext=1; }
  if(R>MAXR||T>MAXT||R*T>MAXC||K>KMAX||K<1){ fprintf(stderr,"too large\n"); return 1; }
  NC=R*T; NS=1<<K; MASK=NS-1; pi_=malloc(sizeof(*pi_)*MAXR); TZ0=NC; TZ1=NC+1; TU=NC+2+K; TS=NC+3+K;
  vp=((2*eps+1)-sqrt(4*eps+1))/(2*eps); A=eps*vp; c1=1-eps;
  int B=NC+2+K+ext; long tot=1; for(int i=0;i<(mode==1?T:NC);i++) tot*=B;
  if(mode==2){ blk=shard; Fblk=nsh; shard=0; nsh=1; B=Fblk+2+K+(ext==2); tot=1; for(int i=0;i<blk;i++) tot*=B; }
  static double starts[5][MAXC];
  for(int k=0;k<NC;k++){ starts[0][k]=0.35;
    for(int s=1;s<5;s++){ unsigned h=(unsigned)(s*2654435761u)^(unsigned)(k*40503u+s*977u); h^=h>>13; h*=0x5bd1e995u; h^=h>>15; starts[s][k]=(h&1)?0.9:0.03; } }
  if(mode==1) for(int f=0;f<T;f++){ starts[1][f]=0.9; starts[1][T+f]=0.03; starts[2][f]=0.03; starts[2][T+f]=0.9; }
  long ndes=0, nflag=0; double bestu=0, bests=0; long cbu=-1, cbs=-1;
  for(long code=0;code<tot;code++){
    if(only>=0 && code!=only) continue;
    if(only<0 && code%nsh!=shard) continue;
    if(!decode(code,mode)) continue;
    ndes++;
    static double fp[5][MAXC]; int nfp=0, noconv=0; double ru[5], rs[5];
    for(int s=0;s<5;s++){ double d[MAXC]; int it=fixedpoint(starts[s],d);
      if(it<0){ noconv=1; continue; }
      int dup=0; for(int j=0;j<nfp;j++){ double mx=0; for(int k=0;k<NC;k++){ double e=fabs(fp[j][k]-d[k]); if(e>mx) mx=e; } if(mx<1e-6) dup=1; }
      if(dup) continue;
      memcpy(fp[nfp],d,sizeof(double)*NC); memcpy(dfix,d,sizeof(double)*NC);
      if(only>=0){ printf("fixed point %d (start %d, %d iterations):",nfp,s,it); for(int k=0;k<NC&&k<16;k++) printf(" %.5f",d[k]); printf("\n"); }
      multipliers(&ru[nfp],&rs[nfp],only>=0);
      if(only>=0) printf("  rho_u = %.6f   rho_s = %.6f\n",ru[nfp],rs[nfp]);
      nfp++; }
    double mu=0, ms=0; for(int j=0;j<nfp;j++){ if(ru[j]>mu) mu=ru[j]; if(rs[j]>ms) ms=rs[j]; }
    if(mu>bestu){ bestu=mu; cbu=code; }
    if(ms>bests){ bests=ms; cbs=code; }
    if(mu>thr||ms>thr||nfp>1||noconv){ nflag++;
      printf("%ld  fp=%d%s  rho_u=%.5f rho_s=%.5f  ",code,nfp,noconv?" NOCONV":"",mu,ms); printdesign();
      if(nfp>0){ printf(" d="); for(int k=0;k<NC&&k<12;k++) printf("%.3f ",fp[0][k]); }
      if(nfp>1){ printf(" d2="); for(int k=0;k<NC&&k<12;k++) printf("%.3f ",fp[1][k]); }
      printf("\n"); }
  }
  if(only<0) printf("# eps=%.4f R=%d T=%d K=%d mode=%d ext=%d a=%d b=%d: %ld designs, %ld printed; max rho_u=%.5f (code %ld), max rho_s=%.5f (code %ld)\n",
    eps,R,T,K,mode,ext,atoi(argv[8]),atoi(argv[9]),ndes,nflag,bestu,cbu,bests,cbs);
  return 0;
}
