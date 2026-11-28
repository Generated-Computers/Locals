// referee: own evaluation on the author's layered instance generator (same RNG => same instance),
// rounds of: plain lower/upper traces (= max D), two-back traces (Def twotraces), set-valued R=2, table-Jacobi R=2.
// Also densities of output class per bundle and round, compared with the trace map Psi (Step 1 of thm:twolayered).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
static uint64_t S[2];
static inline uint64_t rnd(void){uint64_t s1=S[0],s0=S[1];S[0]=s0;s1^=s1<<23;S[1]=s1^s0^(s1>>17)^(s0>>26);return S[1]+s0;}
static inline double unif(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static long n; static int *j; static unsigned char *x;
static double eps,W,A,B,c; static const char*prog; static int T;
// Psi for program over letters E (single class 0), L, N, Z(mu0=1), O(mu1=1)
static double PSL,PSU; static int GEN=0;
static void psi(double vm,double vp,double*am,double*ap){ double Bm=GEN?eps*(1-PSL):B, Bp=GEN?eps*(1-PSU):B, Am=GEN?eps*PSL:A, Ap=GEN?eps*PSU:A;
  double qm[16],qp[16];
  for(int g=0;g<T;g++){ char t=prog[g]; double ym,yp; if(t=='E'){ym=1-vm;yp=1-vp;} else if(t=='N'||t=='O'){ym=yp=1;} else {ym=yp=0;} qm[g]=Bm+c*ym; qp[g]=Bp+c*yp; }
  for(int g=0;g<T;g++){am[g]=1;ap[g]=0;}
  for(int it=0;it<400;it++) for(int g=0;g<T;g++){ int g1=(g-1+3*T)%T,g2=(g-2+3*T)%T,g3=(g-3+3*T)%T;
    double a,b;
    if(prog[g]!='L'){ a=1-qp[g]*ap[g1]; b=1-qm[g]*am[g1]; }
    else if(prog[g1]!='L'){ a=1-(Bp*ap[g1]+c*(1-qm[g1])*ap[g2]); b=1-(Bm*am[g1]+c*(1-qp[g1])*am[g2]); }
    else { a=1-(Bp*ap[g1]+c*Am*ap[g2]+c*c*(1-am[g3])); b=1-(Bm*am[g1]+c*Ap*am[g2]+c*c*(1-ap[g3])); }
    am[g]=a; ap[g]=b; }
}
int main(int argc,char**argv){
  eps=atof(argv[1]); prog=argv[2]; T=strlen(prog); int h=atoi(argv[3]),m=atoi(argv[4]),N=atoi(argv[5]);
  double pf=atof(argv[6]); uint64_t sd=atoll(argv[7]); int doplain=argc>8?atoi(argv[8]):1; int dens=argc>9?atoi(argv[9]):0;
  S[0]=sd*0x9E3779B97F4A7C15ULL+1;S[1]=sd^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();
  { double a=0,b=1; for(int i=0;i<200;i++){double mm=(a+b)/2; if(eps*(1-mm)*(1-mm)-mm>0)a=mm; else b=mm;} W=a; A=eps*W; B=eps*(1-W); c=1-eps; }
  long body=(long)N*m*T*h; long n0=(long)(pf/(1-pf)*body); if(n0<4) n0=4; n=n0+body;
  j=malloc((n+1)*sizeof(int)); x=malloc(n+1);
  long *out=malloc(sizeof(long)*m), *outp=malloc(sizeof(long)*m);
  long *outs=malloc(sizeof(long)*(long)N*m);
  j[0]=0; for(long i=1;i<=n0;i++) j[i]=0;
  long pos=n0;
  for(int b=0;b<N;b++){
    for(int cc=0;cc<m;cc++) for(int r=0;r<T*h;r++){ pos++; char t=prog[r%T];
      if(t=='E') j[pos]= b? (int)outp[rnd()%m] : 1;
      else if(t=='L') j[pos]=pos-2; else if(t=='N') j[pos]=pos-1; else if(t=='Z') j[pos]=0; else j[pos]=1;
      if(r==T*h-1){ out[cc]=pos; outs[(long)b*m+cc]=pos; } }
    memcpy(outp,out,sizeof(long)*m); }
  for(long i=1;i<=n;i++) if(unif()<eps) j[i]=(int)(rnd()%i);
  x[0]=1; for(long i=1;i<=n;i++) x[i]=!(x[i-1]||x[j[i]]);
  unsigned char *L=malloc(n+1),*U=malloc(n+1),*Lp=malloc(n+1),*Up=malloc(n+1);
  // (1) plain traces of lem:lowup: rounds = max D
  int rplain=-1;
  if(doplain){ for(long i=1;i<=n;i++){L[i]=0;U[i]=1;} L[0]=U[0]=1; int s=0;
    for(;;){ long open=0; for(long i=1;i<=n;i++) open+=(L[i]!=U[i]); if(!open) break; memcpy(Lp,L,n+1); memcpy(Up,U,n+1); s++;
      for(long k=1;k<=n;k++){ long p=j[k]; int up=(p==k-1)?0:Up[p], lp=(p==k-1)?0:Lp[p]; L[k]=!(U[k-1]||up); U[k]=!(L[k-1]||lp); } }
    rplain=s; }
  // (2) two-back traces
  double *lam=calloc(N,sizeof(double)),*ups=calloc(N,sizeof(double)),*lamp=calloc(N,sizeof(double)),*upsp=calloc(N,sizeof(double));
  for(long i=1;i<=n;i++){L[i]=0;U[i]=1;} L[0]=U[0]=1; int s=0; double worstlo=0,worsthi=0; double sumgap=0; long ngap=0;
  for(int b=0;b<N;b++){lamp[b]=0;upsp[b]=1;}
  int rtwo;
  GEN=(dens>=3);
  for(;;){ long open=0,cl=0,cu=0,opre=0; for(long i=1;i<=n;i++){ open+=(L[i]!=U[i]); cl+=L[i]; cu+=U[i]; if(i<=n0) opre+=(L[i]!=U[i]); } PSL=(double)cl/n; PSU=(double)cu/n; if(dens>1) printf("  after round %d: open in preamble %ld, psiL %.4f psiU %.4f (varpi %.4f)\n",s,opre,PSL,PSU,W); if(!open){rtwo=s;break;} memcpy(Lp,L,n+1); memcpy(Up,U,n+1); s++;
    for(long k=1;k<=n;k++){ long p=j[k]; int up=(p==k-1)?0:Up[p], lp=(p==k-1)?0:Lp[p]; int l=!(U[k-1]||up), u=!(L[k-1]||lp);
      if(p==k-2){ long q=j[k-1]; int l2,u2;
        if(q==k-2){l2=u2=0;}
        else if(q<k-3){ l2=(1-U[k-2])*Lp[q]; u2=(1-L[k-2])*Up[q]; }
        else { l2=(1-U[k-2])*Lp[k-3]; if(L[k-3]>l2) l2=L[k-3]; u2=(1-L[k-2])*Up[k-3]; if(U[k-3]<u2) u2=U[k-3]; }
        if(l2>l) l=l2; if(u2<u) u=u2; }
      if(l>x[k]||u<x[k]){ printf("UNSOUND at %ld round %d\n",k,s); return 2; }
      if(l<Lp[k]||u>Up[k]){ printf("NOT MONOTONE at %ld round %d\n",k,s); return 2; }
      L[k]=l; U[k]=u; }
    if(dens){ // densities of output classes
      for(int b=0;b<N;b++){ long cl=0,cu=0; for(int cc=0;cc<m;cc++){ long o=outs[(long)b*m+cc]; cl+=L[o]; cu+=U[o]; } lam[b]=(double)cl/m; ups[b]=(double)cu/m; }
      int pout=T-1; double am[16],ap[16];
      for(int b=1;b<N;b++){ psi(lamp[b-1],upsp[b-1],am,ap); double pl=1-am[pout], pu=1-ap[pout];
        double dl=pl-lam[b], du=ups[b]-pu; // should be <= kappa_1 (small)
        if(dl>worstlo) worstlo=dl; if(du>worsthi) worsthi=du; sumgap+=fabs(dl)+fabs(du); ngap+=2;
        if(dens>1&&(b==N-1)) printf("  round %d bundle %d: lam %.4f >= Psi- %.4f ; ups %.4f <= Psi+ %.4f\n",s,b,lam[b],pl,ups[b],pu); }
      memcpy(lamp,lam,N*sizeof(double)); memcpy(upsp,ups,N*sizeof(double)); }
  }
  // (3) set-valued, reach 2: windows (z_{i-1},z_i) as 2 bits (bit0=z_i); S as 4-bit mask
  unsigned char *K=calloc(n+1,1),*Kn=calloc(n+1,1); K[0]=1; int rset=0;
  for(;;){ long nk=0; for(long i=0;i<=n;i++) nk+=K[i]; if(nk==n+1) break; rset++;
    unsigned Sp=1u<<1; Kn[0]=1;
    for(long i=1;i<=n;i++){ long p=j[i]; unsigned Sn=0; int sh=(p>=i-2);
      for(int w=0;w<4;w++) if(Sp>>w&1){ int a=w&1; int b0=0,b1=0;
        if(sh){ int bb=(w>>(i-1-p))&1; if(bb)b1=1; else b0=1; } else if(K[p]){ if(x[p])b1=1; else b0=1; } else b0=b1=1;
        if(b0) Sn|=1u<<(((w<<1)|(!a))&3); if(b1) Sn|=1u<<(((w<<1)|0)&3); }
      int has1=(Sn&0xA)!=0, has0=(Sn&0x5)!=0; Kn[i]=!(has0&&has1); if(Kn[i]&&has1!=x[i]){printf("setval unsound\n");return 2;}
      if(K[i]&&!Kn[i]){printf("setval not monotone\n");return 2;}
      Sp=Sn; }
    // check (c): settled in round rset => in K ; need traces per round -> skipped here (done exhaustively); 
    unsigned char*t=K;K=Kn;Kn=t; if(rset>100000)break; }
  // (4) table-Jacobi reach 2
  unsigned char *y=calloc(n+1,1),*z=calloc(n+1,1); y[0]=1; int app=0,firsteq=-1;
  for(;;){ z[0]=1; for(long i=1;i<=n;i++){ long p=j[i]; int r=(p>=i-2)?z[p]:y[p]; z[i]=!(z[i-1]||r);} app++;
    if(firsteq<0 && memcmp(y,x,n+1)==0) firsteq=app-1;
    if(memcmp(y,z,n+1)==0) break; unsigned char*t=y;y=z;z=t; }
  if(memcmp(y,x,n+1)!=0){printf("table-Jacobi stopped off the trace\n");return 2;}
  printf("eps %.3f prog %s h %d m %d N %d n %ld: maxD(plain traces rounds) %d | two-back rounds %d | set-valued R=2 rounds %d | table-Jacobi R=2 applications %d (first s with y=x: %d)\n",eps,prog,h,m,N,n,rplain,rtwo,rset,app,firsteq);
  if(dens) printf("   density test: max (Psi- - lam) = %.4f, max (ups - Psi+) = %.4f, mean |gap| %.4f  (sampling sd about %.4f)\n",worstlo,worsthi,sumgap/ngap,0.5/sqrt(m));
  return 0; }
