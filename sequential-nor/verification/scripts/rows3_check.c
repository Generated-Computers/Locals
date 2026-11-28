/* rows3_check.c: potentials for the chain of NR rows of a fixed-distance base (NR = 3 by default).
   It is rows_check.c with NR rows above the input row instead of two; for -DNR=2 the check is the
   same as in rows_check.c.

   A state is (x_1,o_1,...,x_NR,o_NR): value and status (o = 1: open) of one gate in each row; the
   gate of row i+1 reads the gate of row i.  A symbol y = (b,w) is the value and the status of the
   position read by the gate of row 1.  One step with the symbol y: for i = 1..NR in this order, the
   gate of row i reads its target (y for i = 1, the new pair of row i-1 otherwise) with probability
   1-eps, a closed 1 with probability eps*d_i and a closed 0 with probability eps*(1-d_i).
   A gate with value pv and status po before it that reads (b,w) gets the value NOR(pv,b) and the status
     w if pv=0,b=1;   po and w if pv=1,b=1;   po or w if pv=0,b=0;   po if pv=1,b=0.
   For a function g of the state, (T_y g)(s) is the largest expected value of g after one step from s
   over eps in [EA,EB] and d_1..d_NR in [DA,DB], where d_i and the noise level of row i may depend on
   the outcomes of the rows below it; T-_y is the same with the least value.

   With rho = 1[o_NR] - theta*1[o_{NR-1}], the program proves for every box of a cover: there are
   vectors c_0..c_3 (indexed by a symbol), zero on the states with no open row, such that for all
   symbols z_1..z_K, y without two adjacent values 1:
     (C1)  T_{z_1} ... T_{z_K} [ T_y(rho + c_y) - c_{z_K} ] <= 0   at all states,
     (C2)  T-_{z_1} ... T-_{z_K} [ c_{z_K} ] >= 0                    at all states,
     (C3)  |c| <= CMAX.
   The vectors are found by a search in floating point (cutting planes taken from the failures of
   the check, and relaxation), which is not part of the proof; only the function check() is.
   Arithmetic of the check, the walk over the words and the cover are those of rows_check.c.
   Compile with:  cc -O2 -ffp-contract=off -o rows3_check rows3_check.c -lm
   usage: rows3_check e0 e1 theta K weps wd     (weps, wd: first widths of the intervals of eps and d)
   The runs quoted in the paper:
     rows3_check 0.30 0.34 0.999 8 0.004 0.016
     rows3_check 0.34 0.38 0.999 8 0.004 0.016
     rows3_check 0.38 0.455 0.999 8 0.004 0.016                                                  */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifndef NR
#define NR 3            /* number of rows above the input row */
#endif
#define NS (1<<(2*NR))
#define OV 2e-5
#define CMAX 4.0
static inline double up(double x){ return nextafter(x, INFINITY); }
static inline double dn(double x){ return nextafter(x, -INFINITY); }
typedef struct{double lo,hi;}iv;
static iv mk(double a,double b){iv r={a,b};return r;}
static iv isub(iv a,iv b){return mk(dn(a.lo-b.hi),up(a.hi-b.lo));}
static iv imul(iv a,iv b){double p[4]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};double l=p[0],h=p[0];
  for(int i=1;i<4;i++){if(p[i]<l)l=p[i];if(p[i]>h)h=p[i];} return mk(dn(l),up(h));}
static iv idiv(iv a,iv b){ if(!(b.lo>0)){fprintf(stderr,"division\n");exit(2);} return imul(a,mk(dn(1.0/b.hi),up(1.0/b.lo))); }

static int K; static double theta;
static int P3[NR+1];
static int *nx;                       /* nx[(y*NS+s)*P3[NR]+k]: next state; k = outcomes of the rows in base 3, row 1 first */
static int CL[NS];                    /* 1 if the state is closed (no row open) */
static void gate(int pv,int po,int b,int ob,int*x,int*o){ *x=!(pv|b); if(!pv&&b)*o=ob; else if(pv&&b)*o=po&&ob; else if(!pv&&!b)*o=po||ob; else *o=po; }
static void init(void){ P3[0]=1; for(int i=1;i<=NR;i++)P3[i]=3*P3[i-1]; nx=malloc(sizeof(int)*4*NS*P3[NR]);
  for(int s=0;s<NS;s++){ CL[s]=1; for(int i=0;i<NR;i++) if((s>>(2*i+1))&1) CL[s]=0; }
  for(int y=0;y<4;y++)for(int s=0;s<NS;s++)for(int k=0;k<P3[NR];k++){ int rb=y&1,ro=y>>1,kk=k,ns=0;
    for(int i=0;i<NR;i++){ int x=(s>>(2*i))&1,o=(s>>(2*i+1))&1,c=kk%3; kk/=3; int b=c==0?rb:(c==1),ob=c==0?ro:0,vx,vo; gate(x,o,b,ob,&vx,&vo); ns|=vx<<(2*i)|vo<<(2*i+1); rb=vx; ro=vo; }
    nx[(y*NS+s)*P3[NR]+k]=ns; } }

/* ---------- rigorous part ---------- */
static double EA,EB,DA,DB;                 /* the box */
static double mulU(double clo,double chi,double g){ if(g==0) return 0; double c=g>0?chi:clo; if(c==0) return 0; return up(c*g); }
static double mulD(double clo,double chi,double g){ if(g==0) return 0; double c=g>0?clo:chi; if(c==0) return 0; return dn(c*g); }
static double addU(double a,double b){ if(a==0) return b; if(b==0) return a; return up(a+b); }
static double addD(double a,double b){ if(a==0) return b; if(b==0) return a; return dn(a+b); }
/* enclosure [lo,hi] of 1-x for 0 <= x <= 1; exact when x = 1 */
static void om(double x,double*lo,double*hi){ if(x==1){ *lo=*hi=0; return; } double r=1-x; *lo=dn(r); if(*lo<0)*lo=0; *hi=up(r); }
/* largest value of (1-e)*gk + e*(d*g1+(1-d)*g0) over e in {EA,EB}, d in {DA,DB}, rounded up */
static double mixU(double gk,double g1,double g0){ double best=-INFINITY,l,h;
  for(int a=0;a<2;a++){ double d=a?DB:DA; om(d,&l,&h); double m=addU(mulU(d,d,g1),mulU(l,h,g0));
    for(int b=0;b<2;b++){ double e=b?EB:EA; om(e,&l,&h); double v=addU(mulU(l,h,gk),mulU(e,e,m)); if(v>best)best=v; } }
  return best; }
/* least value of the same, rounded down */
static double mixD(double gk,double g1,double g0){ double best=INFINITY,l,h;
  for(int a=0;a<2;a++){ double d=a?DB:DA; om(d,&l,&h); double m=addD(mulD(d,d,g1),mulD(l,h,g0));
    for(int b=0;b<2;b++){ double e=b?EB:EA; om(e,&l,&h); double v=addD(mulD(l,h,gk),mulD(e,e,m)); if(v<best)best=v; } }
  return best; }
/* nested maxima: the densities and the noise level of row i may depend on the outcomes of the rows below it */
static double recU(const int*t,const double*g,int row,int k){ if(row==NR) return g[t[k]];
  return mixU(recU(t,g,row+1,k),recU(t,g,row+1,k+P3[row]),recU(t,g,row+1,k+2*P3[row])); }
static double recD(const int*t,const double*g,int row,int k){ if(row==NR) return g[t[k]];
  return mixD(recD(t,g,row+1,k),recD(t,g,row+1,k+P3[row]),recD(t,g,row+1,k+2*P3[row])); }
static void TU(int y,const double*g,double*out){ for(int s=0;s<NS;s++) out[s]=recU(nx+(y*NS+s)*P3[NR],g,0,0); }
static void TD(int y,const double*g,double*out){ for(int s=0;s<NS;s++) out[s]=recD(nx+(y*NS+s)*P3[NR],g,0,0); }
static double SF[40][NS]; static int failed; static long napp;
static void dfsU(int depth,int first){ int allneg=1; for(int s=0;s<NS;s++) if(SF[depth][s]>0){allneg=0;break;}
  if(allneg) return;                         /* T is monotone and T 0 = 0 */
  if(depth==K){ failed=1; return; }
  for(int z=0;z<4&&!failed;z++){ if((z&1)&&(first&1)) continue; TU(z,SF[depth],SF[depth+1]); napp++; dfsU(depth+1,z); } }
static void dfsD(int depth,int first){ int allpos=1; for(int s=0;s<NS;s++) if(SF[depth][s]<0){allpos=0;break;}
  if(allpos) return;
  if(depth==K){ failed=1; return; }
  for(int z=0;z<4&&!failed;z++){ if((z&1)&&(first&1)) continue; TD(z,SF[depth],SF[depth+1]); napp++; dfsD(depth+1,z); } }
/* check (C1)-(C3) for the vectors c on the box; returns 1 if proved */
static int check(const double c[4][NS]){
  for(int h=0;h<4;h++) for(int s=0;s<NS;s++){ if(CL[s]&&c[h][s]!=0) return 0; if(!(fabs(c[h][s])<=CMAX)) return 0; }
  for(int h=0;h<4;h++) for(int y=0;y<4;y++){ if((h&1)&&(y&1)) continue; double v[NS],g[NS];
    for(int s=0;s<NS;s++){ double r=0; if((s>>(2*NR-1))&1) r=1; if((s>>(2*NR-3))&1) r=addU(r,-theta); v[s]=addU(r,c[y][s]); }
    TU(y,v,g); for(int s=0;s<NS;s++) g[s]=addU(g[s],-c[h][s]);
    TU(h,g,SF[1]); napp++; failed=0; dfsU(1,h); if(failed) return 0; }
  for(int h=0;h<4;h++){ TD(h,c[h],SF[1]); napp++; failed=0; dfsD(1,h); if(failed) return 0; }
  return 1; }

/* ---------- search in floating point (not part of the proof) ----------
   Cutting planes: walk the tree as the check does, collect the words and states at which (C1) or (C2)
   fails, turn each into a linear inequality for the vectors at the centre of the box (with a margin),
   and relax.                                                                                        */
static double cs[4][NS];
typedef struct { float p[NS],q[NS]; int h,h2; double rhs; } Con;
static Con *pool; static long npool,maxpool=60000;
static double KAP=0.003, KAP2=0.02, THS;
static double CE,CD;                       /* centre of the box */
static int cpath[40], curh, cury; static long nfound; static const long FCAP=400;
static void step(int y,const double*p,double*q){ double pr[3]={1-CE,CE*CD,CE*(1-CD)}; memset(q,0,sizeof(double)*NS);
  for(int s=0;s<NS;s++) if(p[s]!=0){ const int*t=nx+(y*NS+s)*P3[NR]; for(int k=0;k<P3[NR];k++){ double w=p[s]; int kk=k; for(int i=0;i<NR;i++){ w*=pr[kk%3]; kk/=3; } q[t[k]]+=w; } } }
static void addcut(int s0,int withy){ if(npool>=maxpool) return; double p[NS],q[NS]; memset(p,0,sizeof p); p[s0]=1;
  for(int dpt=K-1;dpt>=1;dpt--){ step(cpath[dpt],p,q); memcpy(p,q,sizeof p); }      /* cpath[dpt]: symbol applied at depth dpt; earliest first */
  step(curh,p,q); memcpy(p,q,sizeof p);
  Con*C=&pool[npool++]; double po=0; for(int a=0;a<NS;a++){ C->p[a]=CL[a]?0:p[a]; if(!CL[a]) po+=p[a]; }
  if(!withy){ for(int a=0;a<NS;a++) C->q[a]=0; C->h=curh; C->h2=-1; C->rhs=-KAP2*po; return; }
  step(cury,p,q); double rhs=0; for(int a=0;a<NS;a++){ C->q[a]=CL[a]?0:q[a]; rhs-=q[a]*((1+KAP)*((a>>(2*NR-1))&1)-THS*((a>>(2*NR-3))&1)); }
  C->h=curh; C->h2=cury; C->rhs=rhs; }
static void cU(int depth,int first){ int allneg=1; for(int s=0;s<NS;s++) if(SF[depth][s]>0){allneg=0;break;}
  if(allneg||nfound>=FCAP) return;
  if(depth==K){ for(int s=0;s<NS;s++) if(SF[depth][s]>0){ nfound++; addcut(s,1); } return; }
  for(int z=0;z<4;z++){ if((z&1)&&(first&1)) continue; cpath[depth]=z; TU(z,SF[depth],SF[depth+1]); cU(depth+1,z); } }
static void cD(int depth,int first){ int allpos=1; for(int s=0;s<NS;s++) if(SF[depth][s]<0){allpos=0;break;}
  if(allpos||nfound>=FCAP) return;
  if(depth==K){ for(int s=0;s<NS;s++) if(SF[depth][s]<0){ nfound++; addcut(s,0); } return; }
  for(int z=0;z<4;z++){ if((z&1)&&(first&1)) continue; cpath[depth]=z; TD(z,SF[depth],SF[depth+1]); cD(depth+1,z); } }
static long collect(void){ long tot=0;
  for(int h=0;h<4;h++) for(int y=0;y<4;y++){ if((h&1)&&(y&1)) continue; double v[NS],g[NS]; curh=h; cury=y;
    for(int s=0;s<NS;s++){ double r=0; if((s>>(2*NR-1))&1) r=1; if((s>>(2*NR-3))&1) r=addU(r,-theta); v[s]=addU(r,cs[y][s]); }
    TU(y,v,g); for(int s=0;s<NS;s++) g[s]=addU(g[s],-cs[h][s]);
    TU(h,g,SF[1]); nfound=0; cU(1,h); tot+=nfound; }
  for(int h=0;h<4;h++){ curh=h; TD(h,cs[h],SF[1]); nfound=0; cD(1,h); tot+=nfound; }
  return tot; }
static void relax(int its){ for(int it=0;it<its;it++){ double mv=0; for(long i=0;i<npool;i++){ Con*C=&pool[i]; double lhs=0,nrm=0; double *c1=cs[C->h],*c2=C->h2>=0?cs[C->h2]:NULL;
      if(C->h2==C->h){ for(int a=0;a<NS;a++){ double co=C->q[a]-C->p[a]; lhs+=co*c1[a]; nrm+=co*co; } }
      else { for(int a=0;a<NS;a++){ lhs-=C->p[a]*c1[a]; nrm+=C->p[a]*C->p[a]; if(c2){ lhs+=C->q[a]*c2[a]; nrm+=C->q[a]*C->q[a]; } } }
      double v=lhs-C->rhs; if(v<=0||nrm<1e-30) continue; if(v/sqrt(nrm)>mv)mv=v/sqrt(nrm); double st=1.3*v/nrm;
      if(C->h2==C->h){ for(int a=0;a<NS;a++) c1[a]-=st*(C->q[a]-C->p[a]); } else { for(int a=0;a<NS;a++){ c1[a]+=st*C->p[a]; if(c2) c2[a]-=st*C->q[a]; } } }
    if(mv<1e-10) break; } }
static int check(const double c[4][NS]);
static long nsearch=0, nrounds=0;
/* find vectors for the box [EA,EB] x [DA,DB]; returns 1 if the rigorous check passes */
static int find(void){ if(check(cs)) return 1; nsearch++; CE=0.5*(EA+EB); CD=0.5*(DA+DB);
  static const double marg[3]={0.04,0.02,0.007}; double start[4][NS]; memcpy(start,cs,sizeof cs);
  for(int lv=0;lv<3;lv++){ THS=theta-marg[lv]; KAP=lv<2?marg[lv]/2:0.003; npool=0; memcpy(cs,start,sizeof cs); int R=lv<2?60:400;
    for(int r=0;r<R;r++){ nrounds++; if(npool>maxpool-2*FCAP*20) npool=0;
      /* collect against the target of this level: temporarily tighten theta */
      double th=theta; theta=THS+0.004; long f=collect(); theta=th; if(f==0){ if(check(cs)) return 1; break; }
      relax(400); int bad=0; for(int h=0;h<4;h++) for(int s=0;s<NS;s++) if(!(fabs(cs[h][s])<1e3)) bad=1; if(bad){ memset(cs,0,sizeof cs); npool=0; } }
    if(check(cs)) return 1; }
  return 0; }

static long nboxes=0; static double cmaxseen=0;
/* enclosure of l_4 and mu_4 on an interval of eps */
static void dens(double ea,double eb,double*lo,double*hi){ iv e=mk(ea,eb), mu=mk(0.5,0.5), l=mk(0,0), one=mk(1,1), two=mk(2,2), three=mk(3,3);
  for(int s=0;s<4;s++){ l=idiv(isub(one,imul(e,mu)),isub(isub(three,e),imul(e,mu))); mu=idiv(isub(one,imul(e,l)),isub(two,imul(e,l))); }
  *lo=l.lo; *hi=mu.hi; }
static double WD0;
static int prove(double ea,double eb,double weps,int depth){
  int n=(int)ceil((eb-ea)/weps); if(n<1)n=1;
  for(int i=0;i<n;i++){ double a=ea+(eb-ea)*i/n, b=ea+(eb-ea)*(i+1)/n; if(i==n-1) b=eb; a=dn(a);
    double lo,hi; dens(a,b,&lo,&hi); lo=dn(lo-up(2*OV)); hi=up(hi+up(2*OV)); int ok=1; long nb=0; double w=WD0, d0=lo;
    EA=a; EB=b;
    for(;;){ DA=d0; DB=d0+w; double save[4][NS]; memcpy(save,cs,sizeof cs);
      if(find()){ nb++; for(int h=0;h<4;h++) for(int s=0;s<NS;s++) if(fabs(cs[h][s])>cmaxseen) cmaxseen=fabs(cs[h][s]);
        if(DB>=hi) break; d0=DB-2.5*OV; w*=1.5; if(w>WD0) w=WD0; }
      else { memcpy(cs,save,sizeof cs); w/=2; if(w<8*OV){ ok=0; break; } } }
    if(ok){ nboxes+=nb; printf("  eps [%.6f,%.6f] d [%.5f,%.5f]: %ld boxes of d (%ld searches, %ld rounds so far)\n",a,b,lo,hi,nb,nsearch,nrounds); fflush(stdout); }
    else { if(depth>=6){ printf("  FAILED on eps [%.6f,%.6f]\n",a,b); return 0; }
      if(!prove(a,b,(b-a)/2,depth+1)) return 0; } }
  return 1; }
int main(int argc,char**argv){ if(argc<7){ fprintf(stderr,"usage: rows3_check e0 e1 theta K weps wd\n"); return 1; }
  double e0=atof(argv[1]),e1=atof(argv[2]); theta=atof(argv[3]); K=atoi(argv[4]); double weps=atof(argv[5]),wd=atof(argv[6]);
  if(!(0<e0&&e0<=e1&&e1<=1&&K>=1&&K<40&&weps>0&&wd>0)){ fprintf(stderr,"bad arguments\n"); return 1; }
  init(); pool=malloc(sizeof(Con)*maxpool); memset(cs,0,sizeof cs);
  WD0=wd; int ok=prove(e0,e1,weps,0);
  printf("%s for eps in [%.6f, %.6f] with theta = %.6f and K = %d; %ld boxes, largest |c| = %.3f, %ld applications of T\n",ok?"PROVED":"NOT PROVED",e0,e1,theta,K,nboxes,cmaxseen,napp);
  return ok?0:1; }
