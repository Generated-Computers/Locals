/* rows_check.c: potentials for the two-row chain of a fixed-distance base (Lemma "potentials").

   States (x1,o1,x2,o2): value and status (o = 1: open) of a gate of the lower row and of the gate of
   the upper row that reads it.  A symbol y = (b,w) is the value and the status of the position read
   by the gate of the lower row.  One step with the symbol y:
     the gate of the lower row reads y with probability 1-eps, a closed 1 with probability eps*d1 and
     a closed 0 with probability eps*(1-d1);  then the gate of the upper row reads the new (x1,o1) with
     probability 1-eps, a closed 1 with probability eps*d2 and a closed 0 with probability eps*(1-d2).
   A gate with value pv and status po before it that reads (b,w) gets the value NOR(pv,b) and the status
     w if pv=0,b=1;   po and w if pv=1,b=1;   po or w if pv=0,b=0;   po if pv=1,b=0.
   For a function g of the state, (T_y g)(s) is the largest expected value of g after one step from s
   over eps in [EA,EB] and d1,d2 in [DA,DB], where d2 may depend on the outcome of the gate of the
   lower row and eps is maximised separately for the two gates; T-_y is the same with the least value.
   For a single eps and a smaller interval of d the operators of the paper lie between T-_y and T_y.

   With rho = 1[o2] - theta*1[o1], the program proves for every box of a cover: there are vectors
   c_0..c_3 (indexed by a symbol), zero on the four states with o1=o2=0, such that for all symbols
   z_1..z_K, y without two adjacent values 1:
     (C1)  T_{z_1} ... T_{z_K} [ T_y(rho + c_y) - c_{z_K} ] <= 0   at all 16 states,
     (C2)  T-_{z_1} ... T-_{z_K} [ c_{z_K} ] >= 0                    at all 16 states,
     (C3)  |c| <= CMAX.
   The vectors are found by a search in floating point (constraint generation and relaxation), which
   is not part of the proof; only the function check() is.  Arithmetic of the check: IEEE doubles,
   every operation rounded one unit in the last place in the safe direction (nextafter); a product
   with an exact zero and a sum with an exact zero are exact.  The check walks the tree of words from
   the last symbol back and leaves a branch as soon as the function is <= 0 (for C1) or >= 0 (for C2)
   at all states: T_y and T-_y are monotone and map 0 to 0.

   The cover: the interval [e0,e1] of eps is cut into intervals; on each, the numbers l_4 and mu_4 of
   the density lemma for fixed-distance bases are enclosed, and [l_4 - 2*OV, mu_4 + 2*OV] is covered
   by boxes of d such that consecutive boxes overlap by at least 2*OV.  So every interval [d-OV,d+OV]
   with d in [l_4 - OV, mu_4 + OV] lies in one box.  An interval of eps on which the search or the
   check fails is halved, together with the width of the boxes of d.
   Compile with:  cc -O2 -ffp-contract=off -o rows_check rows_check.c -lm
   usage: rows_check e0 e1 theta K weps wd     (weps, wd: first widths of the intervals of eps and d)
   The runs quoted in the paper:
     rows_check 0.455 0.5 0.999 8 0.002 0.004
     rows_check 0.5 0.75 0.999 8 0.008 0.012
     rows_check 0.75 1.0 0.999 8 0.008 0.012                                                     */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
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
static int nx[4][16][3][3];
static void gate(int pv,int po,int b,int ob,int*x,int*o){ *x=!(pv|b); if(!pv&&b)*o=ob; else if(pv&&b)*o=po&&ob; else if(!pv&&!b)*o=po||ob; else *o=po; }
static void init(void){ for(int y=0;y<4;y++){ int b=y&1,ob=y>>1; for(int s=0;s<16;s++){ int x1=s&1,o1=(s>>1)&1,x2=(s>>2)&1,o2=(s>>3)&1;
   int r1[3]={b,1,0},q1[3]={ob,0,0}; for(int i=0;i<3;i++){ int a,ao; gate(x1,o1,r1[i],q1[i],&a,&ao); int r2[3]={a,1,0},q2[3]={ao,0,0};
     for(int k=0;k<3;k++){ int c,co; gate(x2,o2,r2[k],q2[k],&c,&co); nx[y][s][i][k]=a|ao<<1|c<<2|co<<3; } } } } }

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
static void TU(int y,const double*g,double*out){ for(int s=0;s<16;s++){ double A[3];
   for(int i=0;i<3;i++) A[i]=mixU(g[nx[y][s][i][0]],g[nx[y][s][i][1]],g[nx[y][s][i][2]]);
   out[s]=mixU(A[0],A[1],A[2]); } }
static void TD(int y,const double*g,double*out){ for(int s=0;s<16;s++){ double A[3];
   for(int i=0;i<3;i++) A[i]=mixD(g[nx[y][s][i][0]],g[nx[y][s][i][1]],g[nx[y][s][i][2]]);
   out[s]=mixD(A[0],A[1],A[2]); } }
static double SF[40][16]; static int failed; static long napp;
static void dfsU(int depth,int first){ int allneg=1; for(int s=0;s<16;s++) if(SF[depth][s]>0){allneg=0;break;}
  if(allneg) return;                         /* T is monotone and T 0 = 0 */
  if(depth==K){ failed=1; return; }
  for(int z=0;z<4&&!failed;z++){ if((z&1)&&(first&1)) continue; TU(z,SF[depth],SF[depth+1]); napp++; dfsU(depth+1,z); } }
static void dfsD(int depth,int first){ int allpos=1; for(int s=0;s<16;s++) if(SF[depth][s]<0){allpos=0;break;}
  if(allpos) return;
  if(depth==K){ failed=1; return; }
  for(int z=0;z<4&&!failed;z++){ if((z&1)&&(first&1)) continue; TD(z,SF[depth],SF[depth+1]); napp++; dfsD(depth+1,z); } }
/* check (C1)-(C3) for the vectors c on the box; returns 1 if proved */
static int check(const double c[4][16]){
  for(int h=0;h<4;h++) for(int s=0;s<16;s++){ if(!(s&10)&&c[h][s]!=0) return 0; if(!(fabs(c[h][s])<=CMAX)) return 0; }
  for(int h=0;h<4;h++) for(int y=0;y<4;y++){ if((h&1)&&(y&1)) continue; double v[16],g[16];
    for(int s=0;s<16;s++){ double r=0; if(s&8) r=1; if(s&2) r=addU(r,-theta); v[s]=addU(r,c[y][s]); }
    TU(y,v,g); for(int s=0;s<16;s++) g[s]=addU(g[s],-c[h][s]);
    TU(h,g,SF[1]); napp++; failed=0; dfsU(1,h); if(failed) return 0; }
  for(int h=0;h<4;h++){ TD(h,c[h],SF[1]); napp++; failed=0; dfsD(1,h); if(failed) return 0; }
  return 1; }

/* ---------- search in floating point (not part of the proof) ---------- */
static double M[4][16][16],rhoS[16],cs[4][16];
typedef struct { float p[16],q[16]; int h,h2; double rhs; } Con;
static Con *pool; static long npool,maxpool=400000; static long nviol; static double P[40][16][16]; static int Z[40];
static double KAP=0.003, KAP2=0.02; static const double TOL=1e-7;
static void addcon(const double*p,const double*q,int h,int h2,double rhs){ if(npool>=maxpool) return; Con*C=&pool[npool++];
  for(int a=0;a<16;a++){ C->p[a]=(a&10)?p[a]:0; C->q[a]=(q&&(a&10))?q[a]:0; } C->h=h; C->h2=h2; C->rhs=rhs; }
static void leaf(void){ int h=Z[K-1];
  for(int s=0;s<16;s++){ double *p=P[K][s]; double po=0,pc=0; for(int a=0;a<16;a++) if(a&10){ po+=p[a]; pc+=p[a]*cs[h][a]; }
    if(po>0&&pc<KAP2*po-TOL*po){ nviol++; addcon(p,NULL,h,-1,-KAP2*po); }
    for(int y=0;y<4;y++){ if((h&1)&&(y&1)) continue; if(po<=0&&!(y&2)) continue;
      double q[16]={0}; for(int a=0;a<16;a++) if(p[a]!=0) for(int b=0;b<16;b++) q[b]+=p[a]*M[y][a][b];
      double lhs=-pc,rhs=0,qo=0; for(int a=0;a<16;a++){ lhs+=q[a]*cs[y][a]; rhs-=q[a]*rhoS[a]; if(a&10) qo+=q[a]; }
      if(lhs-rhs>TOL*(po+qo)){ nviol++; addcon(p,q,h,y,rhs); } } } }
static void dfsS(int depth){ if(depth==K){ leaf(); return; }
  for(int z=0;z<4;z++){ if(depth>0&&(Z[depth-1]&1)&&(z&1)) continue; Z[depth]=z;
    for(int s=0;s<16;s++){ for(int b=0;b<16;b++)P[depth+1][s][b]=0; for(int a=0;a<16;a++){ double pa=P[depth][s][a]; if(pa!=0) for(int b=0;b<16;b++) P[depth+1][s][b]+=pa*M[z][a][b]; } }
    dfsS(depth+1); } }
static void scan(double e,double d){ memset(M,0,sizeof M);
  for(int y=0;y<4;y++) for(int s=0;s<16;s++){ double pr[3]={1-e,e*d,e*(1-d)}; for(int i=0;i<3;i++) for(int k=0;k<3;k++) M[y][s][nx[y][s][i][k]]+=pr[i]*pr[k]; }
  for(int s=0;s<16;s++) for(int b=0;b<16;b++) P[0][s][b]=(s==b);
  dfsS(0); }
static void relax(int its){ for(int it=0;it<its;it++){ double mv=0; for(long i=0;i<npool;i++){ Con*C=&pool[i]; double lhs=0,nrm=0; double *c1=cs[C->h],*c2=C->h2>=0?cs[C->h2]:NULL;
      if(C->h2==C->h){ for(int a=0;a<16;a++){ double co=C->q[a]-C->p[a]; lhs+=co*c1[a]; nrm+=co*co; } }
      else { for(int a=0;a<16;a++){ lhs-=C->p[a]*c1[a]; nrm+=C->p[a]*C->p[a]; if(c2){ lhs+=C->q[a]*c2[a]; nrm+=C->q[a]*C->q[a]; } } }
      double v=lhs-C->rhs; if(v<=0||nrm<1e-30) continue; if(v/sqrt(nrm)>mv)mv=v/sqrt(nrm); double st=1.3*v/nrm;
      if(C->h2==C->h){ for(int a=0;a<16;a++) c1[a]-=st*(C->q[a]-C->p[a]); } else { for(int a=0;a<16;a++){ c1[a]+=st*C->p[a]; if(c2) c2[a]-=st*C->q[a]; } } }
    if(mv<1e-10) break; } }
static int check(const double c[4][16]);
static long nsearch=0, nrounds=0;
/* find vectors for the box [EA,EB] x [DA,DB]; returns 1 if the rigorous check passes */
static int find(void){ if(check(cs)) return 1; nsearch++;
  static const double marg[3]={0.05,0.02,0.007}; double start[4][16]; memcpy(start,cs,sizeof cs);
  for(int lv=0;lv<4;lv++){ if(lv==3){ lv=2; memset(start,0,sizeof start); }      /* last: from zero */
    KAP=lv<2?marg[lv]/2:0.003; memcpy(cs,start,sizeof cs);
    for(int s=0;s<16;s++) rhoS[s]=(1+KAP)*((s>>3)&1)-(theta-marg[lv])*((s>>1)&1);
    npool=0; int R=lv<2?6:40;
    for(int r=0;r<R;r++){ nrounds++; nviol=0; if(npool>maxpool/2) npool=0;
      scan(0.5*(EA+EB),DA); if(DB>DA) scan(0.5*(EA+EB),DB);
      relax(300); if(check(cs)) return 1; if(nviol==0&&r>3) break; }
    if(lv==2){ int z=1; for(int h=0;h<4;h++) for(int s=0;s<16;s++) if(start[h][s]!=0) z=0; if(z) break; } }
  return 0; }

static long nboxes=0; static double cmaxseen=0;
/* enclosure of l_4 and mu_4 on an interval of eps */
static void dens(double ea,double eb,double*lo,double*hi){ iv e=mk(ea,eb), mu=mk(0.5,0.5), l=mk(0,0), one=mk(1,1), two=mk(2,2), three=mk(3,3);
  for(int s=0;s<4;s++){ l=idiv(isub(one,imul(e,mu)),isub(isub(three,e),imul(e,mu))); mu=idiv(isub(one,imul(e,l)),isub(two,imul(e,l))); }
  *lo=l.lo; *hi=mu.hi; }
static double WD0;
static int prove(double ea,double eb,double weps,int depth){
  int n=(int)ceil((eb-ea)/weps); if(n<1)n=1;
  for(int i=0;i<n;i++){ double a=ea+(eb-ea)*i/n, b=ea+(eb-ea)*(i+1)/n; if(i==n-1) b=eb; if(i>0) a=dn(a);
    double lo,hi; dens(a,b,&lo,&hi); lo-=2*OV; hi+=2*OV; int ok=1; long nb=0; double w=WD0, d0=lo;
    EA=a; EB=b;
    for(;;){ DA=d0; DB=d0+w; double save[4][16]; memcpy(save,cs,sizeof cs);
      if(find()){ nb++; for(int h=0;h<4;h++) for(int s=0;s<16;s++) if(fabs(cs[h][s])>cmaxseen) cmaxseen=fabs(cs[h][s]);
        if(DB>=hi) break; d0=DB-2.5*OV; w*=1.5; if(w>WD0) w=WD0; }
      else { memcpy(cs,save,sizeof cs); w/=2; if(w<8*OV){ ok=0; break; } } }
    if(ok){ nboxes+=nb; printf("  eps [%.6f,%.6f] d [%.5f,%.5f]: %ld boxes of d (%ld searches, %ld rounds so far)\n",a,b,lo,hi,nb,nsearch,nrounds); fflush(stdout); }
    else { if(depth>=6){ printf("  FAILED on eps [%.6f,%.6f]\n",a,b); return 0; }
      if(!prove(a,b,(b-a)/2,depth+1)) return 0; } }
  return 1; }
int main_orig(int argc,char**argv){ if(argc<7){ fprintf(stderr,"usage: rows_check e0 e1 theta K weps wd\n"); return 1; }
  double e0=atof(argv[1]),e1=atof(argv[2]); theta=atof(argv[3]); K=atoi(argv[4]); double weps=atof(argv[5]),wd=atof(argv[6]);
  init(); pool=malloc(sizeof(Con)*maxpool); memset(cs,0,sizeof cs);
  WD0=wd; int ok=prove(e0,e1,weps,0);
  printf("%s for eps in [%.6f, %.6f] with theta = %.6f and K = %d; %ld boxes, largest |c| = %.3f, %ld applications of T\n",ok?"PROVED":"NOT PROVED",e0,e1,theta,K,nboxes,cmaxseen,napp);
  return ok?0:1; }

static unsigned long long X=88172645463325252ULL; static double rnd(void){ X^=X<<13; X^=X>>7; X^=X<<17; return (X>>11)*(1.0/9007199254740992.0); }
int main(void){ init(); for(int t=0;t<400;t++){ double g[16],u[16],l[16]; int kind=t%8;
  EA=0.3+0.69*rnd(); EB=EA+0.01*rnd(); if(EB>1||kind==1) EB=1.0; if(kind==2){EA=EB=1.0;} DA=0.3+0.2*rnd(); DB=DA+0.005*rnd(); if(kind==3){DA=0;DB=1;}
  for(int s=0;s<16;s++){ g[s]=8*rnd()-4; if(kind==4&&rnd()<0.5) g[s]=0; if(kind==5) g[s]*=1e-310; if(kind==6) g[s]=(rnd()<0.5?-1:1)*4.9406564584124654e-324; if(kind==7&&rnd()<0.6) g[s]=-fabs(g[s])*1e-300; }
  int y=t%4; TU(y,g,u); TD(y,g,l); printf("%a %a %a %a %d",EA,EB,DA,DB,y); for(int s=0;s<16;s++) printf(" %a",g[s]); for(int s=0;s<16;s++) printf(" %a",u[s]); for(int s=0;s<16;s++) printf(" %a",l[s]); printf("\n"); }
  /* the transition table */
  for(int y=0;y<4;y++) for(int s=0;s<16;s++){ printf("NX %d %d",y,s); for(int i=0;i<3;i++) for(int k=0;k<3;k++) printf(" %d",nx[y][s][i][k]); printf("\n"); }
  return 0; }
