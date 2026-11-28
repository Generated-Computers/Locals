/* window_designs.c -- exact two-layer growth of a flipped position for towers that alternate two kinds of layers
   (ideal model, first order).

   Layer P: a plain chain; every gate lists a uniformly random position of the layer E below.
   Layer E: a chain with a program of period T.  In every period it draws fresh windows of the
   layer P below (window i = W_i consecutive positions at a uniformly random place) and its gates
   read elements of them:  P(i,d) = the value of position d of window i;  T(i,d) = the value of the
   position of the layer E below that the gate at position d of window i lists (two layers down);
   Z = a constant zero;  O = a constant one.
   Every read is resampled with probability eps to a bit that is 1 with probability varpi.

   G2 = expected number of positions of a layer E that change when one position of the layer E two
   below is flipped (both evaluated with the same noise).  With independent reads G2 = lambda^2.

   design string:  widths ':' gates      e.g.  "2:P00,P01"   "1,1:P00,P10"   "2:P00,T00,P01"
   usage: window_designs eps design            (evaluate)
          window_designs eps search T maxw [cut]     (all programs of period T over one window of width <= maxw and
                                    one more window of width 1, and the constant zero Z)                            */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define MAXW 4
#define MAXT 8
#define MAXWIN 3
static double eps,vp,A,B,c;
typedef struct{int nw; int w[MAXWIN]; int T; char kind[MAXT]; int win[MAXT], off[MAXT];} design;
static double dE, alphaP, qP;
static double p0start;            /* P[x_E = 0] before a block, clean */
static double Zk[MAXT+1], Cstart;
static int pw6[MAXW+1]={1,6,36,216,1296};
/* outcome o of one P gate: o%2 = target bit, o/2 = 0 kept, 1 resampled to 1, 2 resampled to 0 */
static inline double oprob(int o,double d){ int t=o&1, s=o>>1; return (t?d:1-d)*(s==0?c:(s==1?A:B)); }
static inline int oread(int o){ int t=o&1, s=o>>1; return s==0?t:(s==1?1:0); }
/* clean window: list of (prob, xP bits, target bits) */
typedef struct{double p; unsigned char x[MAXW], t[MAXW];} wout;
static wout CW[MAXWIN][1296*2]; static int nCW[MAXWIN];
static void clean_windows(const design*D){
  for(int i=0;i<D->nw;i++){ int w=D->w[i], n=0;
    for(int s=0;s<2;s++) for(int code=0;code<pw6[w];code++){ double p=s?(1-alphaP):alphaP; int x=s, cd=code; wout*o=&CW[i][n];
        for(int d=0;d<w;d++){ int oc=cd%6; cd/=6; p*=oprob(oc,dE); x=!(x||oread(oc)); o->x[d]=x; o->t[d]=oc&1; }
        if(p>0){ o->p=p; n++; } }
    nCW[i]=n; }
}
static inline int rd(const design*D,int g,const wout**sel){ char k=D->kind[g];
  if(k=='Z') return 0; if(k=='O') return 1; const wout*o=sel[D->win[g]]; return k=='P'?o->x[D->off[g]]:o->t[D->off[g]]; }
/* iterate over all joint clean outcomes */
static void clean_E(const design*D,double*avg1){
  int T=D->T; double f0[2]={0,0}; double dens[2]={0,0}; double zk[2][MAXT+1]; (void)zk;
  for(int k=0;k<=T;k++) Zk[k]=0;
  int idx[MAXWIN]={0,0,0}; const wout*sel[MAXWIN];
  for(;;){ double p=1; for(int i=0;i<D->nw;i++){ sel[i]=&CW[i][idx[i]]; p*=sel[i]->p; }
    for(int s=0;s<2;s++){ double q0=s?0:1, ones=0;       /* q0 = P[x=0] */
      for(int g=0;g<T;g++){ int v=rd(D,g,sel); double z=B+c*(1-v); double p1=q0*z; ones+=p1; q0=1-p1; }
      f0[s]+=p*q0; dens[s]+=p*ones; }
    { double z=1; for(int g=0;g<T;g++){ int v=rd(D,g,sel); z*=B+c*(1-v); Zk[g+1]+=p*z; } }
    int i=0; while(i<D->nw){ if(++idx[i]<nCW[i]) break; idx[i]=0; i++; } if(i==D->nw) break; }
  /* stationary: P0 = P0 f0[0] + (1-P0) f0[1] */
  p0start=f0[1]/(1-f0[0]+f0[1]);
  *avg1=(p0start*dens[0]+(1-p0start)*dens[1])/T;
  double S=0; for(int k=1;k<=T;k++) S+=Zk[k]; Cstart=S/(1-Zk[T]);
}
static void fixpoint(const design*D){
  dE=0.4;
  for(int it=0;it<200;it++){ qP=B+c*(1-dE); alphaP=1/(1+qP); clean_windows(D); double a; clean_E(D,&a);
    if(fabs(a-dE)<1e-13){ dE=a; break; } dE=0.5*dE+0.5*a; }
  qP=B+c*(1-dE); alphaP=1/(1+qP); clean_windows(D); double a; clean_E(D,&a);
}
/* dirty window iw: two copies.  mode d0 >= 0: dirty gate at offset d0, clean start.  mode -1: the
   run started before the window; start states (1,0) and (0,1) with the given weights. */
typedef struct{double p; unsigned char x[2][MAXW], t[2][MAXW];} dout;
static dout DW[1296*4]; static int nDW;
static void dirty_window(const design*D,int iw,int d0){
  int w=D->w[iw]; nDW=0;
  int ns=(d0<0)?2:2; 
  for(int s=0;s<ns;s++){ double ps; int x0,x1;
    if(d0<0){ double base=c*alphaP/(1-qP*qP); if(s==0){ ps=base; x0=1; x1=0; } else { ps=base*qP; x0=0; x1=1; } }
    else { ps=s?(1-alphaP):alphaP; x0=x1=s; }
    for(int code=0;code<pw6[w];code++){ double p=ps; int a=x0,b=x1,cd=code,skip=0; dout*o=&DW[nDW];
      for(int d=0;d<w;d++){ int oc=cd%6; cd/=6;
        if(d==d0){ if(oc&1){ skip=1; break; }      /* the target bit is (0,1): count each status once */
          int s2=oc>>1; p*=(s2==0?c:(s2==1?A:B)); int r0=s2==0?0:(s2==1?1:0), r1=s2==0?1:(s2==1?1:0);
          a=!(a||r0); b=!(b||r1); o->t[0][d]=0; o->t[1][d]=1; }
        else { p*=oprob(oc,dE); int r=oread(oc); a=!(a||r); b=!(b||r); o->t[0][d]=o->t[1][d]=oc&1; }
        o->x[0][d]=a; o->x[1][d]=b; }
      if(skip||p<=0) continue; o->p=p; nDW++; } }
}
static double dirty_E(const design*D,int iw){
  int T=D->T; double tot=0; int idx[MAXWIN]={0,0,0}; const wout*sel[MAXWIN];
  for(int k=0;k<nDW;k++){ const dout*dw=&DW[k];
    wout c0,c1; memcpy(c0.x,dw->x[0],MAXW); memcpy(c0.t,dw->t[0],MAXW); memcpy(c1.x,dw->x[1],MAXW); memcpy(c1.t,dw->t[1],MAXW);
    for(int i=0;i<D->nw;i++) idx[i]=0;
    for(;;){ double p=dw->p; const wout*s0[MAXWIN],*s1[MAXWIN];
      for(int i=0;i<D->nw;i++){ if(i==iw){ s0[i]=&c0; s1[i]=&c1; } else { sel[i]=&CW[i][idx[i]]; p*=sel[i]->p; s0[i]=s1[i]=sel[i]; } }
      /* joint law of (x,x') : index 2*x+x' */
      double st[4]={p0start,0,0,1-p0start}, cnt=0;
      for(int g=0;g<T;g++){ int v0=rd(D,g,s0), v1=rd(D,g,s1); double nx[4]={0,0,0,0};
        for(int q=0;q<4;q++){ if(st[q]==0) continue; int a=q>>1,b=q&1;
          nx[0]+=st[q]*A;                                   /* resampled to 1 */
          nx[2*(!a)+(!b)]+=st[q]*B;                         /* resampled to 0 */
          nx[2*(!(a||v0))+(!(b||v1))]+=st[q]*c; }
        for(int q=0;q<4;q++) st[q]=nx[q]; cnt+=st[1]+st[2]; }
      tot+=p*(cnt+(st[1]+st[2])*Cstart);
      int i=0; while(i<D->nw){ if(i==iw){ i++; continue; } if(++idx[i]<nCW[i]) break; idx[i]=0; i++; } if(i>=D->nw) break; } }
  return tot;
}
static double G2(const design*D,int verbose){
  fixpoint(D); double g=0;
  for(int iw=0;iw<D->nw;iw++) for(int d0=-1;d0<D->w[iw];d0++){ dirty_window(D,iw,d0); double v=dirty_E(D,iw); g+=v;
    if(verbose) printf("   window %d dirty at %d: %.5f\n",iw,d0,v); }
  return g/D->T;
}
static int parse(const char*s,design*D){ char buf[512]; strncpy(buf,s,511); buf[511]=0; char*col=strchr(buf,':'); if(!col) return 0; *col=0;
  D->nw=0; for(char*t=strtok(buf,",");t;t=strtok(NULL,",")) D->w[D->nw++]=atoi(t);
  D->T=0; for(char*t=strtok(col+1,",");t;t=strtok(NULL,",")){ int g=D->T++; D->kind[g]=t[0]; if(t[0]=='P'||t[0]=='T'){ D->win[g]=t[1]-'0'; D->off[g]=t[2]-'0'; } }
  return 1; }
static void show(const design*D,char*out){ out[0]=0; char b[16]; for(int i=0;i<D->nw;i++){ sprintf(b,"%s%d",i?",":"",D->w[i]); strcat(out,b); } strcat(out,":");
  for(int g=0;g<D->T;g++){ if(D->kind[g]=='P'||D->kind[g]=='T') sprintf(b,"%s%c%d%d",g?",":"",D->kind[g],D->win[g],D->off[g]); else sprintf(b,"%s%c",g?",":"",D->kind[g]); strcat(out,b); } }
int main(int argc,char**argv){
  if(argc<3){ fprintf(stderr,"usage: window_designs eps design | pe eps search T maxw [cut]\n"); return 1; }
  eps=atof(argv[1]); vp=((2*eps+1)-sqrt(4*eps+1))/(2*eps); A=eps*vp; B=eps*(1-vp); c=1-eps;
  double q=(-(1-B)+sqrt((1-B)*(1-B)+4*(1-A)))/2, lam=(1-eps)/(1-q*q);
  if(strcmp(argv[2],"search")){ design D; parse(argv[2],&D); double g=G2(&D,1);
    printf("eps %.4f %s : G2 = %.6f   sqrt %.6f   (lambda^2 = %.6f, lambda = %.6f)  density of E %.5f  alpha_P %.5f\n",eps,argv[2],g,sqrt(g),lam*lam,lam,dE,alphaP); return 0; }
  int T=atoi(argv[3]), maxw=atoi(argv[4]); double cut=argc>5?atof(argv[5]):lam*lam; 
  /* elements: window 0 (width w): P0d, T0d; window 1 (width 1): P10, T10; Z */
  double best=0; char bs[256]="";
  for(int w=1;w<=maxw;w++){ int ne=2*w+3; long tot=1; for(int i=0;i<T;i++) tot*=ne;
    for(long code=0;code<tot;code++){ design D; D.nw=2; D.w[0]=w; D.w[1]=1; D.T=T; long cd=code; int used0=0,usedlast=0,src=0;
      for(int g=0;g<T;g++){ int e=cd%ne; cd/=ne;
        if(e<2*w){ D.kind[g]=(e&1)?'T':'P'; D.win[g]=0; D.off[g]=e/2; used0=1; if(e/2==w-1) usedlast=1; if(e/2==0) src|=1; }
        else if(e==2*w){ D.kind[g]='P'; D.win[g]=1; D.off[g]=0; }
        else if(e==2*w+1){ D.kind[g]='T'; D.win[g]=1; D.off[g]=0; }
        else D.kind[g]='Z'; }
      if(!used0||!usedlast||!(src&1)) continue;        /* window 0 used at both ends: avoids duplicates of smaller w */
      double g2=G2(&D,0); char s[256]; show(&D,s);
      if(g2>cut) printf("%.6f %s\n",g2,s);
      if(g2>best){ best=g2; strcpy(bs,s); } } }
  printf("eps %.4f T %d maxw %d: best G2 %.6f (sqrt %.6f) %s ; lambda^2 %.6f\n",eps,T,maxw,best,sqrt(best),bs,lam*lam);
  return 0;
}
