/* cert_periodic.c: referee sanity check for the lemma "potentials" (floating point, not rigorous).
   Two-row chain of the paper with a fixed density d1 = d2 = d and noise eps, driven by a periodic
   word of symbols (b,w).  Stationary ratio  R = E[#open upper] / E[#open lower]  over one period.
   The lemma implies R <= theta = 0.999 for eps >= 0.455 and d in [l4, mu4].
   Modes:
     cert_periodic word eps d  <symbols as digits 0..3, symbol = b + 2w>   one word
     cert_periodic scan P                 all cyclically admissible words of period <= P, a grid of eps, d
     cert_periodic climb seed iters L     random local search over words of length <= L
   Compile: cc -O2 -o cert_periodic cert_periodic.c -lm                                              */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int fst(int v,int o,int b,int w){ if(!v&&b) return w; if(v&&b) return o&w; if(!v&&!b) return o|w; return o; }
static int NX[4][16][3][3];
static void build(void){ for(int y=0;y<4;y++) for(int s=0;s<16;s++){ int xl=s&1,ol=(s>>1)&1,xu=(s>>2)&1,ou=(s>>3)&1;
    for(int i=0;i<3;i++){ int b=i==0?(y&1):(i==1), w=i==0?(y>>1):0; int nl=!(xl||b), nol=fst(xl,ol,b,w);
      for(int k=0;k<3;k++){ int b2=k==0?nl:(k==1), w2=k==0?nol:0; int nu=!(xu||b2), nou=fst(xu,ou,b2,w2);
        NX[y][s][i][k]=nl+2*nol+4*nu+8*nou; } } } }
static void l4mu4(double e,double*l,double*m){ double mu=0.5,ll=0; for(int s=0;s<4;s++){ ll=(1-e*mu)/(3-e-e*mu); mu=(1-e*ll)/(2-e*ll); } *l=ll; *m=mu; }
/* ratio for a word; returns -1 if the lower row has (almost) no open positions */
static double ratio(const int*wd,int P,double e,double d,double*lowout){
  double pr[3]={1-e,e*d,e*(1-d)}, p[16],q[16]; for(int s=0;s<16;s++) p[s]=1.0/16;
  double up=0,lo=0,pup=-1,plo=-1;
  for(int it=0;it<100000;it++){ up=lo=0;
    for(int t=0;t<P;t++){ memset(q,0,sizeof q); int y=wd[t];
      for(int s=0;s<16;s++) if(p[s]!=0) for(int i=0;i<3;i++) for(int k=0;k<3;k++) q[NX[y][s][i][k]]+=p[s]*pr[i]*pr[k];
      memcpy(p,q,sizeof p); for(int s=0;s<16;s++){ if(s&2) lo+=p[s]; if(s&8) up+=p[s]; } }
    if(it>20&&fabs(up-pup)<1e-14*(1+up)&&fabs(lo-plo)<1e-14*(1+lo)) break; pup=up; plo=lo; }
  if(lowout)*lowout=lo; if(lo<1e-9) return -1; return up/lo; }
static int adm(const int*w,int P){ for(int t=0;t<P;t++) if((w[t]&1)&&(w[(t+1)%P]&1)) return 0; return 1; }
static const double EG[]={0.455,0.46,0.47,0.5,0.55,0.6,0.7,0.8,0.9,0.97,0.999};
#define NE ((int)(sizeof EG/sizeof EG[0]))
static double best=-1; static int bw[256],bP; static double be,bd;
static double evalword(const int*w,int P,int quick){ double m=-1;
  for(int a=0;a<NE;a+=(quick?5:1)) for(int j=0;j<3;j++){ double l,mu; l4mu4(EG[a],&l,&mu); double d=j==0?l:(j==1?0.5*(l+mu):mu);
    double r=ratio(w,P,EG[a],d,NULL); if(r>m)m=r; if(r>best){ best=r; bP=P; memcpy(bw,w,P*sizeof(int)); be=EG[a]; bd=d; } }
  return m; }
static void show(void){ printf("largest ratio %.6f at eps %.4f d %.5f, period %d, word ",best,be,bd,bP); for(int t=0;t<bP;t++) printf("%d",bw[t]); printf("\n"); fflush(stdout); }
int main(int argc,char**argv){ build(); int w[256];
  if(argc>=5&&!strcmp(argv[1],"word")){ double e=atof(argv[2]),d=atof(argv[3]); int P=strlen(argv[4]); for(int t=0;t<P;t++) w[t]=argv[4][t]-'0';
    if(!adm(w,P)){ printf("not admissible\n"); return 1; } double lo; double r=ratio(w,P,e,d,&lo);
    double l,mu; l4mu4(e,&l,&mu); printf("eps %.4f d %.6f (l4 %.6f mu4 %.6f) period %d: open lower per period %.6f, ratio %.6f\n",e,d,l,mu,P,lo,r); return 0; }
  if(argc>=3&&!strcmp(argv[1],"scan")){ int PM=atoi(argv[2]); long n=0;
    for(int P=1;P<=PM;P++){ long tot=1L<<(2*P); for(long c=0;c<tot;c++){ for(int t=0;t<P;t++) w[t]=(c>>(2*t))&3; if(!adm(w,P)) continue; evalword(w,P,0); n++; }
      printf("period <= %d (%ld words x %d eps x 3 d): ",P,n,NE); show(); } return 0; }
  if(argc>=5&&!strcmp(argv[1],"climb")){ srandom(atoi(argv[2])); long iters=atol(argv[3]); int L=atoi(argv[4]);
    for(int rs=0;rs<20;rs++){ int P=2+random()%(L-1); do{ for(int t=0;t<P;t++) w[t]=random()%4; }while(!adm(w,P)); double cur=evalword(w,P,1);
      for(long it=0;it<iters;it++){ int v[256],Q=P; memcpy(v,w,sizeof w); int mv=random()%3;
        if(mv==0){ v[random()%Q]=random()%4; } else if(mv==1&&Q<L){ int pos=random()%(Q+1); memmove(v+pos+1,v+pos,(Q-pos)*sizeof(int)); v[pos]=random()%4; Q++; }
        else if(Q>1){ int pos=random()%Q; memmove(v+pos,v+pos+1,(Q-pos-1)*sizeof(int)); Q--; }
        if(!adm(v,Q)) continue; double r=evalword(v,Q,1); if(r>=cur){ cur=r; P=Q; memcpy(w,v,sizeof w); } }
      evalword(w,P,0); printf("restart %d: local best %.6f; ",rs,cur); show(); } return 0; }
  fprintf(stderr,"usage: see the header\n"); return 1; }
