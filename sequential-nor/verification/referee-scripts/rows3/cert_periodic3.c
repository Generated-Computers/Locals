/* cert_periodic3.c: referee sanity check for the lemma "potentials for three rows" (floating point,
   not rigorous).  Three-row chain of the paper with one fixed density d for all rows and noise eps,
   driven by a periodic word of symbols (b,w).  Stationary ratio over one period
       R = E[#open positions of row 3] / E[#open positions of row 2].
   The lemma implies R <= theta = 0.999 for 0.30 <= eps <= 0.455 and d in [l4, mu4].
   Modes:
     cert_periodic3 word eps d <symbols as digits 0..3, symbol = b + 2w>   one word, all 64 point starts
     cert_periodic3 rand seed N L        N random cyclically admissible words of length 2..L
     cert_periodic3 scan P               all cyclically admissible words of period <= P
     cert_periodic3 climb seed iters L   random local search over words of length <= L, 12 restarts
     cert_periodic3 uniq seed N L        N random words: largest difference of R between point starts
   Compile: cc -O2 -o cert_periodic3 cert_periodic3.c -lm                                            */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int fst(int v,int o,int b,int w){ if(!v&&b) return w; if(v&&b) return o&w; if(!v&&!b) return o|w; return o; }
static int NX[4][64][27];
static void build(void){ for(int y=0;y<4;y++) for(int s=0;s<64;s++) for(int k=0;k<27;k++){
    int k1=k%3,k2=(k/3)%3,k3=k/9; int x1=s&1,o1=(s>>1)&1,x2=(s>>2)&1,o2=(s>>3)&1,x3=(s>>4)&1,o3=(s>>5)&1;
    int b=k1==0?(y&1):(k1==1), w=k1==0?(y>>1):0;   int n1=!(x1||b), m1=fst(x1,o1,b,w);
    b=k2==0?n1:(k2==1); w=k2==0?m1:0;               int n2=!(x2||b), m2=fst(x2,o2,b,w);
    b=k3==0?n2:(k3==1); w=k3==0?m2:0;               int n3=!(x3||b), m3=fst(x3,o3,b,w);
    NX[y][s][k]=n1+2*m1+4*n2+8*m2+16*n3+32*m3; } }
static void l4mu4(double e,double*l,double*m){ double mu=0.5,ll=0; for(int s=0;s<4;s++){ ll=(1-e*mu)/(3-e-e*mu); mu=(1-e*ll)/(2-e*ll); } *l=ll; *m=mu; }
/* ratio for a word from the start distribution p0 (NULL: uniform); -1 if row 2 has (almost) no open positions */
static double ratio(const int*wd,int P,double e,double d,const double*p0,double*lowout){
  double pr[3]={1-e,e*d,e*(1-d)}, pk[27], p[64],q[64]; for(int k=0;k<27;k++) pk[k]=pr[k%3]*pr[(k/3)%3]*pr[k/9];
  for(int s=0;s<64;s++) p[s]=p0?p0[s]:1.0/64;
  double up=0,lo=0,pup=-1,plo=-1;
  for(int it=0;it<200000;it++){ up=lo=0;
    for(int t=0;t<P;t++){ memset(q,0,sizeof q); int y=wd[t];
      for(int s=0;s<64;s++) if(p[s]!=0){ const int*n=NX[y][s]; for(int k=0;k<27;k++) q[n[k]]+=p[s]*pk[k]; }
      memcpy(p,q,sizeof p); for(int s=0;s<64;s++){ if(s&8) lo+=p[s]; if(s&32) up+=p[s]; } }
    if(it>10&&fabs(up-pup)<1e-13*(1+up)&&fabs(lo-plo)<1e-13*(1+lo)) break; pup=up; plo=lo; }
  if(lowout)*lowout=lo; if(lo<1e-9) return -1; return up/lo; }
static int adm(const int*w,int P){ if(P==1) return !(w[0]&1); for(int t=0;t<P;t++) if((w[t]&1)&&(w[(t+1)%P]&1)) return 0; return 1; }
static const double EG[]={0.30,0.40,0.35,0.455};
#define NE 4
static double best=-1,bestE[NE]; static int bw[256],bP; static double be,bd;
static double evalword(const int*w,int P,int ne){ double m=-1;
  for(int a=0;a<ne;a++) for(int j=0;j<3;j++){ double l,mu; l4mu4(EG[a],&l,&mu); double d=j==0?l:(j==1?0.5*(l+mu):mu);
    double r=ratio(w,P,EG[a],d,NULL,NULL); if(r>m)m=r; if(r>bestE[a])bestE[a]=r; if(r>best){ best=r; bP=P; memcpy(bw,w,P*sizeof(int)); be=EG[a]; bd=d; } }
  return m; }
static void show(void){ printf("largest ratio %.6f at eps %.4f d %.5f, period %d, word ",best,be,bd,bP); for(int t=0;t<bP;t++) printf("%d",bw[t]);
  printf("  [by eps:"); for(int a=0;a<NE;a++) printf(" %.3f: %.6f",EG[a],bestE[a]); printf("]\n"); fflush(stdout); }
static void randword(int*w,int P){ do{ for(int t=0;t<P;t++) w[t]=random()%4; }while(!adm(w,P)); }
int main(int argc,char**argv){ build(); int w[256]; for(int a=0;a<NE;a++) bestE[a]=-1;
  if(argc>=5&&!strcmp(argv[1],"word")){ double e=atof(argv[2]),d=atof(argv[3]); int P=strlen(argv[4]); for(int t=0;t<P;t++) w[t]=argv[4][t]-'0';
    if(!adm(w,P)){ printf("not admissible\n"); return 1; } double lo,l,mu,rmin=9,rmax=-9; l4mu4(e,&l,&mu);
    for(int s0=0;s0<64;s0++){ double p0[64]={0}; p0[s0]=1; double r=ratio(w,P,e,d,p0,&lo); if(r<rmin)rmin=r; if(r>rmax)rmax=r; }
    double r=ratio(w,P,e,d,NULL,&lo);
    printf("eps %.4f d %.6f (l4 %.6f mu4 %.6f) period %d: open row 2 per period %.6f, ratio %.6f (point starts: %.6f .. %.6f)\n",e,d,l,mu,P,lo,r,rmin,rmax); return 0; }
  if(argc>=5&&!strcmp(argv[1],"rand")){ srandom(atoi(argv[2])); long N=atol(argv[3]); int L=atoi(argv[4]);
    for(long i=0;i<N;i++){ int P=2+random()%(L-1); randword(w,P); evalword(w,P,NE); }
    printf("%ld random words of length <= %d, %d eps x 3 d: ",N,L,NE); show(); return 0; }
  if(argc>=3&&!strcmp(argv[1],"scan")){ int PM=atoi(argv[2]); long n=0;
    for(int P=1;P<=PM;P++){ long tot=1L<<(2*P); for(long c=0;c<tot;c++){ for(int t=0;t<P;t++) w[t]=(c>>(2*t))&3; if(!adm(w,P)) continue; evalword(w,P,NE); n++; }
      printf("period <= %d (%ld words x %d eps x 3 d): ",P,n,NE); show(); } return 0; }
  if(argc>=5&&!strcmp(argv[1],"climb")){ srandom(atoi(argv[2])); long iters=atol(argv[3]); int L=atoi(argv[4]);
    for(int rs=0;rs<12;rs++){ int P=2+random()%(L-1); randword(w,P); int ne=2; double cur=evalword(w,P,ne);
      for(long it=0;it<iters;it++){ int v[256],Q=P; memcpy(v,w,sizeof w); int mv=random()%3;
        if(mv==0){ v[random()%Q]=random()%4; } else if(mv==1&&Q<L){ int pos=random()%(Q+1); memmove(v+pos+1,v+pos,(Q-pos)*sizeof(int)); v[pos]=random()%4; Q++; }
        else if(Q>1){ int pos=random()%Q; memmove(v+pos,v+pos+1,(Q-pos-1)*sizeof(int)); Q--; }
        if(!adm(v,Q)) continue; double r=evalword(v,Q,ne); if(r>=cur){ cur=r; P=Q; memcpy(w,v,sizeof w); } }
      evalword(w,P,NE); printf("restart %d: local best %.6f, word ",rs,cur); for(int t=0;t<P;t++) printf("%d",w[t]); printf("; "); show(); } return 0; }
  if(argc>=5&&!strcmp(argv[1],"uniq")){ srandom(atoi(argv[2])); long N=atol(argv[3]); int L=atoi(argv[4]); double worst=0;
    for(long i=0;i<N;i++){ int P=2+random()%(L-1); randword(w,P); double e=EG[i%2],l,mu; l4mu4(e,&l,&mu); double rmin=9,rmax=-9;
      for(int s0=0;s0<64;s0++){ double p0[64]={0}; p0[s0]=1; double r=ratio(w,P,e,l,p0,NULL); if(r<rmin)rmin=r; if(r>rmax)rmax=r; }
      if(rmax-rmin>worst) worst=rmax-rmin; }
    printf("%ld random words: largest difference of the ratio between the 64 point starts %.3e\n",N,worst); return 0; }
  fprintf(stderr,"usage: see the header\n"); return 1; }
