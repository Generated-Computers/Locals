// R rows above an input row, fixed distance, fixed density d: ratio E[#open top row]/E[#open row below it]
// over periodic input rows. usage: per3 eps d R T mode [seed]   mode 0 exhaustive, mode>0 hill climbing restarts
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static double eps,d; static int R,T,NS,NO; static int *nxt; static double *pr;
static void gate(int pv,int po,int b,int ob,int*x,int*o){ *x=!(pv|b); if(!pv&&b)*o=ob; else if(pv&&b)*o=po&&ob; else if(!pv&&!b)*o=po||ob; else *o=po; }
static void init(void){ NS=1<<(2*R); NO=1; for(int i=0;i<R;i++)NO*=3; nxt=malloc(sizeof(int)*4*NS*NO); pr=malloc(sizeof(double)*NO); double p[3]={1-eps,eps*d,eps*(1-d)};
  for(int k=0;k<NO;k++){ double q=1; int kk=k; for(int i=0;i<R;i++){ q*=p[kk%3]; kk/=3; } pr[k]=q; }
  for(int y=0;y<4;y++)for(int s=0;s<NS;s++)for(int k=0;k<NO;k++){ int rb=y&1,ro=y>>1,kk=k,ns=0;
    for(int i=0;i<R;i++){ int x=(s>>(2*i))&1,o=(s>>(2*i+1))&1,c=kk%3; kk/=3; int b=c==0?rb:(c==1),ob=c==0?ro:0,nx,no; gate(x,o,b,ob,&nx,&no); ns|=nx<<(2*i)|no<<(2*i+1); rb=nx; ro=no; }
    nxt[(y*NS+s)*NO+k]=ns; } }
static double ratio(const int*w,double*N1){ static double p[4096],q[4096]; memset(p,0,sizeof(double)*NS); p[0]=1; double n1=0,n2=0; int b1=1<<(2*(R-2)+1), b2=1<<(2*(R-1)+1);
  for(int rep=0;rep<30;rep++){ n1=n2=0; for(int j=0;j<T;j++){ memset(q,0,sizeof(double)*NS);
      for(int s=0;s<NS;s++) if(p[s]>0){ const int*nx=nxt+(w[j]*NS+s)*NO; for(int k=0;k<NO;k++) q[nx[k]]+=p[s]*pr[k]; }
      memcpy(p,q,sizeof(double)*NS); for(int s=0;s<NS;s++){ if(s&b2) n2+=p[s]; if(s&b1) n1+=p[s]; } } }
  if(N1)*N1=n1; return n1>1e-12?n2/n1:0; }
static int valid(const int*w){ int any=0; for(int j=0;j<T;j++){ if((w[j]&1)&&(w[(j+1)%T]&1)) return 0; if(w[j]&2) any=1; } return any&&T>1; }
int main(int argc,char**argv){ eps=atof(argv[1]); d=atof(argv[2]); R=atoi(argv[3]); T=atoi(argv[4]); int mode=atoi(argv[5]); init(); int w[64],bw[64]={0}; double best=0;
  if(mode==0){ long N=1L<<(2*T); for(long c=0;c<N;c++){ for(int j=0;j<T;j++) w[j]=(c>>(2*j))&3; if(!valid(w)) continue; double r=ratio(w,NULL); if(r>best){best=r;memcpy(bw,w,sizeof w);} } }
  else { srand(argc>6?atoi(argv[6]):1); for(int rs=0;rs<mode;rs++){ do{ for(int j=0;j<T;j++) w[j]=rand()&3; }while(!valid(w)); double cur=ratio(w,NULL);
      for(int it=0;it<300;it++){ int j=rand()%T,old=w[j]; w[j]=rand()&3; double n1,r=valid(w)?ratio(w,&n1):0; if(r>cur&&n1>1e-7) cur=r; else w[j]=old; }
      if(cur>best){best=cur;memcpy(bw,w,sizeof w);} } }
  printf("eps %.3f d %.4f rows %d T %d: max ratio %.5f word ",eps,d,R,T,best); for(int j=0;j<T;j++) putchar("01oI"[bw[j]]); printf("\n"); return 0; }
