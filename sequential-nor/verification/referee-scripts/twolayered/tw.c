// referee check of Def twotraces / Lemma twotraces on all parent arrays of length n
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define MAXN 12
#define MAXS 14
int n, j[MAXN], x[MAXN], D[MAXN];
int l[MAXS][MAXN], u[MAXS][MAXN], L[MAXS][MAXN], U[MAXS][MAXN];
int K[4][MAXS][MAXN]; int vK[4][MAXS][MAXN]; // reach R index R-1 .. up to 4
int y[4][MAXS+2][MAXN];
long cnt=0, bad_a=0,bad_b=0,bad_c=0,bad_d=0,bad_LU=0, neq_rounds=0, strict=0, bad_e=0,bad_cR=0;
long hist_diff[20];
static int NOR(int a,int b){return !(a||b);}
static int mn(int a,int b){return a<b?a:b;} static int mx(int a,int b){return a>b?a:b;}
void setval(int R,int S){ // compute K^{(s)} for s=0..S
  int (*Kk)[MAXN]=K[R-1], (*vv)[MAXN]=vK[R-1];
  for(int i=0;i<=n;i++){Kk[0][i]=(i==0); vv[0][i]=1;}
  int W=1<<R;
  for(int s=1;s<=S;s++){
    unsigned S_prev = 1u<<1; // window (0,..,0,1): last coordinate = bit0? define window as integer with last coord = lowest bit
    Kk[s][0]=1; vv[s][0]=1;
    for(int i=1;i<=n;i++){
      unsigned Sn=0; int p=j[i]; int isshort = (p>=i-R);
      for(int w=0;w<W;w++) if(S_prev>>w&1){
        int a=w&1;
        int b0=0,b1=0;
        if(isshort){ int off=i-1-p; int b=(w>>off)&1; if(b)b1=1; else b0=1; } // coordinate z_p in window at i-1: z_{i-1} is bit0, z_{i-1-off} is bit off
        else if(Kk[s-1][p]){ if(vv[s-1][p])b1=1; else b0=1; }
        else {b0=b1=1;}
        if(b0){ int nw=((w<<1)|NOR(a,0))&(W-1); Sn|=1u<<nw; }
        if(b1){ int nw=((w<<1)|NOR(a,1))&(W-1); Sn|=1u<<nw; }
      }
      int has0=0,has1=0; for(int w=0;w<W;w++) if(Sn>>w&1){ if(w&1)has1=1; else has0=1; }
      Kk[s][i]=!(has0&&has1); vv[s][i]=has1;
      S_prev=Sn;
    }
  }
}
void tablejac(int R,int S){
  int (*yy)[MAXN]=y[R-1];
  for(int i=0;i<=n;i++) yy[0][i]=(i==0);
  for(int s=0;s<=S;s++){ yy[s+1][0]=1; for(int i=1;i<=n;i++){ int p=j[i]; int r=(p>=i-R)?yy[s+1][p]:yy[s][p]; yy[s+1][i]=NOR(yy[s+1][i-1],r);} }
}
void check(void){
  cnt++;
  x[0]=1; for(int i=1;i<=n;i++) x[i]=NOR(x[i-1],x[j[i]]);
  // D by lem:status
  D[0]=0; D[1]=1;
  for(int k=2;k<=n;k++){ int p=j[k]; int reset=x[p]==1, rel=x[k-1]==0, notg=(p==k-1);
    if(reset&&rel) D[k]=1+D[p];
    else if(reset&&!rel) D[k]=mn(D[k-1],1+D[p]);
    else if(rel&&!reset&&!notg) D[k]=mx(D[k-1],1+D[p]);
    else D[k]=D[k-1]; }
  int S=n+2;
  for(int s=0;s<=S;s++){ l[s][0]=u[s][0]=L[s][0]=U[s][0]=1; }
  for(int i=1;i<=n;i++){ l[0][i]=L[0][i]=0; u[0][i]=U[0][i]=1; }
  for(int s=1;s<=S;s++) for(int k=1;k<=n;k++){
    int p=j[k];
    // L,U via lem:lowup
    { int up_=(p==k-1)?0:U[s-1][p], lp=(p==k-1)?0:L[s-1][p];
      L[s][k]=NOR(U[s][k-1],up_); U[s][k]=NOR(L[s][k-1],lp); }
    int up_=(p==k-1)?0:u[s-1][p], lp=(p==k-1)?0:l[s-1][p];
    int l1=NOR(u[s][k-1],up_), u1=NOR(l[s][k-1],lp);
    if(p!=k-2){ l[s][k]=l1; u[s][k]=u1; }
    else { int q=j[k-1]; int l2,u2;
      if(q==k-2){l2=u2=0;}
      else if(q<k-3){ l2=(1-u[s][k-2])*l[s-1][q]; u2=(1-l[s][k-2])*u[s-1][q]; }
      else { l2=mx(l[s][k-3],(1-u[s][k-2])*l[s-1][k-3]); u2=mn(u[s][k-3],(1-l[s][k-2])*u[s-1][k-3]); }
      l[s][k]=mx(l1,l2); u[s][k]=mn(u1,u2); }
  }
  for(int R=1;R<=4;R++){ setval(R,S); tablejac(R,S); }
  int sset=-1, sK[5]={-1,-1,-1,-1,-1}, sD=0;
  for(int i=0;i<=n;i++) if(D[i]>sD) sD=D[i];
  for(int s=0;s<=S;s++){
    int all=1;
    for(int i=0;i<=n;i++){
      if(!(l[s][i]<=x[i]&&x[i]<=u[s][i])) bad_a++;
      if(s<S && !(l[s][i]<=l[s+1][i]&&u[s+1][i]<=u[s][i])) bad_a++;
      if(!(L[s][i]<=l[s][i]&&u[s][i]<=U[s][i])) bad_b++;
      if((D[i]<=s)!=(L[s][i]==U[s][i])) bad_LU++;
      if(D[i]<=s && l[s][i]!=u[s][i]) bad_b++;
      int set=(l[s][i]==u[s][i]);
      if(set&&!K[1][s][i]) bad_c++;
      if(set&&!(K[2][s][i]&&K[3][s][i])) bad_cR++;
      if(K[1][s][i]&&!set) strict++;
      if(K[1][s][i]&&!K[2][s][i]) bad_e++; if(K[2][s][i]&&!K[3][s][i]) bad_e++; if(K[0][s][i]&&!K[1][s][i]) bad_e++;
      if(!set) all=0;
    }
    if(all&&sset<0) sset=s;
    for(int R=1;R<=4;R++){ int a=1; for(int i=0;i<=n;i++) if(!K[R-1][s][i]) a=0; if(a&&sK[R]<0) sK[R]=s; }
  }
  if(sset<0){bad_d++; return;}
  for(int R=2;R<=4;R++){
    if(sK[R]<0||sK[R]>sset) bad_d++;
    for(int i=0;i<=n;i++) if(y[R-1][sset][i]!=x[i]) bad_d++;
    // number of applications: first s with y^{s+1}==y^{s}
    int st=-1; for(int s=0;s<=S&&st<0;s++){ int eq=1; for(int i=0;i<=n;i++) if(y[R-1][s][i]!=y[R-1][s+1][i]) eq=0; if(eq) st=s; }
    if(st<0||st+1>sset+1) bad_d++;
  }
  if(sset>sD) bad_b++;
  if(sset!=sK[2]) neq_rounds++;
  hist_diff[sD-sset]++;
}
void rec(int i){ if(i>n){check();return;} for(int p=0;p<i;p++){ j[i]=p; rec(i+1);} }
int main(int argc,char**argv){ int n0=atoi(argv[1]), n1=atoi(argv[2]);
  for(n=n0;n<=n1;n++){ cnt=0; memset(hist_diff,0,sizeof hist_diff); rec(1);
    printf("n=%d arrays=%ld bad_a=%ld bad_b=%ld bad_LU=%ld bad_c=%ld bad_cR(R=3,4)=%ld bad_d=%ld bad_e=%ld | K2-not-settled pairs=%ld, arrays with rounds(settled)!=rounds(K,R=2): %ld\n",n,cnt,bad_a,bad_b,bad_LU,bad_c,bad_cR,bad_d,bad_e,strict,neq_rounds);
    printf("  hist maxD - settle rounds:"); for(int d=0;d<10;d++) printf(" %ld",hist_diff[d]); printf("\n"); }
  return 0; }
