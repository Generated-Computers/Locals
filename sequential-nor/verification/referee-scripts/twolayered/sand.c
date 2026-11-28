// referee: deterministic sandwich of Step 1 of thm:twolayered: l^(s)_g >= X^-_g, u^(s)_g <= X^+_g,
// with y^-_g := effective lower read (round s-1), y^+_g := effective upper read, any subset of pointer-(g-2) gates called "kept L2",
// any chain start g1 (trivial bounds before it). Exhaustive over parent arrays of length n.
#include <stdio.h>
#include <stdlib.h>
#define MAXN 12
#define MAXS 14
int n,j[MAXN],x[MAXN],l[MAXS][MAXN],u[MAXS][MAXN]; long cnt,bad,badtrue,cases;
static int NOR(int a,int b){return !(a||b);} static int mx(int a,int b){return a>b?a:b;} static int mn(int a,int b){return a<b?a:b;}
void check(void){ cnt++; int S=n+1;
  x[0]=1; for(int i=1;i<=n;i++) x[i]=NOR(x[i-1],x[j[i]]);
  for(int s=0;s<=S;s++) l[s][0]=u[s][0]=1; for(int i=1;i<=n;i++){l[0][i]=0;u[0][i]=1;}
  for(int s=1;s<=S;s++) for(int k=1;k<=n;k++){ int p=j[k]; int up_=(p==k-1)?0:u[s-1][p], lp=(p==k-1)?0:l[s-1][p];
    int l1=NOR(u[s][k-1],up_),u1=NOR(l[s][k-1],lp);
    if(p==k-2){ int q=j[k-1],l2,u2; if(q==k-2){l2=u2=0;} else if(q<k-3){l2=(1-u[s][k-2])*l[s-1][q];u2=(1-l[s][k-2])*u[s-1][q];}
      else {l2=mx(l[s][k-3],(1-u[s][k-2])*l[s-1][k-3]);u2=mn(u[s][k-3],(1-l[s][k-2])*u[s-1][k-3]);} l1=mx(l1,l2);u1=mn(u1,u2);} 
    l[s][k]=l1;u[s][k]=u1; }
  int two[MAXN],nt=0; for(int g=2;g<=n;g++) if(j[g]==g-2) two[nt++]=g;
  for(int mask=0;mask<(1<<nt);mask++){ int kept[MAXN+1]={0}; for(int t=0;t<nt;t++) if(mask>>t&1) kept[two[t]]=1;
    for(int g1=1;g1<=n;g1++) for(int s=1;s<=S;s++) for(int mode=0;mode<2;mode++){ cases++;
      // mode 0: bounds for traces (reads from round s-1 traces); mode 1: bounds for true values (true reads)
      int Xm[MAXN+4],Xp[MAXN+4],ym[MAXN+4],yp[MAXN+4]; 
      #define I(g) ((g)+3)
      for(int g=-3;g<g1;g++){Xm[I(g)]=0;Xp[I(g)]=1;ym[I(g)]=0;yp[I(g)]=1;}
      for(int g=g1;g<=n;g++){ int p=j[g]; int lr,ur; if(p==g-1){lr=ur=0;} else if(mode==0){lr=l[s-1][p];ur=u[s-1][p];} else {lr=ur=x[p];}
        ym[I(g)]=lr; yp[I(g)]=ur; int kprev=(g-1>=g1)?kept[g-1]:0;
        if(!kept[g]){ Xm[I(g)]=NOR(Xp[I(g-1)],ur); Xp[I(g)]=NOR(Xm[I(g-1)],lr); }
        else if(!kprev){ Xm[I(g)]=(1-Xp[I(g-2)])*ym[I(g-1)]; Xp[I(g)]=(1-Xm[I(g-2)])*yp[I(g-1)]; }
        else { Xm[I(g)]=Xm[I(g-3)]; Xp[I(g)]=Xp[I(g-3)]; }
        if(mode==0){ if(!(l[s][g]>=Xm[I(g)]&&u[s][g]<=Xp[I(g)])) bad++; }
        else { if(!(Xm[I(g)]<=x[g]&&x[g]<=Xp[I(g)])) badtrue++; }
        if(Xm[I(g)]>Xp[I(g)]) bad++; } } }
}
void rec(int i){ if(i>n){check();return;} for(int p=0;p<i;p++){j[i]=p;rec(i+1);} }
int main(int argc,char**argv){ for(n=atoi(argv[1]);n<=atoi(argv[2]);n++){cnt=bad=badtrue=cases=0;rec(1);printf("n=%d arrays=%ld cases=%ld violations(traces)=%ld violations(true values)=%ld\n",n,cnt,cases,bad,badtrue);} }
