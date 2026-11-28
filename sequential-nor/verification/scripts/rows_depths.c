/* rows_depths.c: the eps-smoothing of the fixed-distance base j_i = max(i-H,0).
   For n = H*L gates, L = Lmax/8, Lmax/4, Lmax/2, Lmax, it prints the largest pruned depth D over the
   upper half of the instance and the number of rounds of the two-valued prefix-Jacobi iteration,
   averaged over the runs.  A simulation; nothing in a proof depends on it.
   Compile with:  cc -O2 -o rows_depths rows_depths.c
   usage: rows_depths eps H Lmax [runs] [seed]        quoted in the paper: rows_depths 0.46 2000 320 3 1 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static unsigned long long s=88172645463325252ULL;
static inline double U(){ s^=s<<13; s^=s>>7; s^=s<<17; return (s>>11)*(1.0/9007199254740992.0);}
int main(int c,char**v){ double eps=atof(v[1]); int H=atoi(v[2]), Lmax=atoi(v[3]); int runs=c>4?atoi(v[4]):3; s+=c>5?atoll(v[5]):0;
 for(int L=Lmax/8; L<=Lmax; L*=2){ double sd=0, sj=0; for(int rr=0; rr<runs; rr++){
  int n=H*L; int *j=malloc(4*(n+1)); char *x=malloc(n+1), *y=malloc(n+1), *z=malloc(n+1); int *D=malloc(4*(n+1));
  x[0]=1; D[0]=0; j[0]=0; int mx=0;
  for(int k=1;k<=n;k++){ int p; if(U()<eps) p=(int)(U()*k); else p = k-H>0?k-H:0; j[k]=p;
    int a=x[k-1], b=x[p]; x[k]=!(a|b);
    if(k==1){D[k]=1;} else if(p==k-1) D[k]=D[k-1];
    else if(!a&&b) D[k]=1+D[p]; else if(a&&b) D[k]= D[k-1]<1+D[p]?D[k-1]:1+D[p];
    else if(!a&&!b) D[k]= D[k-1]>1+D[p]?D[k-1]:1+D[p]; else D[k]=D[k-1];
    if(k>n/2 && D[k]>mx) mx=D[k]; }
  // prefix-Jacobi from all-zero guess: y_k = NOR(y_{k-1}, z_{j_k}) with z lagged
  memset(z,0,n+1); z[0]=1; int it=0; for(;;){ it++; y[0]=1; for(int k=1;k<=n;k++) y[k]=!(y[k-1]|z[j[k]]); if(!memcmp(y,x,n+1)) break; memcpy(z,y,n+1); if(it>100000) break; }
  sd+=mx; sj+=it; free(j);free(x);free(y);free(z);free(D);} 
  printf("eps %.2f H %d layers %d  maxD %.1f  jacobi %.1f\n",eps,H,L,sd/runs,sj/runs);} }
