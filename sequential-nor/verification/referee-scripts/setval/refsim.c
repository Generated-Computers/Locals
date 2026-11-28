/* Referee's own simulator for Remark rem:setval (written before reading the author's code).
   usage: refsim L m eps seed back R [prefrac_num prefrac_den]
   Mixed base: preamble of zero gates (j_i=0), then layer 0 of m zero gates, then layers 1..L of m
   positions: even offset -> j_i=i-back, odd offset -> uniform even-offset position of the layer
   below. Preamble length P = L*m*num/den (default 1/1: half of the positions, not counting layer 0).
   Then each gate is resampled with probability eps to a uniform position below it.
   Prints: maxD, maxD_R, maxD'_R, rounds of the set-valued iteration with reach R, applications
   of G_R; checks soundness, monotonicity, prop (b), (c) along the way. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static uint64_t st;
static uint64_t rnd(void){ uint64_t z=(st+=0x9E3779B97F4A7C15ULL); z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;
  z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
static double unif(void){ return (rnd()>>11)*(1.0/9007199254740992.0); }
static int64_t below(int64_t k){ /* uniform on 0..k-1 */ uint64_t lim=UINT64_MAX-(UINT64_MAX%(uint64_t)k); uint64_t r;
  do r=rnd(); while(r>=lim); return (int64_t)(r%(uint64_t)k); }

int main(int argc,char**argv){
  if(argc<7){fprintf(stderr,"usage\n");return 1;}
  int L=atoi(argv[1]), m=atoi(argv[2]); double eps=atof(argv[3]); uint64_t seed=strtoull(argv[4],0,10);
  int back=atoi(argv[5]), R=atoi(argv[6]);
  int num=argc>8?atoi(argv[7]):1, den=argc>8?atoi(argv[8]):1;
  if(R<1||R>6){fprintf(stderr,"R in 1..6\n");return 1;}
  st=seed*0x2545F4914F6CDD1DULL+12345;
  int64_t P=(int64_t)L*m*num/den, n=P+(int64_t)(L+1)*m;
  int32_t *j=malloc((n+1)*sizeof(int32_t));
  for(int64_t i=1;i<=P;i++) j[i]=0;
  for(int t=0;t<=L;t++){ int64_t s0=P+1+(int64_t)t*m;
    for(int o=0;o<m;o++){ int64_t i=s0+o;
      if(t==0) j[i]=0;
      else if(o%2==0) j[i]=(int32_t)(i-back);
      else j[i]=(int32_t)(s0-m+2*below(m/2)); } }
  for(int64_t i=1;i<=n;i++) if(unif()<eps) j[i]=(int32_t)below(i);
  /* trace */
  uint8_t *x=malloc(n+1); x[0]=1; for(int64_t i=1;i<=n;i++) x[i]=!(x[i-1]|x[j[i]]);
  /* depths: D, D_R, D'_R by the four cases */
  int32_t *D=malloc((n+1)*4),*DR=malloc((n+1)*4),*DP=malloc((n+1)*4);
  D[0]=DR[0]=DP[0]=0; D[1]=DR[1]=DP[1]=1; int mD=1,mDR=1,mDP=1;
  for(int64_t k=2;k<=n;k++){ int p=j[k]; int notg=(p==k-1), reset=(!notg&&x[p]), rel=!x[k-1];
    int sh=(p>=k-R); int a=1+D[p], b=(sh?0:1)+DR[p], c=sh?(DP[p]>1?DP[p]:1):1+DP[p];
    if(reset&&rel){D[k]=a;DR[k]=b;DP[k]=c;}
    else if(reset){D[k]=D[k-1]<a?D[k-1]:a; DR[k]=DR[k-1]<b?DR[k-1]:b; DP[k]=DP[k-1]<c?DP[k-1]:c;}
    else if(rel&&!notg){D[k]=D[k-1]>a?D[k-1]:a; DR[k]=DR[k-1]>b?DR[k-1]:b; DP[k]=DP[k-1]>c?DP[k-1]:c;}
    else {D[k]=D[k-1];DR[k]=DR[k-1];DP[k]=DP[k-1];}
    if(D[k]>mD)mD=D[k]; if(DR[k]>mDR)mDR=DR[k]; if(DP[k]>mDP)mDP=DP[k]; }
  /* set-valued iteration, literally: S_i as a bitmask over windows; window w: bit 0 = last coordinate,
     bit d-1 = z_{i-d} when the window stands at position i-1 ... i.e. coordinate z_{(i-1)-(d-1)} */
  int W=1<<R; uint64_t wmask=W-1;
  uint8_t *K=calloc(n+1,1),*Kn=malloc(n+1),*v=calloc(n+1,1),*vn=malloc(n+1);
  uint8_t *y=calloc(n+1,1),*z=malloc(n+1); y[0]=1; /* table-Jacobi iterate y^(s) alongside */
  int32_t *first=malloc((n+1)*4); for(int64_t i=0;i<=n;i++) first[i]=-1; first[0]=0;
  K[0]=1; v[0]=1; int64_t nk=1; int rounds=0, bad=0, apps=-1;
  int64_t undet_complement=0;
  while(nk<n+1){
    uint64_t S=1ULL<<1; /* sigma_0=(0,...,0,1) -> integer 1 */
    Kn[0]=1; vn[0]=1; int64_t c=1;
    for(int64_t i=1;i<=n;i++){ int p=j[i]; uint64_t T=0; int d=(int)(i-p);
      for(int w=0;w<W;w++) if(S>>w&1){ int a=w&1;
        if(d<=R){ int b=(w>>(d-1))&1; T|=1ULL<<(((w<<1)&wmask)|(!(a|b))); }
        else if(K[p]){ int b=v[p]; T|=1ULL<<(((w<<1)&wmask)|(!(a|b))); }
        else { T|=1ULL<<(((w<<1)&wmask)|(!a)); T|=1ULL<<((w<<1)&wmask); } }
      S=T; int has0=0,has1=0; for(int w=0;w<W;w++) if(S>>w&1){ if(w&1)has1=1; else has0=1; }
      if(has0&&has1){Kn[i]=0;vn[i]=0;} else {Kn[i]=1;vn[i]=has1;c++; if(first[i]<0)first[i]=rounds+1;
        if(vn[i]!=x[i])bad|=1;}
      if(K[i]&&!Kn[i])bad|=2; }
    /* table-Jacobi step */
    z[0]=1; for(int64_t i=1;i<=n;i++){ int p=j[i]; int r=(i-p<=R)?z[p]:y[p]; z[i]=!(z[i-1]|r); }
    if(apps<0 && memcmp(y,z,n+1)==0) apps=rounds+1;
    memcpy(y,z,n+1);
    rounds++;
    for(int64_t i=0;i<=n;i++) if(Kn[i]&&y[i]!=x[i]) bad|=4; /* (c): y^(s)=x on K^(s) */
    memcpy(K,Kn,n+1); memcpy(v,vn,n+1); nk=c;
    if(rounds>100000){bad|=64;break;}
  }
  /* finish table-Jacobi */
  { int s=rounds; while(apps<0){ z[0]=1; for(int64_t i=1;i<=n;i++){ int p=j[i]; int r=(i-p<=R)?z[p]:y[p]; z[i]=!(z[i-1]|r);}
      s++; if(memcmp(y,z,n+1)==0) apps=s; memcpy(y,z,n+1); } }
  if(memcmp(y,x,n+1)!=0) bad|=8;
  /* (b): D_R(i)<=s => i in K^(s+1), i.e. first[i]<=D_R(i)+1 ; extras: first<=D, first<=D'_R */
  int64_t eqDP=0; for(int64_t i=0;i<=n;i++){ if(first[i]>DR[i]+1)bad|=16; if(first[i]>D[i]||first[i]>DP[i])bad|=32;
    if(first[i]==DP[i])eqDP++; }
  if(apps>rounds+1)bad|=128;
  (void)undet_complement;
  printf("L=%d m=%d eps=%.2f seed=%llu back=%d R=%d n=%lld  maxD=%d maxD_R=%d maxD'_R=%d  setrounds=%d  Gapps=%d  frac(first=D'_R)=%.4f bad=%d\n",
    L,m,eps,(unsigned long long)seed,back,R,(long long)n,mD,mDR,mDP,rounds,apps,(double)eqDP/(n+1),bad);
  return 0;
}
