// canonical.c: exact first moment of the number of CANONICAL base-pointer chains (lowest target, then
// highest gate) for a periodic base with short pointers, in the model where a resampled gate reads an
// independent Bernoulli(psi) bit.  Canonical constraint used (local part only): a kept gate of a window
// may not read the step's own target nor any position of the next window [k_{s+1}, P_s].
// usage: canonical eps psi(<=0: varpi) mode p d0 d1 ...   mode 0 = plain count, 1 = canonical
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
static int D,NB,p,d[8],mode;static double eps,psi;
#define IDX(S,F,L) ((((S)*NB)+(F))*(D+1)+(L))
static int NST;
static double *W[8],*Wn[8],*tmp,*a[8];
int main(int argc,char**argv){
  eps=atof(argv[1]);psi=atof(argv[2]);if(psi<=0)psi=((2*eps+1)-sqrt(4*eps+1))/(2*eps);
  mode=atoi(argv[3]);p=atoi(argv[4]);D=1;for(int i=0;i<p;i++){d[i]=atoi(argv[5+i]);if(d[i]>D)D=d[i];}
  D+=1; NB=1<<D; NST=NB*NB*(D+1);
  for(int f=0;f<p;f++){W[f]=calloc(NST,8);Wn[f]=calloc(NST,8);a[f]=calloc(NST,8);} tmp=calloc(NST,8);
  // stationary law of S per phase (full transitions)
  double *pi[8];for(int f=0;f<p;f++)pi[f]=calloc(NB,8); pi[0][0]=1;
  double *t2=calloc(NB,8);
  for(int it=0;it<3000;it++)for(int f=0;f<p;f++){int g=(f+1)%p;memset(t2,0,8*NB);
    for(int S=0;S<NB;S++){double m=pi[f][S];if(!m)continue;int b0=S&1,dd=d[g],v=dd==1?b0:((S>>(dd-1))&1);
      t2[((S<<1)|!(b0|v))&(NB-1)]+=m*(1-eps); t2[((S<<1)|0)&(NB-1)]+=m*eps*psi; t2[((S<<1)|!b0)&(NB-1)]+=m*eps*(1-psi);}
    memcpy(pi[g],t2,8*NB);}
  // W_0: chains of length 0 from q: count 1, window extent L = 0 (no window), F irrelevant (0)
  for(int f=0;f<p;f++)for(int S=0;S<NB;S++)W[f][IDX(S,0,0)]=pi[f][S];
  double mu=0;
  double *v1=calloc(NB*(D+1),8),*v2=calloc(NB*(D+1),8);
  for(int t=1;t<=80;t++){
    // chain-gate source terms a[f]
    for(int f=0;f<p;f++){memset(a[f],0,8*NST);int dd=d[f];if(dd<2)continue;
      int src=((f-dd)%p+p)%p;
      // V'(S,e) = sum_F W[src][S,F,L=e]
      memset(v1,0,8*NB*(D+1));
      for(int S=0;S<NB;S++)for(int F=0;F<NB;F++)for(int L=0;L<=D;L++)v1[S*(D+1)+L]+=W[src][IDX(S,F,L)];
      // push forward through free gates src+1 .. f-1
      for(int k=dd-1;k>=1;k--){int g=((f-k)%p+p)%p;memset(v2,0,8*NB*(D+1));
        for(int S=0;S<NB;S++)for(int e=0;e<=D;e++){double m=v1[S*(D+1)+e];if(!m)continue;int b0=S&1,de=d[g],v=de==1?b0:((S>>(de-1))&1);
          v2[(((S<<1)|!(b0|v))&(NB-1))*(D+1)+e]+=m*(1-eps); v2[(((S<<1)|0)&(NB-1))*(D+1)+e]+=m*eps*psi; v2[(((S<<1)|!b0)&(NB-1))*(D+1)+e]+=m*eps*(1-psi);}
        memcpy(v1,v2,8*NB*(D+1));}
      // gate f kept: reads offset dd from f (bit dd-1 of S at f-1)
      for(int S=0;S<NB;S++)for(int e=0;e<=D;e++){double m=v1[S*(D+1)+e];if(!m)continue;int b0=S&1,v=(S>>(dd-1))&1;
        int Sn=((S<<1)|!(b0|v))&(NB-1);
        // forbidden mask relative to f: positions target-e..target = offsets dd..dd+e
        int F=0; if(mode==1) for(int o=dd;o<=dd+e&&o<D;o++)F|=1<<o;
        a[f][IDX(Sn,F,0)]+=m*(1-eps);}
    }
    // window fixed point: Wn[f] = a[f] + window(Wn[f-1])
    for(int f=0;f<p;f++)memcpy(Wn[f],a[f],8*NST);
    for(int it=0;it<4000;it++){double diff=0;
      for(int f=0;f<p;f++){int g=(f-1+p)%p;int dd=d[f];memset(tmp,0,8*NST);
        for(int S=0;S<NB;S++)for(int F=0;F<NB;F++)for(int L=0;L<=D;L++){double m=Wn[g][IDX(S,F,L)];if(!m)continue;
          int b0=S&1,v=dd==1?b0:((S>>(dd-1))&1);int Sn=((S<<1)|!b0)&(NB-1);int Fn=(F<<1)&(NB-1);int Ln=L<D?L+1:D;
          // kept non-reset, allowed
          int reset=(dd>=2)&&v; int forb=(dd>=2)&&((F>>(dd-1))&1);
          if(!reset&&!forb)tmp[IDX(Sn,Fn,Ln)]+=m*(1-eps);
          tmp[IDX(Sn,Fn,Ln)]+=m*eps*(1-psi);}
        for(int i=0;i<NST;i++){double nv=a[f][i]+tmp[i];diff+=fabs(nv-Wn[f][i]);Wn[f][i]=nv;}}
      if(diff<1e-12)break;}
    double sn=0,so=0;for(int f=0;f<p;f++)for(int i=0;i<NST;i++){sn+=Wn[f][i];so+=W[f][i];}
    mu=sn/so;for(int f=0;f<p;f++)for(int i=0;i<NST;i++)W[f][i]=Wn[f][i]/sn;
  }
  printf("%.6f\n",mu);return 0;}
