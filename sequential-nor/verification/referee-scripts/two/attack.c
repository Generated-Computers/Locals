/* Referee attack on Theorem thm:two: sup over periodic chains of N_h / f(alpha_h),
   computed from the joint law of two copies (16 states), not from the a,b,c recursion.
   gate types: 0 = mixed with y (target is 0 w.p. y), s = fhat(y) (worst case); 1 = L1; 2 = L2.
   usage: attack seed ntrials Tmax eps_mode(0: eps=0.455, 1: random in [0.455,1)) */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define TM 40
static double eps,vp,A,B,c1;
static double f(double x){ return pow(x,2.15)*pow(1-x,1.65); }
static double S1,S2;
static double fhat(double y){ if(y<=23.0/56) return S1*y; if(y>=43.0/56) return S2*(1-y); return f(y); }
static int NORf(int a,int b){ return (!a&&!b); }
typedef struct{int T; int ty[TM]; double y[TM];} design;
static double rnd(void){ return (double)random()/2147483648.0; }
/* returns max_h ratio; *wh = argmax phase; alpha out */
static double eval(const design*D,int*wh,double*alout,double*Nout){
  int T=D->T; double law[TM][4]; double d[4]={0.25,0.25,0.25,0.25};
  /* single chain: state s = 2*u+v, u=x_{h-1}, v=x_h */
  for(int it=0;it<4000;it++){
    double diff=0;
    for(int h=0;h<T;h++){
      double n[4]={0,0,0,0};
      for(int s=0;s<4;s++){ if(d[s]==0) continue; int u=s>>1,v=s&1;
        n[2*v+NORf(v,1)]+=d[s]*A; n[2*v+NORf(v,0)]+=d[s]*B;
        if(D->ty[h]==0){ n[2*v+NORf(v,0)]+=d[s]*c1*D->y[h]; n[2*v+NORf(v,1)]+=d[s]*c1*(1-D->y[h]); }
        else if(D->ty[h]==1) n[2*v+NORf(v,v)]+=d[s]*c1;
        else n[2*v+NORf(v,u)]+=d[s]*c1; }
      for(int s=0;s<4;s++){ diff+=fabs(n[s]-law[h][s]); law[h][s]=n[s]; d[s]=n[s]; }
    }
    if(it>3&&diff<1e-16) break;
  }
  double m[16]; memset(m,0,sizeof m); double N[TM]; for(int h=0;h<T;h++) N[h]=-1;
  for(int it=0;it<6000;it++){
    double diff=0;
    for(int h=0;h<T;h++){
      double n[16]; memset(n,0,sizeof n);
      for(int s=0;s<16;s++){ if(m[s]==0) continue; int u=(s>>3)&1,v=(s>>2)&1,u2=(s>>1)&1,v2=s&1;
        int rk,rk2; double pk[4]; int r1[4],r2[4]; int nk=0;
        pk[nk]=A; r1[nk]=1; r2[nk]=1; nk++; pk[nk]=B; r1[nk]=0; r2[nk]=0; nk++;
        if(D->ty[h]==0){ pk[nk]=c1*D->y[h]; r1[nk]=0; r2[nk]=0; nk++; pk[nk]=c1*(1-D->y[h]); r1[nk]=1; r2[nk]=1; nk++; (void)rk;(void)rk2;
          for(int k=0;k<nk;k++){ int a=NORf(v,r1[k]),b=NORf(v2,r2[k]); if(v==v2&&a==b) continue; n[8*v+4*a+2*v2+b]+=m[s]*pk[k]; } continue; }
        if(D->ty[h]==1){ pk[nk]=c1; r1[nk]=v; r2[nk]=v2; nk++; } else { pk[nk]=c1; r1[nk]=u; r2[nk]=u2; nk++; }
        for(int k=0;k<nk;k++){ int a=NORf(v,r1[k]),b=NORf(v2,r2[k]); if(v==v2&&a==b) continue; n[8*v+4*a+2*v2+b]+=m[s]*pk[k]; }
      }
      if(D->ty[h]==0){ /* injection: source gate h, kept, target 0 in x and 1 in x' */
        const double*pl=law[(h+T-1)%T]; double sw=fhat(D->y[h]);
        for(int s=0;s<4;s++){ int v=s&1; int a=NORf(v,0),b=NORf(v,1); if(a==b) continue; n[8*v+4*a+2*v+b]+=sw*c1*pl[s]; } }
      double Nh=0; for(int s=0;s<16;s++){ m[s]=n[s]; if(((s>>2)&1)!=(s&1)) Nh+=n[s]; }
      diff+=fabs(Nh-N[h]); N[h]=Nh;
    }
    if(it>3&&diff<1e-15) break;
  }
  double best=-1; for(int h=0;h<T;h++){ double al=law[h][0]+law[h][2]; double r=N[h]/f(al); if(alout) alout[h]=al; if(Nout) Nout[h]=N[h]; if(r>best){best=r; if(wh)*wh=h;} }
  return best;
}
static void seteps(double e){ eps=e; vp=((2*e+1)-sqrt(4*e+1))/(2*e); A=e*vp; B=e*(1-vp); c1=1-e; }
static void pr(const design*D){ for(int h=0;h<D->T;h++){ if(D->ty[h]==0) printf("M(%.4f) ",D->y[h]); else printf("L%d ",D->ty[h]); } }
int main(int argc,char**argv){
  long seed=atol(argv[1]); int ntr=atoi(argv[2]); int Tmax=atoi(argv[3]); int emode=atoi(argv[4]);
  srandom(seed); S1=f(23.0/56)/(23.0/56); S2=f(43.0/56)/(13.0/56);
  double gbest=-1,gbestL2=-1,gbestAtL2=-1; design bestD,bestDL2; double be=0,beL2=0;
  for(int tr=0;tr<ntr;tr++){
    seteps(emode==0?0.455:(emode==1?0.455+0.545*rnd()*rnd():0.455+0.02*rnd()));
    design D; D.T=1+random()%Tmax; int mode=random()%4; int hasM=0;
    for(int h=0;h<D.T;h++){ double u=rnd(); D.ty[h]= mode==0?(u<0.34?0:(u<0.5?1:2)) : mode==1?(u<0.2?0:(u<0.25?1:2)) : mode==2?(u<0.6?0:(u<0.8?1:2)):(u<0.5?0:2);
      D.y[h]= rnd()<0.15?(rnd()<0.5?0:1):rnd(); if(D.ty[h]==0) hasM=1; }
    if(!hasM){ D.ty[random()%D.T]=0; }
    int wh; double cur=eval(&D,&wh,0,0);
    /* local search on y and types */
    double step=0.25;
    for(int it=0;it<300;it++){
      design E=D; int h=random()%D.T; int mv=random()%10;
      if(mv<7&&E.ty[h]==0){ E.y[h]+=step*(2*rnd()-1); if(E.y[h]<0)E.y[h]=0; if(E.y[h]>1)E.y[h]=1; }
      else if(mv<9){ E.ty[h]=random()%3; }
      else { E.y[h]=rnd()<0.5?0:1; }
      int ok=0; for(int i=0;i<E.T;i++) if(E.ty[i]==0) ok=1; if(!ok) continue;
      int w2; double v=eval(&E,&w2,0,0);
      if(v>cur){ cur=v; D=E; wh=w2; } else if(it%40==39) step*=0.6;
    }
    int hasL2=0; for(int h=0;h<D.T;h++) if(D.ty[h]==2) hasL2=1;
    if(cur>gbest){ gbest=cur; bestD=D; be=eps; }
    if(hasL2&&cur>gbestL2){ gbestL2=cur; bestDL2=D; beL2=eps; }
    if(hasL2){ double al[TM],N[TM]; eval(&D,0,al,N); for(int h=0;h<D.T;h++) if(D.ty[h]==2){ double r=N[h]/f(al[h]); if(r>gbestAtL2) gbestAtL2=r; } }
  }
  printf("max ratio overall %.6f at eps %.5f: ",gbest,be); pr(&bestD); printf("\n");
  printf("max ratio among designs with an L2 gate %.6f at eps %.5f: ",gbestL2,beL2); pr(&bestDL2); printf("\n");
  printf("max ratio at an L2 gate %.6f\n",gbestAtL2);
  return 0;
}
