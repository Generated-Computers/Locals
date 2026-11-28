/* compare the enclosures of scan_check.c (point intervals) with the referee's 16-state computation */
#define main scan_main
#define A sA
#define B sB
#define C sC
#define Q sQ
#define K sK
#define theta stheta
#define run srun
#define nb snb
#define d sd_
#include "scan_check.c"
#undef main
#undef A
#undef B
#undef C
#undef Q
#undef K
#undef theta
#undef run
#undef nb
#undef d
#define main g2_main
#define G2 myG2
#include "g2_exact.c"
#undef main
int main(int argc,char**argv){
  double e=atof(argv[1]), dd=atof(argv[2]); sK=120;
  if(!setup(e,e)){printf("setup failed\n");return 1;}
  iv D=mk(dd,dd); sQ=iadd(sB,imul(sC,isub(ONE,D))); AL=idiv(ONE,iadd(ONE,sQ)); srun();
  eps=e; vp=((2*eps+1)-sqrt(4*eps+1))/(2*eps); A=eps*vp; B=eps*(1-vp); c=1-eps; setd(dd);
  double maxerr=0, maxw=0;
  for(int k=0;k<=120;k++){
    double S,Qq; pass(k,&S,&Qq);
    double v[2][4];
    for(int st=0;st<2;st++){ double p=st; double o,on; clean_block(k,p,&o,&on);
      /* cost and X by my method */
      double acc[16]={0};
      { double s[16]={0}; s[0]=alpha; s[12]=1-alpha; win_adv(s,1);
        for(int j=-1;j>=-JMAX;j--){ for(int q=0;q<16;q++) acc[q]+=(((q>>3)&1)!=((q>>2)&1))?s[q]:0; win_adv(s,0); for(int q=0;q<16;q++) if(((q>>3)&1)==((q>>2)&1)) s[q]=0; } }
      double cost_=0,X_=0;
      { double s[16]={0}; for(int q=0;q<16;q++) if(acc[q]>0){ s[(q&12)+0]+=acc[q]*p; s[(q&12)+3]+=acc[q]*(1-p);} 
        for(int t=0;t<=k;t++){ win_adv(s,0); e_read(s); cost_+=ediff(s);} X_+=ediff(s); }
      for(int j=0;j<=k;j++){ double s[16]={0};
        for(int w=0;w<2;w++) for(int ee=0;ee<2;ee++) s[w*12+ee*3]=(w?1-alpha:alpha)*(ee?1-p:p);
        for(int t=0;t<=k;t++){ win_adv(s,t==j); e_read(s); cost_+=ediff(s);} X_+=ediff(s); }
      v[st][0]=cost_; v[st][1]=X_; v[st][2]=o; v[st][3]=on;
      iv a[4]={cost[st][k],Xk[st][k],outk[st][k],ones[st][k]};
      for(int i=0;i<4;i++){ double mid=0.5*(a[i].lo+a[i].hi), err=fabs(mid-v[st][i]); if(err>maxerr) maxerr=err; if(a[i].hi-a[i].lo>maxw) maxw=a[i].hi-a[i].lo;
        if(k<3||k==120) printf("k %d p %d q%d author [%.12f,%.12f] mine %.12f\n",k,st,i,a[i].lo,a[i].hi,v[st][i]); } }
    double e1=fabs(0.5*(Sk[k].lo+Sk[k].hi)-S), e2=fabs(0.5*(Qk[k].lo+Qk[k].hi)-Qq); if(e1>maxerr)maxerr=e1; if(e2>maxerr)maxerr=e2;
  }
  printf("eps %g d %g: max |author mid - mine| = %.3e, max width %.3e\n",e,dd,maxerr,maxw); return 0; }
