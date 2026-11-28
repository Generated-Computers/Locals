/* cert_cmp2.c: referee harness. Compile twice: -DWHICH=2 includes rows_check.c, -DWHICH=3 includes
   rows3_check.c (with -DNR=2). Prints the transition table, TU/TD on random vectors (hex) and the verdict
   of check() on random small vectors, for several boxes; the two outputs must be identical. */
#define main main_orig
#if WHICH==2
#include "../../scripts/rows_check.c"
#define NSS 16
#define NXT(y,s,i,k) nx[y][s][i][k]
#else
#include "../../scripts/rows3_check.c"
#define NSS NS
#define NXT(y,s,i,k) nx[((y)*NS+(s))*P3[NR]+(i)+3*(k)]
#endif
#undef main
static unsigned long long X=88172645463325252ULL; static double rnd(void){ X^=X<<13; X^=X>>7; X^=X<<17; return (double)(X>>11)/9007199254740992.0; }
int main(int argc,char**argv){ init(); theta=0.999; K=8;
  for(int a=1;a<argc;a++){ FILE*f=fopen(argv[a],"r"); if(!f) return 1; double c[4][16]; int h;       /* dumps of two-row boxes */
    if(fscanf(f," BOX %la %la %la %la",&EA,&EB,&DA,&DB)!=4) return 1;
    for(int i=0;i<4;i++){ if(fscanf(f," C %d",&h)!=1) return 1; for(int s=0;s<16;s++) if(fscanf(f,"%la",&c[h][s])!=1) return 1; }
    fclose(f); long n0=napp; int v=check(c); printf("dump %d: check %d %ld\n",a,v,napp-n0);
    c[0][10]+=0.05; n0=napp; v=check(c); printf("dump %d perturbed: check %d %ld\n",a,v,napp-n0); }
  for(int y=0;y<4;y++)for(int s=0;s<16;s++){ for(int i=0;i<3;i++)for(int k=0;k<3;k++) printf("%d ",NXT(y,s,i,k)); printf("\n"); }
  double bx[4][4]={{0.455,0.4555,0.33,0.332},{0.6,0.61,0.3,0.31},{0.99,1.0,0.2,0.5},{0.455,0.455,0.34,0.34}};
  for(int b=0;b<4;b++){ EA=bx[b][0];EB=bx[b][1];DA=bx[b][2];DB=bx[b][3];
    for(int r=0;r<200;r++){ double g[16],o[16]; for(int s=0;s<16;s++){ g[s]=(rnd()<0.2)?0:(rnd()*2-1)*(r%3==0?1e-3:3); }
      for(int y=0;y<4;y++){ TU(y,g,o); for(int s=0;s<16;s++) printf("%a ",o[s]); TD(y,g,o); for(int s=0;s<16;s++) printf("%a ",o[s]); printf("\n"); } }
    for(int r=0;r<300;r++){ double c[4][16]; double sc=r<100?0:(r<200?1e-3:0.3); for(int h=0;h<4;h++)for(int s=0;s<16;s++) c[h][s]=(s&10)?sc*rnd():0;
      if(r==299) c[2][0]=1e-9; if(r==298) c[1][7]=4.0000001;
      long n0=napp; int v=check(c); printf("check %d %ld\n",v,napp-n0); } }
  return 0; }
