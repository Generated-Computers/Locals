// setval_table.c: the sound "set-valued" table iteration on the mixed base (even offsets j=i-H, odd offsets a
// random even offset of the layer below; zero gates below). State after gate i: the set of possible
// windows (z_{i-R+1..i}) when every long read of a position not yet known may be 0 or 1.
// A position becomes known when all windows of the set agree on it. Rounds until all are known,
// and the least s with y^(s) = trace for the two-valued table-Jacobi iteration of the same reach
// (one less than its number of applications of G_R in the definition) (the remark after the
// proposition on the set-valued table iteration). R <= 12 or so; the sets are kept as byte arrays.
// The paper: ./setval_table 0.50 600 0.5 L 2 1 for L = 40, 80, 160, 320; the same at 0.46;
//   (seeds 1, 2, 3 at 0.50); ./setval_table EPS 600 0.5 L R 2 6 for R = 5, 6, L = 40, 160,
//   EPS = 0.46, 0.50; ./setval_table EPS 600 0.5 L 2 1 for EPS = 0.10, 0.20, 0.30, 0.40, L = 40, 160, 640.
// usage: setval_table eps m af L R seed [H]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static uint64_t s[2];
static inline uint64_t rnd(void){uint64_t a=s[0],b=s[1];s[0]=b;a^=a<<23;s[1]=a^b^(a>>17)^(b>>26);return s[1]+b;}
static inline double uni(void){return (rnd()>>11)*(1.0/9007199254740992.0);}
static inline uint32_t below(uint32_t k){return (uint32_t)(uni()*k);}
int main(int argc,char**argv){
  double eps=atof(argv[1]);uint32_t m=atoi(argv[2]);double af=atof(argv[3]);uint32_t L=atoi(argv[4]);int R=atoi(argv[5]);uint64_t sd=strtoull(argv[6],0,10);uint32_t H=argc>7?atoi(argv[7]):2;
  m-=m%2; uint32_t top=(L+1)*m,n0=(uint32_t)(top*(1.0-af)/af),n=n0+top;
  s[0]=sd*0x9E3779B97F4A7C15ULL+1;s[1]=sd^0xD1B54A32D192ED03ULL;for(int i=0;i<20;i++)rnd();
  uint32_t*j=malloc(4*(size_t)(n+1));uint8_t*x=malloc(n+1),*kn=malloc(n+1),*kn2=malloc(n+1),*y=malloc(n+1),*z=malloc(n+1);
  j[0]=0;
  for(uint32_t i=1;i<=n;i++){uint32_t b=0; if(i>n0+m){uint32_t off=i-n0-1,t=off/m,o=off%m,base=n0+1+(t-1)*m; b=(o%2==0)?i-H:base+2*below(m/2);} j[i]=b;}
  for(uint32_t i=1;i<=n;i++) if(uni()<eps) j[i]=below(i);
  x[0]=1; for(uint32_t k=1;k<=n;k++) x[k]=!(x[k-1]|x[j[k]]);
  int NS=1<<R; uint32_t msk=NS-1;
  memset(kn,0,n+1); kn[0]=1; int rounds=0; uint32_t unknown=n;
  while(unknown>0 && rounds<5000){ rounds++;
    // window state: bit t of sigma = z_{i-t} (bit 0 = most recent)
    uint8_t *cur=calloc(NS,1),*nxt=calloc(NS,1); cur[1]=1; // z_0=1, earlier positions 0
    kn2[0]=1; unknown=0;
    for(uint32_t i=1;i<=n;i++){ memset(nxt,0,NS); uint32_t p=j[i], d=i-p; int any0=0,any1=0;
      for(int sg=0;sg<NS;sg++) if(cur[sg]){ int a=sg&1;
        if(d<=(uint32_t)R){ int r=(sg>>(d-1))&1; int v=!(a|r); nxt[((sg<<1)|v)&msk]=1; if(v)any1=1;else any0=1; }
        else if(kn[p]){ int v=!(a|x[p]); nxt[((sg<<1)|v)&msk]=1; if(v)any1=1;else any0=1; }
        else { int v0=!(a|0), v1=0; nxt[((sg<<1)|v0)&msk]=1; nxt[((sg<<1)|v1)&msk]=1; any0=1; if(v0)any1=1; } }
      kn2[i]=!(any0&&any1); if(!kn2[i]) unknown++;
      uint8_t*w=cur;cur=nxt;nxt=w; }
    free(cur);free(nxt); uint8_t*w=kn;kn=kn2;kn2=w; }
  // two-valued table-Jacobi with the same reach
  memset(y,0,n+1);y[0]=1; int r2=0;
  for(;r2<5000;r2++){ if(memcmp(y,x,n+1)==0) break; z[0]=1;
    for(uint32_t k=1;k<=n;k++){uint32_t t=j[k];uint8_t v=(k-t<=(uint32_t)R)?z[t]:y[t];z[k]=!(z[k-1]|v);} uint8_t*w=y;y=z;z=w;}
  printf("eps=%.2f H=%u R=%d L=%u: set-valued rounds=%d (unknown left %u), table-Jacobi reaches the trace at s=%d\n",eps,H,R,L,rounds,unknown,r2);
  return 0;}
