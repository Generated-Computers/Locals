/* rows_ideal.c: the ideal model of the rows of a fixed-distance base.
   A row has W positions.  A gate reads the same position of the row below with probability 1-eps,
   and otherwise a closed bit that is 1 with probability d, the density of ones of a row (found by
   iterating sixty rows).  Statuses follow the rule f of the paper.  A fraction delta of the
   positions of the first row is open.  It prints the number of open positions and of runs of open
   positions row by row.  A simulation; nothing in a proof depends on it.
   Compile with:  cc -O2 -o rows_ideal rows_ideal.c
   usage: rows_ideal eps W L delta                      quoted in the paper: rows_ideal 0.46 200000 12 0.4 */
#include <stdio.h>
#include <stdlib.h>
static unsigned long long s=88172645463325252ULL;
static inline double U(){ s^=s<<13; s^=s>>7; s^=s<<17; return (s>>11)*(1.0/9007199254740992.0);}
int main(int c,char**v){ double eps=atof(v[1]); int W=atoi(v[2]), L=atoi(v[3]); double delta=atof(v[4]);
 char *x=malloc(W),*o=malloc(W),*y=malloc(W),*p=malloc(W); double d=0.3;
 // warm up density
 for(int a=0;a<W;a++){x[a]=0;o[a]=0;}
 for(int l=0;l<60;l++){ int pv=0; long ones=0; for(int a=0;a<W;a++){ int b = U()<eps ? (U()<d) : x[a]; y[a]=!(pv|b); pv=y[a]; ones+=pv;} d=(double)ones/W; char*t=x;x=y;y=t; }
 for(int a=0;a<W;a++) o[a]= U()<delta;
 long prevn=0,prevr=0;
 for(int l=0;l<L;l++){ long n=0,r=0, len2=0; for(int a=0;a<W;a++){ if(o[a]){n++; if(a==0||!o[a-1]) r++;} }
   printf("layer %2d open %8ld runs %8ld meanlen %.2f ratio_open %.4f ratio_runs %.4f\n",l,n,r,r?(double)n/r:0, prevn?(double)n/prevn:0, prevr?(double)r/prevr:0); prevn=n; prevr=r;
   int pv=0,po=0; long ones=0; for(int a=0;a<W;a++){ int b,ob; if(U()<eps){ b=U()<d; ob=0;} else { b=x[a]; ob=o[a]; }
     int out=!(pv|b), oo; if(!pv&&b) oo=ob; else if(pv&&b) oo=po&&ob; else if(!pv&&!b) oo=po||ob; else oo=po;
     y[a]=out; p[a]=oo; pv=out; po=oo; ones+=out; }
   char*t=x;x=y;y=t; t=o;o=p;p=t; }
 printf("density %.4f\n",d); }
