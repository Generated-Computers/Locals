# rows_word.py: the same ratio as rows_periodic.c for given words of symbols (0, 1 closed; o open 0;
# I open 1).  In a word, A stands for thirty times "10", B for thirty times "01", Z for forty zeros.
# usage: python3 rows_word.py eps d word...      quoted in the paper: python3 rows_word.py 0.455 0.3333333333 o1oA
import sys
eps=float(sys.argv[1]); d=float(sys.argv[2])
def gate(pv,po,b,ob):
    out=0 if (pv or b) else 1
    if not pv and b: oo=ob
    elif pv and b: oo=po and ob
    elif not pv and not b: oo=po or ob
    else: oo=po
    return out,int(bool(oo))
M={}
for sym in "01oI":
    b=1 if sym in "1I" else 0; ob=1 if sym in "oI" else 0; T={}
    for s in range(16):
        x1,o1,x2,o2=s&1,(s>>1)&1,(s>>2)&1,(s>>3)&1; row={}
        for p1,r1,q1 in [(1-eps,b,ob),(eps*d,1,0),(eps*(1-d),0,0)]:
            nx1,no1=gate(x1,o1,r1,q1)
            for p2,r2,q2 in [(1-eps,nx1,no1),(eps*d,1,0),(eps*(1-d),0,0)]:
                nx2,no2=gate(x2,o2,r2,q2); u=nx1|no1<<1|nx2<<2|no2<<3; row[u]=row.get(u,0)+p1*p2
        T[s]=row
    M[sym]=T
def ratio(w,reps=60):
    p=[0.0]*16; p[0]=1.0
    for rep in range(reps):
        n1=n2=0.0
        for ch in w:
            q=[0.0]*16
            for s in range(16):
                if p[s]:
                    for u,pr in M[ch][s].items(): q[u]+=p[s]*pr
            p=q; n1+=sum(p[s] for s in range(16) if s&2); n2+=sum(p[s] for s in range(16) if s&8)
    return n2/n1,n1,n2
for w in sys.argv[3:]:
    w2=w.replace("A","10"*30).replace("B","01"*30).replace("Z","0"*40)
    r,n1,n2=ratio(w2); print(w,"ratio %.5f  N1 %.4f N2 %.4f"%(r,n1,n2))
