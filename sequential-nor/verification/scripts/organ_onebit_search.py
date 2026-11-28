# Local search over one-bit designs of one rail: maximize rho_u/lambda at a given noise level.
import sys, os, random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import organ_onebit as onebit
def specrad_fast(M):
    n=len(M); v=[1.0]*n; r=0.0
    for it in range(400):
        w=[sum(M[i][j]*v[j] for j in range(n)) for i in range(n)]
        w=[0.5*(w[i]+r*v[i]) for i in range(n)]; s=sum(w)
        if s==0: return 0.0
        r=2*s-r if False else r
        v=[x/s for x in w]
        if it==0: r=s
        else: r=sum(sum(M[i][j]*v[j] for j in range(n)) for i in range(n))
    return r
onebit.specrad=specrad_fast
def name(d): return " ".join(t if isinstance(t,str) else f"E{t[1]}" for t in d)
def search(T,e,restarts,steps,seed):
    rnd=random.Random(seed); lam=onebit.consts(e)[4]
    types=[('E',c) for c in range(T)]+['U','Z0','Z1']
    best=(0,None)
    for rs in range(restarts):
        d=[rnd.choice(types) for _ in range(T)]
        if all(isinstance(t,str) and t!='U' for t in d): d[0]=('E',0)
        cur=onebit.rho(d,e,iters=3000)[0]/lam
        for st in range(steps):
            nd=list(d); k=rnd.randrange(T); nd[k]=rnd.choice(types)
            if rnd.random()<0.3: k2=rnd.randrange(T); nd[k2]=rnd.choice(types)
            if all(isinstance(t,str) and t!='U' for t in nd): continue
            v=onebit.rho(nd,e,iters=3000)[0]/lam
            if v>=cur: d,cur=nd,v
        nonplain=any(t in('Z0','Z1') for t in d)
        if cur>best[0] and nonplain: best=(cur,list(d))
        print(f"  T={T} eps={e} restart {rs}: {cur:.5f}  {name(d)}",flush=True)
    return best
if __name__=="__main__":
    T=int(sys.argv[1]); e=float(sys.argv[2]); restarts=int(sys.argv[3]); steps=int(sys.argv[4])
    b=search(T,e,restarts,steps,1)
    print("best design with an idle gate:",b[0],name(b[1]) if b[1] else None)
