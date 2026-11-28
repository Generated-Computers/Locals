# Designs with independent reads and mixed types: R rails of period T; gate (r,f) lists position 0
# with probability mu[0], position 1 with mu[1], class c with mu[2+c].  rho_u = spectral radius of
# Mbar; hill climbing on rho_u/lambda at a noise level.   usage: mixed_search.py eps R T restarts steps seed
import math, random, sys
def consts(e):
    vp=((2*e+1)-math.sqrt(4*e+1))/(2*e); A=e*vp; B=e*(1-vp)
    q=(-(1-B)+math.sqrt((1-B)**2+4*(1-A)))/2
    return vp,A,B,q,(1-e)/(1-q*q)
def rho(des,e,R,T):
    vp,A,B,qs,lam=consts(e); c=1-e; NC=R*T
    al=[0.6]*NC
    for it in range(5000):
        q=[B+c*(des[g][1]+sum(des[g][2+k]*al[k] for k in range(NC))) for g in range(NC)]
        new=al[:]
        for r in range(R):
            a=al[r*T+T-1]
            for rep in range(3):
                for f in range(T):
                    a=1-q[r*T+f]*a; new[r*T+f]=a
        df=max(abs(new[i]-al[i]) for i in range(NC)); al=new
        if df<1e-13: break
    else:
        return None,lam,al
    q=[B+c*(des[g][1]+sum(des[g][2+k]*al[k] for k in range(NC))) for g in range(NC)]
    M=[[0.0]*NC for _ in range(NC)]
    for r in range(R):
        Q=1.0
        for f in range(T): Q*=q[r*T+f]
        for f in range(T):
            g=r*T+f; w=c*al[r*T+(f-1)%T]/(1-Q); pr=1.0
            for j in range(T):
                k=r*T+(f+j)%T
                for cc in range(NC):
                    if des[g][2+cc]: M[k][cc]+=pr*w*des[g][2+cc]
                pr*=q[r*T+(f+j+1)%T]
    v=[1.0]*NC; val=0
    for it in range(600):
        w=[sum(M[i][j]*v[j] for j in range(NC)) for i in range(NC)]
        w=[0.5*(w[i]+val*v[i]) for i in range(NC)]; s=sum(w)
        if s==0: return 0.0,lam,al
        v=[x/s for x in w]
        val=sum(sum(M[i][j]*v[j] for j in range(NC)) for i in range(NC))
    return val,lam,al
def normalize(mu):
    mu=[max(0.0,x) for x in mu]; s=sum(mu); return [x/s for x in mu]
def main():
    e=float(sys.argv[1]); R=int(sys.argv[2]); T=int(sys.argv[3]); restarts=int(sys.argv[4]); steps=int(sys.argv[5]); rnd=random.Random(int(sys.argv[6]))
    NC=R*T; best=0
    for rs in range(restarts):
        if rs%2==0: des=[normalize([rnd.random()**3*0.3,rnd.random()**3*0.3]+[rnd.random()**2 for _ in range(NC)]) for g in range(NC)]
        else: des=[normalize([0.0,0.0]+[rnd.random() for _ in range(NC)]) for g in range(NC)]
        cur,lam,al=rho(des,e,R,T); cur=(cur or 0)/lam; sc=0.3
        for st in range(steps):
            nd=[m[:] for m in des]; g=rnd.randrange(NC); i=rnd.randrange(NC+2); nd[g][i]+=sc*(rnd.random()-0.3); nd[g]=normalize(nd[g])
            if sum(nd[g][2:])<1e-6: continue
            v,lam,al2=rho(nd,e,R,T)
            if v is None: continue
            v/=lam
            if v>=cur: des,cur,al=nd,v,al2
            if st%300==299: sc=max(0.02,sc*0.7)
        const=sum(m[0]+m[1] for m in des)/NC
        print(f"eps={e} R={R} T={T} restart {rs}: rho_u/lambda = {cur:.6f}   mean weight on the constants {const:.4f}   alpha in [{min(al):.3f},{max(al):.3f}]",flush=True)
        best=max(best,cur)
    print("largest:",round(best,6))
main()
