# One-bit designs (types E(c), U, Z0, Z1; a NOT gate has the law of Z1): rho_u by closed formulas.
# design: list of types per phase: ('E',c) / 'U' / 'Z0' / 'Z1'.   One rail.  No numpy.
import math, sys
def consts(e):
    vp=((2*e+1)-math.sqrt(4*e+1))/(2*e); A=e*vp; B=e*(1-vp)
    q=(-(1-B)+math.sqrt((1-B)**2+4*(1-A)))/2
    return vp,A,B,q,(1-e)/(1-q*q)
def specrad(M):
    n=len(M); v=[1.0]*n; r=0
    for it in range(4000):
        w=[sum(M[i][j]*v[j] for j in range(n)) for i in range(n)]
        w=[0.5*(w[i]+r*v[i]) if it>0 else w[i] for i in range(n)]   # lazy step against periodicity
        s=sum(w)
        if s==0: return 0.0
        v=[x/s for x in w]
    w=[sum(M[i][j]*v[j] for j in range(n)) for i in range(n)]
    return sum(w)/sum(v)
def rho(design,e,iters=20000):
    vp,A,B,q,lam=consts(e); c=1-e; T=len(design)
    d=[0.35]*T
    def qs(d):
        m=sum(d)/T; out=[]
        for t in design:
            if t=='Z0': out.append(1-A-c)
            elif t=='Z1': out.append(1-A)
            elif t=='U': out.append(1-A-c*m)
            else: out.append(1-A-c*d[t[1]])
        return out
    p=0.4
    for it in range(iters):
        qq=qs(d); nd=[0]*T
        for rep in range(3):
            for f in range(T):
                p=(1-p)*qq[f]; nd[f]=p
        df=max(abs(nd[i]-d[i]) for i in range(T)); d=nd
        if df<1e-14: break
    qq=qs(d)
    a=[1-d[(f-1)%T] for f in range(T)]
    Q=1.0
    for x in qq: Q*=x
    M=[[0.0]*T for _ in range(T)]
    for g,t in enumerate(design):
        if t in('Z0','Z1'): continue
        w=c*a[g]/(1-Q); pr=1.0; cnt=[0.0]*T
        for j in range(T):
            cnt[(g+j)%T]+=pr*w; pr*=qq[(g+j+1)%T]
        if t=='U':
            for k in range(T):
                for cc in range(T): M[k][cc]+=cnt[k]/T
        else:
            for k in range(T): M[k][t[1]]+=cnt[k]
    return specrad(M),lam,d
if __name__=="__main__":
    print("checks (expect 1.74741 and 1.25473):", round(rho([('E',0)]*3+['Z1'],0.10)[0],5), round(rho([('E',0),('E',0),'Z1',('E',0),('E',3)],0.30)[0],5))
    es=(0.10,0.20,0.30,0.36,0.40,0.44,0.46)
    print("family E(0)^(T-1) Z1: rho_u / lambda")
    print("  T   "+"   ".join(f"{e:.2f}" for e in es))
    for T in (2,3,4,5,6,8,10,14,20,30):
        print(f"{T:3d}  "+"  ".join(f"{rho([('E',0)]*(T-1)+['Z1'],e)[0]/consts(e)[4]:.4f}" for e in es))
