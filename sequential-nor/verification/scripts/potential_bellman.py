# Relaxed problem for one-bit designs: the largest growth rate of
#   V(alpha') = sup [ q(alpha2) V(alpha1) + c*alpha1*V(alpha2) ],  alpha' = 1 - alpha1*q(alpha2),
# over alpha1, alpha2 in the invariant interval, q(alpha2) = B + c*alpha2, plus the dead reads
# q = B (Z0) and q = 1-A (Z1) without the second term.  alpha = probability that a class holds 0.
import math, sys
def consts(e):
    vp=((2*e+1)-math.sqrt(4*e+1))/(2*e); A=e*vp; B=e*(1-vp)
    q=(-(1-B)+math.sqrt((1-B)**2+4*(1-A)))/2
    return vp,A,B,q,(1-e)/(1-q*q)
def growth(e,N=240,iters=400,verbose=False):
    vp,A,B,qs,lam=consts(e); c=1-e
    hi=(1-B)/(1-B*(1-A)); lo=1-hi*(1-A)
    xs=[lo+(hi-lo)*i/(N-1) for i in range(N)]
    f=[1.0]*N
    def interp(f,a):
        t=(a-lo)/(hi-lo)*(N-1)
        if t<=0: return f[0]
        if t>=N-1: return f[-1]
        i=int(t); u=t-i; return f[i]*(1-u)+f[i+1]*u
    rate=0
    for it in range(iters):
        g=[0.0]*N
        for k,ap in enumerate(xs):
            best=0.0
            for (q,a2idx) in [(B,None),(1-A,None)]+[(B+c*xs[j],j) for j in range(N)]:
                a1=(1-ap)/q
                if a1<lo-1e-12 or a1>hi+1e-12: continue
                v=q*interp(f,a1)
                if a2idx is not None: v+=c*a1*f[a2idx]
                if v>best: best=v
            g[k]=best
        s=max(g); rate=s/max(f); f=[v/s for v in g]
    if verbose:
        print("  f at lo, plain point, hi:",round(f[0],4),round(interp(f,1/(1+qs)),4),round(f[-1],4))
    return rate,lam,lo,hi
if __name__=="__main__":
    for e in [float(a) for a in sys.argv[1:]]:
        r,lam,lo,hi=growth(e,N=160,iters=200,verbose=True)
        print(f"eps={e}: relaxed growth rate {r:.4f}   plain lambda {lam:.4f}   interval [{lo:.3f},{hi:.3f}]",flush=True)
