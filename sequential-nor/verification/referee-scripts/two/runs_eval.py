# Referee evaluation of Lemma runs (b),(c),(d) and the numbers of Remark two(i). Floats on grids,
# local optimisation, and a 60-digit Decimal re-evaluation at the worst points.
import math, sys, random
from decimal import Decimal as D, getcontext
getcontext().prec=60
def consts(e):
    vp=((2*e+1)-math.sqrt(4*e+1))/(2*e); A=e*vp; B=e*(1-vp); r=1-A
    hi=(1-B)/(1-B*r); lo=1-r*hi
    return vp,A,B,r,lo,hi
def f(t): return t**2.15*(1-t)**1.65 if 0<t<1 else 0.0
t1,t2=23/56,43/56; S1=f(t1)/t1; S2=f(t2)/(1-t2)
def fhat(y): return S1*y if y<=t1 else (S2*(1-y) if y>=t2 else f(y))
def ratios(e,x,y,K):
    vp,A,B,r,lo,hi=consts(e); c=1-e; q=B+c*y
    am,a0=x,1-x*q
    v=(c*x*fhat(y), q*f(x), (1-q)*f(x)); out=[]
    for k in range(1,K+1):
        v=(c*v[2], r*v[0]+B*v[1], A*v[0]+(1-B)*v[1])
        am,a0=a0,(2-e)-r*a0-c*am
        out.append(((v[0]+v[1])/f(a0), sum(v)/min(f(lo),f(hi)), a0, (lo,hi)))
    return out
def ratios_dec(e,x,y,K):
    e=D(repr(e)); x=D(repr(x)); y=D(repr(y))
    vp=((2*e+1)-(4*e+1).sqrt())/(2*e); A=e*vp; B=e*(1-vp); r=1-A; hi=(1-B)/(1-B*r); lo=1-r*hi; c=1-e
    a_,b_=D(43)/20,D(33)/20
    F=lambda t: t**a_*(1-t)**b_
    T1,T2=D(23)/56,D(43)/56
    FH=lambda t: F(T1)/T1*t if t<=T1 else (F(T2)/(1-T2)*(1-t) if t>=T2 else F(t))
    q=B+c*y; am,a0=x,1-x*q; v=(c*x*FH(y),q*F(x),(1-q)*F(x)); out=[]
    for k in range(1,K+1):
        v=(c*v[2], r*v[0]+B*v[1], A*v[0]+(1-B)*v[1]); am,a0=a0,(2-e)-r*a0-c*am
        out.append((float((v[0]+v[1])/F(a0)), float(sum(v)/min(F(lo),F(hi)))))
    return out
K=40
def grid(e,nx,ny):
    vp,A,B,r,lo,hi=consts(e)
    best=[(-1,None)]*K; bestc=(-1,None); arange=0
    for i in range(nx+1):
        x=lo+(hi-lo)*i/nx
        for j in range(ny+1):
            y=j/ny; R=ratios(e,x,y,K)
            for k in range(K):
                if R[k][0]>best[k][0]: best[k]=(R[k][0],(x,y))
                if not (lo-1e-12<=R[k][2]<=hi+1e-12): arange+=1
            if R[17][1]>bestc[0]: bestc=(R[17][1],(x,y))
    return best,bestc,arange
if __name__=='__main__':
    print("Remark (i) numbers, grid 400x400:")
    for e in (0.455,0.50,0.80):
        best,bestc,ar=grid(e,400,400)
        print(" eps",e,"k=1..6:",["%.4f"%b[0] for b in best[:6]],"argmax k=1",best[0][1],"| max over k>=6: %.4f"%max(b[0] for b in best[5:]),"| (c) ratio at 18: %.3e"%bestc[0],"alpha out of range:",ar)
    # y=1 case at 0.455
    e=0.455; vp,A,B,r,lo,hi=consts(e); m=[-1]*K
    for i in range(4001):
        x=lo+(hi-lo)*i/4000; R=ratios(e,x,1.0,K)
        for k in range(K): m[k]=max(m[k],R[k][0])
    print(" y=1 (NOT gate before run), eps 0.455: k=1..5",["%.4f"%z for z in m[:5]])
    # eps sweep
    print("eps sweep (grid 120x120): max over k of ratio (b)/(d), and (c) ratio at k=18")
    gb=(-1,None); gc=(-1,None)
    es=[0.455+i*0.0005 for i in range(0,91)]+[0.5+i*0.01 for i in range(1,50)]+[0.995,0.999,0.9999,0.999999]
    for e in es:
        best,bestc,ar=grid(e,120,120)
        mk=max(range(K),key=lambda k:best[k][0])
        if best[mk][0]>gb[0]: gb=(best[mk][0],(e,mk+1)+best[mk][1])
        if bestc[0]>gc[0]: gc=(bestc[0],(e,)+bestc[1])
        if ar: print("  alpha out of [lo,hi] at eps",e,ar)
    print(" global max ratio (b):",gb); print(" global max (c):",gc)
    # local optimisation (random restarts, coordinate perturbation) over (eps,x,y) per k
    rng=random.Random(7)
    for k in (1,2,3,4,5,17,18,25):
        bv=-1;bp=None
        for rs in range(30):
            e=rng.uniform(0.455,0.999); vp,A,B,r,lo,hi=consts(e); u=rng.random(); y=rng.random()
            cur=ratios(e,lo+(hi-lo)*u,y,k)[k-1][0]; st=0.2
            for it in range(600):
                e2=min(max(e+st*rng.uniform(-1,1)*0.5,0.455),0.999999); u2=min(max(u+st*rng.uniform(-1,1),0),1); y2=min(max(y+st*rng.uniform(-1,1),0),1)
                vp,A,B,r,lo,hi=consts(e2); val=ratios(e2,lo+(hi-lo)*u2,y2,k)[k-1][0]
                if val>cur: cur=val;e,u,y=e2,u2,y2
                elif it%30==29: st*=0.7
            if cur>bv: bv=cur; vp,A,B,r,lo,hi=consts(e); bp=(e,lo+(hi-lo)*u,y)
        dd=ratios_dec(bp[0],bp[1],bp[2],k)[k-1][0]
        print(" k=%d local-opt max ratio %.6f at (eps,x,y)=(%.6f,%.6f,%.6f); Decimal recheck %.6f"%(k,bv,bp[0],bp[1],bp[2],dd))
    # (c) worst via Decimal
    e,x,y=gc[1]; print(" (c) Decimal recheck at worst grid point:",ratios_dec(e,x,y,18)[17][1])
    # rho_u of the design (L2, E(phase 0)) at 0.46, and lambda
    e=0.46; vp,A,B,r,lo,hi=consts(e); c=1-e
    # fixed point: alpha0 (L2 gate, class read), alpha1 (E gate)
    a0,a1=0.6,0.6
    for it in range(5000):
        y=a0                       # E gate reads class of phase 0: target 0 w.p. alpha0
        a1n=1-(B+c*y)*a0           # gate at phase 1 follows phase 0
        a0n=(2-e)-r*a1n-c*a0       # L2 at phase 0 follows phase 1, two back is phase 0
        a0,a1=a0n,a1n
    q=B+c*a0; M=0.0; v=(c*a0,0.0,0.0)   # source at phase 1, predecessor phase 0
    for t in range(400):
        # next gate: L2 (phase 0) -> count; then E (phase 1)
        v=(c*v[2], r*v[0]+B*v[1], A*v[0]+(1-B)*v[1]); M+=v[0]+v[1]
        v=(0.0,q*(v[0]+v[1]),(1-q)*(v[0]+v[1]))
    qq=None
    # q(eps) of the plain chain: fixed point pi0 = q/(1+q) with 1-q = A+(1-eps)pi0
    p=0.4
    for it in range(2000): qv=1-(A+c*p); p=qv/(1+qv)
    print(" rho_u(L2,E) at 0.46 = %.4f (fixed point alpha0=%.4f alpha1=%.4f); lambda=%.4f"%(M,a0,a1,c/(1-qv*qv)))
