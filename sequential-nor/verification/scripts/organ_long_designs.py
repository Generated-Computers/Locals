# Independent check of the referee's one-rail designs (types E(c) and Z1), written from the model:
# alpha_f = P[class f holds 0];  q_f = B + c*alpha_{c(f)} for E(c), q_f = 1-A for Z1;
# alpha_f = 1 - q_f alpha_{f-1} (periodic).  Density map Phi: alpha_below -> alpha.
# Mbar[k][c] = sum over gates g listing class c of c*alpha_{g-1} * sum_{j>=0, g+j = k mod T} prod q / (1-Q).
from fractions import Fraction
import math, sys, os
def consts(e):
    vp=((2*e+1)-math.sqrt(4*e+1))/(2*e); A=e*vp; B=e*(1-vp)
    q=(-(1-B)+math.sqrt((1-B)**2+4*(1-A)))/2
    return A,B,q,(1-e)/(1-q*q)
def parse(s): return [('Z1' if t=='Z1' else int(t[1:])) for t in s.split()]
def phi(des,al_below,e):
    A,B,_,_=consts(e); c=1-e; T=len(des)
    q=[(1-A) if t=='Z1' else B+c*al_below[t] for t in des]
    # periodic solution of a_f = 1 - q_f a_{f-1}: a_{T-1} = fixed point of the period map
    a=0.5
    for f in range(T): a=1-q[f]*a
    # period map is affine: a -> P + Q a with Q = (-1)^T prod q; solve
    Qp=1.0
    for x in q: Qp*=-x
    a0=0.0
    for f in range(T): a0=1-q[f]*a0
    P=a0; aT=P/(1-Qp)
    out=[0]*T; a=aT
    for f in range(T): a=1-q[f]*a; out[f]=a
    return out,q
def specrad(M):
    n=len(M); import copy
    X=[row[:] for row in M]; lg=0.0; vals=[]
    for s in range(50):
        nr=math.sqrt(sum(x*x for r in X for x in r))
        X=[[x/nr for x in r] for r in X]; lg+=math.log(nr); vals.append(lg/2**s)
        X=[[sum(X[i][k]*X[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        lg*=2
    return math.exp(vals[-1])
def analyse(name,s,e):
    des=parse(s); T=len(des); A,B,qs,lam=consts(e); c=1-e
    al=[0.6]*T
    for it in range(20000):
        new,_=phi(des,al,e)
        # damped iteration to find the fixed point even if it repels
        al2=[0.5*(x+y) for x,y in zip(al,new)]
        if max(abs(x-y) for x,y in zip(al,new))<1e-15: break
        al=al2
    new,q=phi(des,al,e); res=max(abs(x-y) for x,y in zip(al,new))
    Q=1.0
    for x in q: Q*=x
    M=[[0.0]*T for _ in range(T)]; J=[[0.0]*T for _ in range(T)]
    for g,t in enumerate(des):
        if t=='Z1': continue
        w=c*al[(g-1)%T]/(1-Q); pr=1.0
        for j in range(T):
            k=(g+j)%T; M[k][t]+=pr*w; J[k][t]+=pr*w*(-1)**j
            pr*=q[(g+j+1)%T]
    ru=specrad(M); rs=specrad(J)
    print(f"{name} eps={e}: fixed point residual {res:.1e}; rho_u={ru:.6f} rho_s={rs:.6f} lambda={lam:.6f} ratio={ru/lam:.5f}")
    return des,al
def cycle(name,s,e,iters=200000):
    des=parse(s); T=len(des)
    d=[0.65]*T   # all densities of ones 0.35
    for it in range(iters): d,_=phi(des,d,e)
    d1,_=phi(des,d,e); d2,_=phi(des,d1,e)
    print(f"{name} eps={e}: after {iters} steps of Phi: |Phi(a)-a|={max(abs(x-y) for x,y in zip(d,d1)):.4f}  |Phi^2(a)-a|={max(abs(x-y) for x,y in zip(d,d2)):.2e}")
D12="E2 E4 E2 E6 E2 Z1 E1 E2 E2 E4 E2 E4"
D16="E11 E12 E12 E14 E12 E14 E12 E14 E12 E14 E12 E14 E12 E0 E11 Z1"
S24=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"design_S24.txt")).read().split("\n")[0]
D40=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"design_D40.txt")).read().split("\n")[0]
for e in (0.445,0.46): analyse("D12",D12,e)
for e in (0.445,0.447,0.45,0.46): analyse("D16",D16,e)
for e in (0.445,0.448,0.4485,0.449,0.455): analyse("D40",D40,e)
for e in (0.445,0.46): analyse("S24",S24,e)
for e in (0.4444445,0.445,0.446,0.447,0.448,0.46): cycle("S24",S24,e,iters=20000)
cycle("latch E1 E0","E1 E0",0.445,iters=20000)
