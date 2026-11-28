#!/usr/bin/env python3 -I
"""exact_indep.py -- exact growth (branching mean matrix) for one-rail periodic designs whose reads
are independent in the limit M -> infinity:  U, Ic, O, Z, N, Hd, V, Jc   (referee's own code).

Model (Definition 'periodic designs' of the paper, one rail): gate of phase f reads a one with
probability y1_f = eps*varpi + (1-eps)*tau_f, tau_f the density of the listed class (0 for O and N,
1 for Z).  alpha_f = P(x=0) solves alpha_f = 1 - alpha_{f-1} q_f, q_f = 1 - y1_f.
A flipped position of class c changes a gate g that lists it with probability (1-eps) alpha_{g-1},
and the change continues to the next gate with probability q.  M1[k][c], M2[k][c] = mean number of
changed positions of class k one / two layers up.
Output: per-generation value rho(M1+M2); per-layer growth g with rho(M1/g + M2/g^2) = 1.
usage: python3 -I exact_indep.py eps design [design ...]
"""
import sys, math

def varpi(e):
    return (2*e+1-math.sqrt(4*e+1))/(2*e)

def parse(s):
    out=[]
    for tok in s.split(','):
        t=tok[0]; p=int(tok[1:]) if len(tok)>1 else 0
        assert t in 'UIOZNHVJ', tok
        out.append((t,p))
    return out

def target(rule, f, d, T):
    t,p=rule
    if t in 'UV': return sum(d)/T
    if t in 'IJ': return d[p]
    if t=='H': return d[(f+p)%T]
    if t=='Z': return 1.0
    return 0.0   # O and N

def density(rules, eps, iters=100000, tol=1e-15):
    T=len(rules); w=varpi(eps)
    d=[0.37+0.01*i/T for i in range(T)]
    dprev=d[:]
    for it in range(iters):
        # H rules read two layers below: second-order map; iterate with (d, dprev)
        q=[]
        for f,r in enumerate(rules):
            src = dprev if r[0] in 'HVJ' else d
            q.append(1-(eps*w+(1-eps)*target(r,f,src,T)))
        # periodic solution of alpha_f = 1 - alpha_{f-1} q_f
        # alpha_{T-1} = A + B*alpha_{T-1}
        A=0.0;B=1.0
        for f in range(T):
            A=1-A*q[f]; B=-B*q[f]
        aT=A/(1-B)
        al=[0.0]*T; prev=aT
        for f in range(T):
            prev=1-prev*q[f]; al[f]=prev
        dn=[1-x for x in al]
        err=max(abs(dn[i]-d[i]) for i in range(T))+max(abs(d[i]-dprev[i]) for i in range(T))
        dprev=d; d=dn
        if err<tol: break
    return d, err

def matrices(rules, eps):
    T=len(rules); w=varpi(eps)
    d,err=density(rules,eps)
    q=[1-(eps*w+(1-eps)*target(r,f,d,T)) for f,r in enumerate(rules)]
    al=[1-x for x in d]
    # N[g][k] = (1-eps) alpha_{g-1} sum_{j>=0, g+j=k mod T} prod_{i=1..j} q_{g+i}
    P=1.0
    for f in range(T): P*=q[f]
    N=[[0.0]*T for _ in range(T)]
    for g in range(T):
        base=(1-eps)*al[(g-1)%T]
        prod=1.0
        for j in range(T):
            k=(g+j)%T
            if j>0: prod*=q[k]
            N[g][k]=base*prod/(1-P)
    M1=[[0.0]*T for _ in range(T)]; M2=[[0.0]*T for _ in range(T)]
    for g,(t,p) in enumerate(rules):
        for c in range(T):
            w1=w2=0.0
            if t=='U': w1=1.0/T
            elif t=='I': w1=1.0 if p==c else 0.0
            elif t=='V': w2=1.0/T
            elif t=='J': w2=1.0 if p==c else 0.0
            elif t=='H': w2=1.0 if (g+p)%T==c else 0.0
            if w1:
                for k in range(T): M1[k][c]+=w1*N[g][k]
            if w2:
                for k in range(T): M2[k][c]+=w2*N[g][k]
    return d,err,M1,M2

def rho(A, iters=200000):
    n=len(A)
    B=[[A[i][j]+(1.0 if i==j else 0.0) for j in range(n)] for i in range(n)]
    v=[1.0]*n; lam=0.0
    for it in range(iters):
        u=[sum(B[i][j]*v[j] for j in range(n)) for i in range(n)]
        s=sum(u)
        new=s/sum(v)
        u=[x/s for x in u]
        if abs(new-lam)<1e-14 and it>50: lam=new; break
        lam=new; v=u
    return lam-1.0

def layer_growth(M1,M2):
    n=len(M1)
    if all(M2[i][j]==0 for i in range(n) for j in range(n)): return rho(M1)
    lo,hi=0.2,3.0
    for _ in range(60):
        g=(lo+hi)/2
        A=[[M1[i][j]/g+M2[i][j]/g/g for j in range(n)] for i in range(n)]
        if rho(A,20000)>1: lo=g
        else: hi=g
    return (lo+hi)/2

if __name__=='__main__':
    eps=float(sys.argv[1])
    w=varpi(eps); A=eps*w; B=eps-A
    q=(-(1-B)+math.sqrt((1-B)**2+4*(1-A)))/2
    lam=(1-eps)/(1-q*q)
    print("eps %.4f varpi %.6f q %.6f pi0 %.6f lambda %.6f"%(eps,w,q,q/(1+q),lam))
    for s in sys.argv[2:]:
        rules=parse(s)
        d,err,M1,M2=matrices(rules,eps)
        n=len(rules)
        S=[[M1[i][j]+M2[i][j] for j in range(n)] for i in range(n)]
        f2=sum(1 for t,p in rules if t in 'HVJ')/sum(1 for t,p in rules if t in 'UIHVJ')
        gl=layer_growth(M1,M2)
        naive=None
        if f2>0:
            a=(1-f2)*lam; b=f2*lam
            naive=(a+math.sqrt(a*a+4*b))/2
        print("%-60s per-generation %.5f  per-layer %.5f%s  (density err %.1e, min/max density %.4f %.4f)"%(
            s,rho(S),gl,("  [mixing formula g^2=(1-f)lam g+f lam, f=%.3f: %.5f]"%(f2,naive)) if naive else "",err,min(d),max(d)))
