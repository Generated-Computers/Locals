# Referee check of Lemma twochain (a),(b),(c): own joint Markov chain of two copies vs. the paper's recursions.
import sys, random, itertools
from fractions import Fraction as Fr
def NOR(a,b): return 1 if (a==0 and b==0) else 0
# gate = ('M', t) target is 1 w.p. t ; ('L1',) ; ('L2',)
def outcomes(gate, eps, vp, one):
    """list of (prob, kind) with kind in 'r0','r1','k0','k1','kL1','kL2'"""
    A=eps*vp; B=eps*(one-vp)
    o=[(A,'r1'),(B,'r0')]
    if gate[0]=='M':
        t=gate[1]; o+= [((one-eps)*t,'k1'),((one-eps)*(one-t),'k0')]
    elif gate[0]=='L1': o.append((one-eps,'kL1'))
    else: o.append((one-eps,'kL2'))
    return o
def read(kind,u,v,force=None):
    if kind=='r1': return 1
    if kind=='r0': return 0
    if kind=='k1' or kind=='k0':
        return force if force is not None else (1 if kind=='k1' else 0)
    if kind=='kL1': return v
    return u
def step1(dist, gate, eps, vp, one):
    new={}
    for (u,v),p in dist.items():
        if p==0: continue
        for pr,kind in outcomes(gate,eps,vp,one):
            s=(v,NOR(v,read(kind,u,v)))
            new[s]=new.get(s,0)+p*pr
    return new
def solve(Mat, rhs):
    n=len(rhs); M=[row[:]+[rhs[i]] for i,row in enumerate(Mat)]
    for c in range(n):
        piv=max(range(c,n), key=lambda r: abs(M[r][c]))
        M[c],M[piv]=M[piv],M[c]
        for r in range(n):
            if r!=c and M[r][c]!=0:
                fct=M[r][c]/M[c][c]
                M[r]=[a-fct*b for a,b in zip(M[r],M[c])]
    return [M[i][n]/M[i][i] for i in range(n)]
ST=[(0,0),(0,1),(1,0),(1,1)]
def stationary(gates, eps, vp, one):
    """law of (x_{g-1},x_g) after gate index T-1 (i.e. before gate 0), then all phases"""
    T=len(gates); zero=one-one
    # period matrix
    cols=[]
    for s in ST:
        d={s:one}
        for g in gates: d=step1(d,g,eps,vp,one)
        cols.append([d.get(t,zero) for t in ST])
    # pi[t] = sum_s pi[s] cols[s][t]
    Mat=[[cols[s][t]-(one if s==t else zero) for s in range(4)] for t in range(4)]
    Mat[3]=[one]*4; rhs=[zero,zero,zero,one]
    pi=solve(Mat,rhs)
    d={ST[i]:pi[i] for i in range(4)}
    laws=[]   # laws[i] = law of (x_{i-1},x_i) after gate i
    for g in gates:
        d=step1(d,g,eps,vp,one); laws.append(d)
    return laws
def two_copies(gates, laws, g, nsteps, eps, vp, one):
    """source gate g (phase index); returns list over h=g..g+nsteps of (a,b,c,signed) from the joint law"""
    T=len(gates); zero=one-one
    prev=laws[(g-1)%T]
    joint={}
    gate=gates[g]; assert gate[0]=='M'
    A=eps*vp; B=eps*(one-vp)
    for (u,v),p in prev.items():
        if p==0: continue
        for pr,kind in [(A,'r1'),(B,'r0'),(one-eps,'k')]:
            if kind=='k': r0,r1=0,1
            else: r0=r1=read(kind,u,v)
            s=(v,NOR(v,r0),v,NOR(v,r1))
            joint[s]=joint.get(s,zero)+p*pr
    res=[]
    def stats(j):
        a=b=c=sg=zero
        for (u,v,u2,v2),p in j.items():
            if v!=v2 and u==u2: a+=p
            if v!=v2 and u!=u2: b+=p
            if v==v2 and u!=u2: c+=p
            if (v,v2)==(0,1): sg+=p
            if (v,v2)==(1,0): sg-=p
        return (a,b,c,sg)
    res.append(stats(joint))
    for k in range(1,nsteps+1):
        gate=gates[(g+k)%T]; new={}
        for (u,v,u2,v2),p in joint.items():
            if p==0: continue
            for pr,kind in outcomes(gate,eps,vp,one):
                s=(v,NOR(v,read(kind,u,v)),v2,NOR(v2,read(kind,u2,v2)))
                new[s]=new.get(s,zero)+p*pr
        joint=new; res.append(stats(joint))
    return res
def paper_recursion(gates, alpha, g, nsteps, eps, vp, one):
    T=len(gates); zero=one-one
    A=eps*vp; B=eps*(one-vp); r=one-A
    a=(one-eps)*alpha[(g-1)%T]; b=zero; c=zero; res=[(a,b,c)]
    for k in range(1,nsteps+1):
        gate=gates[(g+k)%T]
        if gate[0]=='L2':
            a,b,c=(one-eps)*c, r*a+B*b, A*a+(one-B)*b
        else:
            q = B+(one-eps)*(one-gate[1]) if gate[0]=='M' else one-A
            a,b,c=zero, q*(a+b), (one-q)*(a+b)
        res.append((a,b,c))
    return res
def check_design(gates, eps, vp, one, exact, nsteps):
    T=len(gates); zero=one-one
    laws=stationary(gates,eps,vp,one)
    alpha=[sum(p for (u,v),p in laws[i].items() if v==0) for i in range(T)]
    A=eps*vp; B=eps*(one-vp); r=one-A
    hi=(one-B)/(one-B*r); lo=one-r*hi
    err=0.0
    # no consecutive ones
    for i in range(T):
        e=laws[i].get((1,1),zero); err=max(err,abs(float(e)))
    # (a)
    for i in range(T):
        gate=gates[i]
        if gate[0]=='L2': pred=(2*one-eps)-r*alpha[(i-1)%T]-(one-eps)*alpha[(i-2)%T]
        else:
            q = B+(one-eps)*(one-gate[1]) if gate[0]=='M' else one-A
            pred=one-q*alpha[(i-1)%T]
        err=max(err,abs(float(pred-alpha[i])))
        tol=0 if exact else 1e-13
        assert lo-tol<=alpha[i]<=hi+tol, ("alpha range",gates,eps,alpha[i],lo,hi)
    # (c)
    for g in range(T):
        if gates[g][0]!='M': continue
        J=two_copies(gates,laws,g,nsteps,eps,vp,one)
        P=paper_recursion(gates,alpha,g,nsteps,eps,vp,one)
        for (a,b,c,sg),(pa,pb,pc) in zip(J,P):
            err=max(err,abs(float(a-pa)),abs(float(b-pb)),abs(float(c-pc)))
            assert abs(sg)<=a+b+ (0 if exact else 1e-15)
    return err, alpha
def rand_design(rng, T, one, exact, force=None):
    gates=[]
    for i in range(T):
        k=rng.choice(['M','M','L1','L2','L2','Z0','Z1']) if force is None else force[i]
        if k=='M':
            t=Fr(rng.randint(0,40),40) if exact else rng.random()
            gates.append(('M',t if exact else t))
        elif k=='Z0': gates.append(('M',one))      # target bit 1 (position 0 holds 1)
        elif k=='Z1': gates.append(('M',one-one))
        else: gates.append((k,))
    return gates
if __name__=='__main__':
    rng=random.Random(12345)
    # exact rational runs: varpi rational, eps = varpi/(1-varpi)^2
    worst=0
    cnt=0
    for vpn in [Fr(1,4),Fr(13,50),Fr(3,10),Fr(7,20),Fr(1,5)]:
        eps=vpn/(1-vpn)**2
        for trial in range(60):
            T=rng.randint(1,7)
            gates=rand_design(rng,T,Fr(1),True)
            if not any(g[0]=='M' for g in gates): gates[rng.randrange(T)]=('M',Fr(rng.randint(0,40),40))
            e,al=check_design(gates,eps,vpn,Fr(1),True,3*T+6)
            worst=max(worst,e); cnt+=1
    print("exact rational designs:",cnt,"max discrepancy (must be exactly 0):",worst)
    # special structured patterns, exact
    pats=[['M','L2'],['L2','M'],['M','L2','L2'],['M','L2','L2','L2','L2','L2','L2'],['L2','L2','M','L2'],
          ['L1','L2'],['L1','L2','L2','M'],['M','L1','L2','L2','L1','L2','M'],['L2','L2','L2','M','L1'],
          ['L2','L1','L2','L1','M'],['Z0','L2','L2'],['Z1','L2','L2','M'],['M'],['M','L1'],['L2','L2','L2','L2','L2','L2','M']]
    for vpn in [Fr(1,4),Fr(13,50)]:
        eps=vpn/(1-vpn)**2
        for p in pats:
            for trial in range(4):
                gates=rand_design(rng,len(p),Fr(1),True,force=p)
                e,al=check_design(gates,eps,vpn,Fr(1),True,4*len(p)+8); worst=max(worst,e)
    print("structured patterns exact: max discrepancy:",worst)
    # all-L2 chain and L1/L2-only chains: alpha formula only
    for p in [['L2'],['L2','L2','L2'],['L1','L2'],['L1']]:
        vpn=Fr(13,50); eps=vpn/(1-vpn)**2
        gates=[(k,) for k in p]
        e,al=check_design(gates,eps,vpn,Fr(1),True,5); print(p,"alpha",[float(x) for x in al],"err",e)
    # float runs across eps in (0,1)
    import math
    wf=0; n=0
    for trial in range(1500):
        eps=rng.uniform(0.02,0.999); vp=((2*eps+1)-math.sqrt(4*eps+1))/(2*eps)
        T=rng.randint(1,9); gates=rand_design(rng,T,1.0,False)
        if not any(g[0]=='M' for g in gates): gates[rng.randrange(T)]=('M',rng.random())
        e,al=check_design(gates,eps,vp,1.0,False,5*T+10); wf=max(wf,e); n+=1
    print("float designs:",n,"max discrepancy:",wf)
