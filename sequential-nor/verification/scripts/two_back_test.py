# two_back_test.py -- numerical test of the theorem on designs with pointers two back (not a proof).
# For random designs with the types E(c), Z0, Z1, L1, L2 and random d it computes
#   sum_c Mbar[k][c](d) f(1-d_c) / f(alpha_k),      which the theorem bounds by 0.9998,
# with Mbar computed twice: from the joint law of two copies of the chain (brute force over pairs
# of windows), and from the three-state recursion of the lemma on chains with pointers two back.
# It also compares alpha at the gates L2 with the second-order recursion of that lemma.
# usage: python3 -I two_back_test.py seed number_of_designs
import sys, random
W = [(0,0),(0,1),(1,0)]                       # windows (x_{i-1}, x_i)
W = [(0,0),(0,1),(1,0)]                       # windows (z_{i-1}, z_i)

def varpi(e):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        m = (lo+hi)/2
        if e*(1-m)**2 - m > 0: lo = m
        else: hi = m
    return lo

def step(w, r):                                # window after a gate that reads r
    return (w[1], 0 if (w[1] or r) else 1)

def reads(typ, w):                             # read of a kept gate of a local type
    return w[1] if typ == 'L1' else (w[0] if typ == 'L2' else (0 if typ == 'Z1' else 1))

def clean(eps, des, d, iters=4000):
    """stationary laws of the true window before each phase; d[c] = density of ones of class c"""
    T = len(des); w = varpi(eps)
    law = {x: 1/3 for x in W}
    out = None
    for it in range(iters):
        laws = []
        for ph in range(T):
            laws.append(dict(law)); typ = des[ph]; new = {x: 0.0 for x in W}
            for x, p in law.items():
                for r, pr in ((1, eps*w), (0, eps*(1-w))):
                    new[step(x, r)] += p*pr
                if typ[0] == 'E':
                    dc = d[int(typ[1:])]
                    new[step(x, 1)] += p*(1-eps)*dc; new[step(x, 0)] += p*(1-eps)*(1-dc)
                else:
                    new[step(x, reads(typ, x))] += p*(1-eps)
            law = new
        out = laws
    return out                                  # out[ph] = law of the window before gate ph

pa,pb=43/20,33/20
f=lambda t: t**pa*(1-t)**pb if 0<t<1 else 0.0

def mbar_brute(eps,des,d,horizon=300):
    T=len(des); w=varpi(eps); laws=clean(eps,des,d,iters=400)
    M=[[0.0]*T for _ in range(T)]
    for g in range(T):
        if des[g][0]!='E': continue
        c=int(des[g][1:]); st={}
        for x,p in laws[g].items():          # copy 1 gets target 0, copy 2 target 1, read kept
            key=(step(x,0),step(x,1)); st[key]=st.get(key,0.0)+p*(1-eps)
        ph=g
        for t in range(horizon):
            M[ph][c]+=sum(p for (x,z),p in st.items() if x[1]!=z[1])
            ph=(ph+1)%T; typ=des[ph]; new={}
            for (x,z),p in st.items():
                if x==z: continue
                opts=[(1,eps*w,None),(0,eps*(1-w),None)]
                if typ[0]=='E':
                    dc=d[int(typ[1:])]; opts+=[(1,(1-eps)*dc,None),(0,(1-eps)*(1-dc),None)]
                else: opts+=[(None,1-eps,typ)]
                for r,pr,loc in opts:
                    key=(step(x,r),step(z,r)) if loc is None else (step(x,reads(loc,x)),step(z,reads(loc,z)))
                    new[key]=new.get(key,0.0)+p*pr
            st=new
    alpha=[sum(p for x,p in laws[(k+1)%T].items() if x[1]==0) for k in range(T)]
    return M,alpha

def mbar_rec(eps,des,d,alpha,horizon=300):
    T=len(des); w=varpi(eps); A=eps*w; B=eps*(1-w); r=1-A; c1=1-eps
    M=[[0.0]*T for _ in range(T)]
    for g in range(T):
        if des[g][0]!='E': continue
        c=int(des[g][1:]); v=(c1*alpha[(g-1)%T],0.0,0.0); ph=g
        for t in range(horizon):
            M[ph][c]+=v[0]+v[1]; ph=(ph+1)%T; typ=des[ph]
            if typ=='L2': v=(c1*v[2], r*v[0]+B*v[1], A*v[0]+(1-B)*v[1])
            else:
                if typ[0]=='E': q=B+c1*(1-d[int(typ[1:])])
                elif typ=='L1': q=r
                elif typ=='Z0': q=B
                else: q=B+c1
                v=(0.0, q*(v[0]+v[1]), (1-q)*(v[0]+v[1]))
    return M

rng=random.Random(int(sys.argv[1]) if len(sys.argv)>1 else 1); n=int(sys.argv[2]) if len(sys.argv)>2 else 200
worst=0; maxdiff=0; maxal=0
for it in range(n):
    eps=rng.choice([0.455,0.455,0.46,0.5,0.6,0.8]); T=rng.randint(2,7)
    des=[]
    for i in range(T):
        u=rng.random()
        des.append('L2' if u<0.45 else 'L1' if u<0.55 else 'Z0' if u<0.6 else 'Z1' if u<0.65 else 'E%d'%rng.randrange(T))
    if not any(t[0]=='E' for t in des): des[rng.randrange(T)]='E%d'%rng.randrange(T)
    d=[rng.random() for _ in range(T)]
    if rng.random()<0.3: d=[rng.choice([0.0,1.0,rng.random()]) for _ in range(T)]
    M,alpha=mbar_brute(eps,des,d); M2=mbar_rec(eps,des,d,alpha)
    maxdiff=max(maxdiff,max(abs(M[k][c]-M2[k][c]) for k in range(T) for c in range(T)))
    # second-order recursion of alpha at L2 gates
    for k in range(T):
        if des[k]=='L2':
            w=varpi(eps); pred=2-eps-(1-eps*w)*alpha[(k-1)%T]-(1-eps)*alpha[(k-2)%T]
            maxal=max(maxal,abs(pred-alpha[k]))
    for k in range(T):
        lhs=sum(M[k][c]*f(1-d[c]) for c in range(T))
        if lhs>0:
            ratio=lhs/f(alpha[k])
            if ratio>worst: worst=ratio; wd=(eps,des,[round(x,3) for x in d],k)
print('designs',n,'largest ratio %.5f'%worst,wd)
print('largest difference between brute force and recursion %.2e; alpha recursion %.2e'%(maxdiff,maxal))
