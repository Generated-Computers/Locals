# setval_check.py: the proposition on the set-valued table iteration, on ALL parent arrays with
# 2 <= n <= NMAX and every reach 1 <= R <= RMAX. For each array and reach it runs the set-valued
# iteration from its definition and checks, for every round s:
#   (a) the values of determined positions are those of the trace, and K^(s) is contained in K^(s+1);
#   (b) D_R <= D, D(i) <= s implies i in K^(s), and D_R(i) <= s implies i in K^(s+1)   (D by the
#       four cases of the lemma on the pruned depth with D(0)=0, D(1)=1; D_R with c_k + D_R(p) in
#       place of 1 + D(p)); for R = 1 also K^(s) = {i : D(i) <= s};
#   (c) the iterate y^(s) of the table-Jacobi iteration agrees with the trace on K^(s), and the
#       number of applications of G_R until the fixed point is confirmed is at most rounds + 1;
#   (e) K^(s) for the reach R is contained in K^(s) for the reach R+1, for every s;
#   (f) the number of rounds is at most ceil(n/(R+1)).
# usage: python3 -I setval_check.py NMAX RMAX        (the paper: 7 4)
import itertools, sys
def trace(j,n):
    x=[1]+[0]*n
    for i in range(1,n+1): x[i]=int(not(x[i-1] or x[j[i]]))
    return x
def depth(j,x,n,R):          # R=None: the depth D; otherwise D_R
    D=[0]*(n+1)
    for k in range(1,n+1):
        p=j[k]; notg=(p==k-1); reset=(not notg) and x[p]==1; rel=(x[k-1]==0)
        c=1 if (R is None or k-p>R) else 0
        if k==1: D[k]=1
        elif reset and rel: D[k]=c+D[p]
        elif reset: D[k]=min(D[k-1],c+D[p])
        elif rel and not notg: D[k]=max(D[k-1],c+D[p])
        else: D[k]=D[k-1]
    return D
def setround(j,n,R,K,v):     # one round: returns (K', w)
    S={tuple([0]*(R-1)+[1])}; Kn={0}; w={0:1}
    for i in range(1,n+1):
        p=j[i]; T=set()
        for sg in S:
            a=sg[-1]
            if i-p<=R: B=(sg[R-(i-p)],)
            elif p in K: B=(v[p],)
            else: B=(0,1)
            for b in B: T.add(sg[1:]+(int(not(a or b)),))
        S=T; last={sg[-1] for sg in S}
        if len(last)==1: Kn.add(i); w[i]=next(iter(last))
    return Kn,w
def G(j,n,R,y):
    z=[1]+[0]*n
    for i in range(1,n+1):
        p=j[i]; r=z[p] if i-p<=R else y[p]; z[i]=int(not(z[i-1] or r))
    return z
nmax=int(sys.argv[1]); rmax=int(sys.argv[2]); cnt=0; worst=0; tight=0
for n in range(2,nmax+1):
    for tail in itertools.product(*[range(i) for i in range(1,n+1)]):
        j=(0,)+tail; x=trace(j,n); D=depth(j,x,n,None)
        prev=None
        for R in range(1,rmax+1):
            DR=depth(j,x,n,R); assert all(DR[i]<=D[i] for i in range(n+1)),("D_R<=D",j,R)
            K={0}; v={0:1}; y=[1]+[0]*n; s=0; hist=[set(K)]
            while True:
                assert all(v[p]==x[p] for p in K),("a",j,R,s)
                assert all(y[i]==x[i] for i in K),("c",j,R,s)
                assert all(i in K for i in range(n+1) if D[i]<=s),("b, D",j,R,s)
                if R==1: assert K=={i for i in range(n+1) if D[i]<=s},("R=1",j,s)
                if len(K)==n+1: break
                Kn,w=setround(j,n,R,K,v); assert K<=Kn,("mono",j,R,s)
                assert all(i in Kn for i in range(n+1) if DR[i]<=s),("b",j,R,s)
                K,v=Kn,w; y=G(j,n,R,y); s+=1; hist.append(set(K)); assert s<=n+2
            rounds=s
            if prev is not None:
                assert len(hist)<=len(prev),("e rounds",j,R)
                for q in range(len(prev)): assert prev[q]<=hist[min(q,len(hist)-1)],("e",j,R,q)
            prev=hist
            assert rounds<=min(max(D),max(DR)+1),("rounds",j,R)
            assert rounds<=-(-n//(R+1)),("f",j,R)
            y=[1]+[0]*n; apps=0
            while True:
                z=G(j,n,R,y); apps+=1
                if z==y: break
                y=z
            assert y==x and apps<=rounds+1,("apps",j,R,apps,rounds)
            worst=max(worst,apps-rounds); tight+=(rounds==max(DR)+1); cnt+=1
print("set-valued table iteration: %d (array, reach) pairs with 2 <= n <= %d, R <= %d: (a), (b), (c), (e), (f) hold;"%(cnt,nmax,rmax))
print("largest (applications of G_R) - (rounds) = %d; rounds = max D_R + 1 in %d pairs"%(worst,tight))
