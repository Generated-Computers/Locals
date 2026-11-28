# twoback_check.py: the lower and upper traces with pointers two back, on ALL parent arrays with
# 2 <= n <= NMAX.  For every round s it checks
#   (a) L^(s) <= x <= U^(s), L^(s-1) <= L^(s), U^(s) <= U^(s-1);
#   (b) D(i) <= s implies L^(s)_i = U^(s)_i                      (at least as fast as the pruned iteration);
#   (c) L^(s)_i = U^(s)_i implies i in K^(s) of the set-valued table iteration with reach 2.
# The traces: the standard rule of the relevance-pruned iteration, met (max of the lower, min of the
# upper values) at a gate k with pointer k-2 with the rules  x_k = (not x_{k-2}) and (read of gate k-1),
# x_k = x_{k-3} if gate k-1 has pointer k-3,  x_k = 0 if gate k-1 is a NOT gate.
# usage: python3 -I twoback_check.py NMAX        (the paper: 8)
import itertools, sys
def trace(j,n):
    x=[1]+[0]*n
    for i in range(1,n+1): x[i]=int(not(x[i-1] or x[j[i]]))
    return x
def depth(j,x,n):
    D=[0]*(n+1)
    for k in range(1,n+1):
        p=j[k]; notg=(p==k-1); reset=(not notg) and x[p]==1; rel=(x[k-1]==0)
        if k==1: D[k]=1
        elif reset and rel: D[k]=1+D[p]
        elif reset: D[k]=min(D[k-1],1+D[p])
        elif rel and not notg: D[k]=max(D[k-1],1+D[p])
        else: D[k]=D[k-1]
    return D
def setround(j,n,R,K,v):
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
def tworound(j,n,Lp,Up,meet=True):
    L=[1]+[0]*n; U=[1]+[1]*n
    for k in range(1,n+1):
        p=j[k]
        if p==k-1: l=1-U[k-1]; u=1-L[k-1]
        else: l=(1-U[k-1])&(1-Up[p]); u=(1-L[k-1])&(1-Lp[p])
        if p==k-2:
            q=j[k-1]
            if q==k-2: lt,ut=0,0
            else:
                lt=(1-U[k-2])&Lp[q]; ut=(1-L[k-2])&Up[q]
                if q==k-3: lt=max(lt,L[k-3]); ut=min(ut,U[k-3])
            if meet: l=max(l,lt); u=min(u,ut)
            else: l,u=lt,ut
        L[k]=l; U[k]=u
    return L,U
nmax=int(sys.argv[1]); cnt=0; faster=0; slower_than_sv=0; tot=[0,0,0]; nomeet_bad=0
for n in range(2,nmax+1):
    for tail in itertools.product(*[range(i) for i in range(1,n+1)]):
        j=(0,)+tail; x=trace(j,n); D=depth(j,x,n)
        L=[1]+[0]*n; U=[1]+[1]*n; K={0}; v={0:1}; s=0; r2=None; rsv=None
        L2=list(L); U2=list(U)
        while True:
            if r2 is None and L==U: r2=s
            if rsv is None and len(K)==n+1: rsv=s
            if r2 is not None and rsv is not None: break
            Ln,Un=tworound(j,n,L,U); Kn,w=setround(j,n,2,K,v); s+=1
            assert all(Ln[i]<=x[i]<=Un[i] for i in range(n+1)),("sound",j,s)
            assert all(L[i]<=Ln[i] and Un[i]<=U[i] for i in range(n+1)),("mono",j,s)
            assert all(Ln[i]==Un[i] for i in range(n+1) if D[i]<=s),("b",j,s)
            assert all(i in Kn for i in range(n+1) if Ln[i]==Un[i]),("c",j,s)
            L,U,K,v=Ln,Un,Kn,w
            La,Ua=tworound(j,n,L2,U2,meet=False)
            if not all(La[i]<=x[i]<=Ua[i] for i in range(n+1)): raise AssertionError(("nomeet sound",j,s))
            if not all(La[i]==Ua[i] for i in range(n+1) if D[i]<=s): nomeet_bad+=1
            L2,U2=La,Ua
            assert s<=n+2
        cnt+=1; faster+=(r2<max(D)); slower_than_sv+=(r2>rsv); tot[0]+=max(D); tot[1]+=r2; tot[2]+=rsv
print("%d arrays with 2 <= n <= %d: (a), (b), (c) hold"%(cnt,nmax))
print("rounds: fewer than max D in %d arrays, more than the set-valued iteration in %d; means %.3f (max D), %.3f (two back), %.3f (set-valued, reach 2)"%(faster,slower_than_sv,tot[0]/cnt,tot[1]/cnt,tot[2]/cnt))
print("without the meet with the standard rule: rounds where (b) fails:",nomeet_bad)
