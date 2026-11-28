# Exhaustive check, all parent arrays with n <= 8 (46,233 arrays):
#   after s rounds the two-valued prefix-Jacobi iterate agrees with the trace on every position
#   that the relevance-pruned iteration (001c, Definition 3.10) knows after s rounds;
#   a fixed point of the Jacobi map is the trace.
import itertools, sys
def trace(j):
    x=[1]
    for i in range(1,len(j)): x.append(1-(x[i-1]|x[j[i]]))
    return x
def pruned_rounds(j,x):
    n=len(j)-1; K={0}; v={0:1}; hist=[set(K)]
    while len(K)<n+1:
        res=[False]*(n+1)
        for k in range(1,n+1): res[k]=(j[k]==k-1) or (j[k] in K)
        K2=set(K); v2=dict(v)
        r=0
        for i in range(1,n+1):
            if res[i] and j[i]!=i-1 and v[j[i]]==1: r=i
            vr=1 if r==0 else 0
            ok=True
            for k in range(r+1,i+1):
                a_prev=vr^((k-1-r)&1)
                if a_prev==0 and not res[k]: ok=False;break
            if ok:
                K2.add(i); v2[i]=vr^((i-r)&1)
        assert K2>K, "pruned iteration stalled"
        K,v=K2,v2; hist.append(set(K))
    for i in range(n+1): assert v[i]==x[i]
    return hist
def jacobi_step(j,y):
    z=[1]
    for i in range(1,len(j)):
        r=z[i-1] if j[i]==i-1 else y[j[i]]
        z.append(1-(z[i-1]|r))
    return z
tot=0; worse=0; maxgap=0
for n in range(1,9):
    for tail in itertools.product(*[range(i) for i in range(1,n+1)]):
        j=[0]+list(tail); x=trace(j); hist=pruned_rounds(j,x)
        y=[1]+[0]*n; s=0
        while True:
            K=hist[min(s,len(hist)-1)]
            for i in K: assert y[i]==x[i], (j,s,i)
            if y==x: break
            y=jacobi_step(j,y); s+=1
        assert jacobi_step(j,x)==x
        assert s<=len(hist)-1
        maxgap=max(maxgap,len(hist)-1-s); tot+=1
print("arrays checked:",tot,"largest saving in rounds:",maxgap)
