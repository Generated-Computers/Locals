# allbases_exact.py: checks for the section "All bases: the relevance-pruned iteration below 3/4".
#
# Part 1 (Lemma "one gate, two kinds of open positions"): parts (a), (b), (c) on ALL parent arrays
#   with 2 <= n <= NMAX.
#
# Part 2: the inequalities of the proof, in the form in which the densities are those of the
#   instance, by summing over ALL outcomes of the noise, for random bases of length n (floating
#   point sums; a violation is a difference above 1e-12). With O_g the number of ones among
#   x_0..x_g, "open" for D > s, and all probabilities over the smoothing:
#   (A)  P[D(k)>s+1, x_k=1] <= E[1(D(k-1)>s+1, x_{k-1}=0)(1 - eps O_{k-1}/k)]
#          + (1-eps) 1[d_k>=2] E[1(x_j=0, D(j)>s) g_k] + (eps/k) sum_{p<=k-2} P[x_p=0, D(p)>s]
#   (B)  P[D(k)>s+1, x_k=0] <= E[1(D(k-1)>s+1, x_{k-1}=1)(1 - eps (O_{k-1}-1)/k)]
#          + (1-eps) 1[d_k>=2] E[1(x_j=1, D(j)>s) g'_k] + (2 eps/k) sum_{p<=k-2} P[x_p=1, D(p)>s]
#        g_k = g'_k = 1 - eps^2 (O_{k-3}/(k-2)) ((k-1-O_{k-3})/(k-1)) for d_k >= 3;
#        for d_k = 2: g_k = 1 - eps (k-1-O_{k-2})/(k-1) and g'_k = 1.
#   For a gate k with base pointer k-2, with Z = P[x_{k-2}=0, x_{k-1}=0, D(k-1)>s+1] and
#   Y = P[x_{k-2}=1, D(k-1)>s+1]:
#   (S)  P[D(k)>s+1] <= (1-eps) P[x_{k-2}=1, D(k-2)>s]
#          + (1-eps) E[1(x_{k-2}=0, D(k-2)>s)(1 - eps (k-1-O_{k-2})/(k-1))]
#          + (1-eps)(P[D(k-1)>s+1, x_{k-1}=1] + Z) + E[1(D(k-1)>s+1) eps (k-O_{k-2})/k]
#          + (eps/k) sum_{p<=k-2} P[D(p)>s]
#   (Z)  Z <= (1-eps) 1[d_{k-1}>=2] P[x_{j_{k-1}}=1, D(j_{k-1})>s] + (eps/(k-1)) sum_{p<=k-3} P[x_p=1, D(p)>s]
#   (Y)  Y <= E[1(x_{k-2}=1, D(k-2)>s+1)(1 - eps (O_{k-2}-1)/(k-1))] + (eps/(k-1)) sum_{p<=k-3} P[x_p=1, D(p)>s]
#   The constants c, beta, eta of the proof bound these densities from one side, so the
#   inequalities of the paper follow from these with G* the sure event.
#
# usage: python3 -I allbases_exact.py NMAX n SEEDS      (the paper: 7 7 30)
import itertools, random, sys
def trace(ptr,n):
    x=[1]+[0]*n; D=[0]*(n+1)
    for q in range(1,n+1):
        p=ptr[q]; x[q]=int(not(x[q-1] or x[p])); notg=(p==q-1); reset=(not notg) and x[p]==1; rel=(x[q-1]==0)
        if q==1: D[q]=1
        elif reset and rel: D[q]=1+D[p]
        elif reset: D[q]=min(D[q-1],1+D[p])
        elif rel and not notg: D[q]=max(D[q-1],1+D[p])
        else: D[q]=D[q-1]
    return x,D
def lemma(nmax):
    cnt=0
    for n in range(2,nmax+1):
        for tail in itertools.product(*[range(i) for i in range(1,n+1)]):
            ptr=(0,)+tail; x,D=trace(ptr,n); cnt+=1
            for k in range(2,n+1):
                p=ptr[k]; notg=(p==k-1)
                assert D[k]<=k, ("c",ptr,k)
                for s in range(0,n):
                    if D[k]<=s+1: continue
                    if x[k]==1:
                        assert x[k-1]==0 and x[p]==0 and (D[k-1]>s+1 or ((not notg) and D[p]>s)), ("a",ptr,k,s)
                    else:
                        ok1=(x[k-1]==0 and (not notg) and D[p]>s and x[p]==1)
                        ok2=(x[k-1]==1 and D[k-1]>s+1 and (notg or x[p]==0 or (D[p]>s and x[p]==1)))
                        assert ok1 or ok2, ("b",ptr,k,s)
    return cnt
def run(n,eps,seed,mix):
    rnd=random.Random(seed)
    j=[0]+[rnd.randrange(i) for i in range(1,n+1)]
    for i in range(3,n+1):                       # bias the base toward reach 2 and 3
        u=rnd.random()
        if u<mix[0]: j[i]=i-2
        elif u<mix[1]: j[i]=i-3
    out=[]
    def rec(k,ptr,w):
        if k>n:
            x,D=trace(ptr,n); O=[0]*(n+1); c=0
            for q in range(n+1): c+=x[q]; O[q]=c
            out.append((w,x,D,O)); return
        ptr[k]=j[k]; rec(k+1,ptr,w*(1-eps))
        for r in range(k):
            ptr[k]=r; rec(k+1,ptr,w*eps/k)
    rec(1,[0]*(n+1),1.0)
    worst=0.0
    for s in range(0,n):
        for k in range(5,n+1):
            d=k-j[k]; jk1=j[k-1]; dk1=(k-1)-jk1
            La=Ra=Lb=Rb=L=S1=Z=ZR=Y=YR=0.0
            for w,x,D,O in out:
                o1=D[k]>s+1; c1=D[k-1]>s+1
                n0=sum(1 for p in range(0,k-1) if x[p]==0 and D[p]>s)
                n1=sum(1 for p in range(0,k-1) if x[p]==1 and D[p]>s)
                if o1 and x[k]==1: La+=w
                if o1 and x[k]==0: Lb+=w
                if c1 and x[k-1]==0: Ra+=w*(1-eps*O[k-1]/k)
                if c1 and x[k-1]==1: Rb+=w*(1-eps*(O[k-1]-1)/k)
                if d>=3 and D[j[k]]>s:
                    g=1-eps*eps*(O[k-3]/(k-2))*((k-1-O[k-3])/(k-1))
                    if x[j[k]]==0: Ra+=w*(1-eps)*g
                    else: Rb+=w*(1-eps)*g
                if d==2 and D[k-2]>s:
                    if x[k-2]==0: Ra+=w*(1-eps)*(1-eps*(k-1-O[k-2])/(k-1))
                    else: Rb+=w*(1-eps)
                Ra+=w*(eps/k)*n0; Rb+=w*(2*eps/k)*n1
                if d==2:
                    m1=sum(1 for p in range(0,k-2) if x[p]==1 and D[p]>s)
                    if o1: L+=w
                    if x[k-2]==1 and D[k-2]>s: S1+=w*(1-eps)
                    if x[k-2]==0 and D[k-2]>s: S1+=w*(1-eps)*(1-eps*(k-1-O[k-2])/(k-1))
                    if c1 and (x[k-1]==1 or x[k-2]==0): S1+=w*(1-eps)
                    if c1: S1+=w*eps*(k-O[k-2])/k
                    S1+=w*(eps/k)*(n0+n1)
                    if x[k-2]==0 and x[k-1]==0 and c1: Z+=w
                    if dk1>=2 and x[jk1]==1 and D[jk1]>s: ZR+=w*(1-eps)
                    ZR+=w*(eps/(k-1))*m1
                    if x[k-2]==1 and c1: Y+=w
                    if x[k-2]==1 and D[k-2]>s+1: YR+=w*(1-eps*(O[k-2]-1)/(k-1))
                    YR+=w*(eps/(k-1))*m1
            worst=max(worst,La-Ra,Lb-Rb)
            if d==2: worst=max(worst,L-S1,Z-ZR,Y-YR)
    return worst,j
if __name__=="__main__":
    nmax=int(sys.argv[1]); n=int(sys.argv[2]); seeds=int(sys.argv[3])
    print("Lemma (a),(b),(c): %d parent arrays with 2 <= n <= %d, no exception"%(lemma(nmax),nmax),flush=True)
    tot=0; mx=0.0; bad=0
    for mix in ((0.35,0.6),(0.5,0.65)):
        for eps in (0.3,0.5,0.7,0.73):
            for seed in range(seeds):
                w,j=run(n,eps,seed,mix); tot+=1; mx=max(mx,w)
                if w>1e-12: bad+=1; print("VIOLATION n=%d eps=%s seed=%d base=%s max(LHS-RHS)=%.3e"%(n,eps,seed,j[1:],w),flush=True)
    print("(A),(B),(S),(Z),(Y): n=%d, %d (base, eps) pairs, %d violations, largest LHS-RHS = %.2e"%(n,tot,bad,mx))
