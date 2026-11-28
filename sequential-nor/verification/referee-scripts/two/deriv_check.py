# Lemma twochain(b): d alpha_h / d t_g by central differences vs. signed two-copy sum; bound by sum of p_g(h).
import random, math, sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from chain_exact import *
rng=random.Random(99); worst=0; worstb=-1
for trial in range(300):
    eps=rng.uniform(0.3,0.95); vp=((2*eps+1)-math.sqrt(4*eps+1))/(2*eps)
    T=rng.randint(1,7); gates=rand_design(rng,T,1.0,False)
    gates=[('M',0.1+0.8*g[1]) if g[0]=='M' else g for g in gates]
    ms=[i for i in range(T) if gates[i][0]=='M']
    if not ms: continue
    g=rng.choice(ms); L=60*T+600
    laws=stationary(gates,eps,vp,1.0)
    J=two_copies(gates,laws,g,L,eps,vp,1.0)
    def P1(t):
        gg=list(gates); gg[g]=('M',t); lw=stationary(gg,eps,vp,1.0)
        return [sum(p for (u,v),p in lw[i].items() if v==1) for i in range(T)]
    dlt=1e-5; t=gates[g][1]; up=P1(t+dlt); dn=P1(t-dlt)
    for h in range(T):
        sg=sum(J[k][3] for k in range(L+1) if (g+k)%T==h); ab=sum(J[k][0]+J[k][1] for k in range(L+1) if (g+k)%T==h)
        fd=(up[h]-dn[h])/(2*dlt)
        worst=max(worst,abs(fd-sg)); worstb=max(worstb,abs(sg)-ab)
print("max |finite difference - signed two-copy sum| =",worst,"; max(|J|-Mbar) =",worstb)
