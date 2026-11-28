import sys, random
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from tm import *
def psi(e,prog,v,C):
    am,ap=solve(e,prog,v[:C],v[C:],iters=220); return am+ap
def ratios(e,prog,v,C,h=1e-6):
    T=len(prog); base=psi(e,prog,v,C); J=[[0.0]*(2*C) for _ in range(2*T)]
    for i in range(2*C):
        vp=list(v); vm=list(v); vp[i]=min(1.0,v[i]+h); vm[i]=max(0.0,v[i]-h)
        a=psi(e,prog,vp,C); b=psi(e,prog,vm,C)
        for k in range(2*T): J[k][i]=(a[k]-b[k])/(vp[i]-vm[i])
    return [sum(abs(J[k][i])*f(1-v[i]) for i in range(2*C))/f(base[k]) for k in range(2*T)]
rng=random.Random(int(sys.argv[1])); C=2; worstL=0; e=float(sys.argv[3])
for tr in range(int(sys.argv[2])):
    T=rng.randint(3,6); prog=randprog(rng,T,C,pL=0.5,pN=0.1)
    Lrows=[g for g in range(T) if prog[g][0]=='L']
    if not Lrows: continue
    v=[rng.random() for _ in range(2*C)]
    def val(v):
        r=ratios(e,prog,v,C); return max(max(r[g],r[g+T]) for g in Lrows)
    cur=val(v); step=0.3
    for it in range(80):
        i=rng.randrange(2*C); v2=list(v); v2[i]=min(1.0,max(0.0,v[i]+rng.uniform(-step,step)))
        if rng.random()<0.1: v2[i]=rng.choice([0.0,1.0])
        c2=val(v2)
        if c2>cur: cur=c2; v=v2
        else: step=max(0.01,step*0.97)
    if cur>worstL: worstL=cur; print('L-row worst %.5f'%cur,pstr(prog),[round(z,3) for z in v]); sys.stdout.flush()
print('worst ratio at rows of type L2:',worstL)
