# referee: Prop tracemap (d): sum |dPsi/dv| f(1-v) <= theta f(1-Psi); and (a) range, (c) monotonicity, (e) kappa=0
import sys, random, math
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from tm import *
def psi(e,prog,v,C):
    am,ap=solve(e,prog,v[:C],v[C:],iters=260)
    return am+ap   # a^-_g (g=0..T-1), a^+_g ; Psi = 1 - these
def ratio(e,prog,v,C,h=1e-6):
    T=len(prog); base=psi(e,prog,v,C); J=[[0.0]*(2*C) for _ in range(2*T)]
    for i in range(2*C):
        vp=list(v); vm=list(v); hh=h
        vp[i]=min(1.0,v[i]+h); vm[i]=max(0.0,v[i]-h)
        a=psi(e,prog,vp,C); b=psi(e,prog,vm,C)
        for k in range(2*T): J[k][i]=(a[k]-b[k])/(vp[i]-vm[i])
    best=0; 
    for k in range(2*T):
        num=sum(abs(J[k][i])*f(1-v[i]) for i in range(2*C)); den=f(base[k])
        best=max(best,num/den)
    return best,base
def main():
    rng=random.Random(int(sys.argv[1])); trials=int(sys.argv[2]); elist=[float(z) for z in sys.argv[3].split(',')]
    C=3; worst=0; wrange=0
    for tr in range(trials):
        e=rng.choice(elist); w,A,B,r,lo,hi=consts(e)
        T=rng.randint(1,6); prog=randprog(rng,T,C,pL=rng.choice([0.2,0.5,0.7]),pN=rng.choice([0,0.15]))
        v=[rng.random() for _ in range(2*C)]
        if rng.random()<0.3: v=[rng.choice([0.0,1.0,rng.random()]) for _ in range(2*C)]
        if rng.random()<0.3: v=v[:C]+v[:C]
        rt,base=ratio(e,prog,v,C)
        if not all(lo-1e-12<=a<=hi+1e-12 for a in base): wrange+=1; print('RANGE VIOLATION',e,prog,v,base,lo,hi)
        # hill climb on v
        step=0.2
        for it in range(60):
            i=rng.randrange(2*C); v2=list(v); v2[i]=min(1.0,max(0.0,v[i]+rng.uniform(-step,step)))
            r2,_=ratio(e,prog,v2,C)
            if r2>rt: rt=r2; v=v2
            else: step=max(0.01,step*0.97)
        if rt>worst: worst=rt; print('new worst %.6f'%rt,'eps',e,pstr(prog),prog,[round(z,4) for z in v]); sys.stdout.flush()
    print('worst ratio',worst,'range violations',wrange)
main()
