# referee: Prop tracemap (e) at kappa=0 and the remark after it: w (U-recursion) == sum_g s_g p_g(h) of Def two (a,b,c recursion); w <= theta f(alpha)
import sys, random, math
import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from tm import *
def chain_alpha(e,prog,d,iters=500):
    w,A,B,r,lo,hi=consts(e); T=len(prog); al=[0.5]*T
    q=[None if g[0]=='L' else yq(e,B,g,d)[1] for g in prog]
    for it in range(iters):
        for g in range(T):
            if prog[g][0]!='L': al[g]=1-q[g]*al[g-1]
            else: al[g]=(2-e)-r*al[g-1]-(1-e)*al[(g-2)%T]
    return al,q
def U(e,prog,d,ec,kappa=0.0,iters=800):
    w_,A,B,r,lo,hi=consts(e); T=len(prog); al,q=chain_alpha(e,prog,d); c=1-e
    src=[sum(wt*ec[cl] for cl,wt in g[3]) if g[0]=='M' else 0.0 for g in prog]
    w=[0.0]*T
    for it in range(iters):
        for g in range(T):
            t=prog[g][0]; t1=prog[g-1][0]
            if t!='L': w[g]=(q[g]+kappa)*w[g-1]+kappa*(w[(g-2)%T]+w[(g-3)%T])+c*(al[g-1]+kappa)*src[g]
            elif t1!='L': w[g]=(B+kappa)*w[g-1]+(c*(1-q[g-1])+kappa)*w[(g-2)%T]+kappa*w[(g-3)%T]+c*(c+kappa)*(al[(g-2)%T]+kappa)*src[(g-1)%T]
            else: w[g]=(B+kappa)*w[g-1]+(c*A+kappa)*w[(g-2)%T]+(c*c+kappa)*w[(g-3)%T]
    return w,al
def abc(e,prog,d,ec,iters=800):
    # N^a,N^b,N^c of thm:two via lem:twochain(c), periodic
    w_,A,B,r,lo,hi=consts(e); T=len(prog); al,q=chain_alpha(e,prog,d); c=1-e
    src=[sum(wt*ec[cl] for cl,wt in g[3]) if g[0]=='M' else 0.0 for g in prog]
    a=[0.0]*T;b=[0.0]*T;cc=[0.0]*T
    for it in range(iters):
        for g in range(T):
            if prog[g][0]!='L':
                p=a[g-1]+b[g-1]; a[g]=c*al[g-1]*src[g]; b[g]=q[g]*p; cc[g]=(1-q[g])*p
            else:
                a_,b_,c_=a[g-1],b[g-1],cc[g-1]; a[g]=c*c_; b[g]=r*a_+B*b_; cc[g]=A*a_+(1-B)*b_
    return [x+y for x,y in zip(a,b)]
rng=random.Random(int(sys.argv[1])); worst=0; wdiff=0; worstk=0
for tr in range(int(sys.argv[2])):
    e=rng.choice([0.455,0.455,0.46,0.5,0.7,0.9]); C=3; T=rng.randint(1,8)
    prog=randprog(rng,T,C,pL=rng.choice([0.2,0.5,0.7]),pN=rng.choice([0,0.15]))
    d=[rng.random() for _ in range(C)]
    if rng.random()<0.5:
        lo,hi=consts(e)[4:6]; d=[1-(lo+(hi-lo)*rng.random()) for _ in range(C)]
    ec=[f(1-z) for z in d]
    w,al=U(e,prog,d,ec); p=abc(e,prog,d,ec)
    wdiff=max(wdiff,max(abs(x-y) for x,y in zip(w,p)))
    rt=max(w[g]/f(al[g]) for g in range(T)); 
    if rt>worst: worst=rt; print('worst ratio %.6f'%rt,e,pstr(prog))
    A=consts(e)[1]; wk,_=U(e,prog,d,ec,kappa=1e-4); worstk=max(worstk,max((wk[g]-w[g])/1e-4 for g in range(T)))
print('max |w_U - N(Def two)| =',wdiff,' worst ratio w/f(alpha) =',worst,' max dw/dkappa ~',worstk)
