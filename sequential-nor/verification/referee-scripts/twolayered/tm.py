# referee: trace recursion (T1)-(T3): Monte Carlo of the X^-/X^+ chain of Step 1 (ideal thresholds) vs recursion,
# diagonal vs direct chain, Jacobian test of Prop tracemap (d), and (e) at kappa=0.
import sys, random, math
def varpi(e):
    lo,hi=0.0,1.0
    for _ in range(200):
        m=(lo+hi)/2
        if e*(1-m)**2-m>0: lo=m
        else: hi=m
    return lo
def consts(e):
    w=varpi(e); A=e*w; B=e*(1-w); r=1-A; hi=(1-B)/(1-B*r); lo=1-r*hi
    return w,A,B,r,lo,hi
def f(x):
    if x<=0 or x>=1: return 0.0
    return x**2.15*(1-x)**1.65
# gate: ('M',mu0,mu1,[(c,w),...]) | ('N',) | ('L',)
def yq(e,B,g,vs):  # y and q of gate g for sign vector vs (list over classes)
    if g[0]=='N': return 1.0, B+(1-e)
    y=g[2]+sum(w*(1-vs[c]) for c,w in g[3]); return y, B+(1-e)*y
def solve(e,prog,vm,vp,iters=600):
    w,A,B,r,lo,hi=consts(e); T=len(prog); c=1-e
    qm=[None]*T; qp=[None]*T
    for g in range(T):
        if prog[g][0]!='L': qm[g]=yq(e,B,prog[g],vm)[1]; qp[g]=yq(e,B,prog[g],vp)[1]
    am=[1.0]*T; ap=[0.0]*T
    for it in range(iters):
        for g in range(T):
            t=prog[g][0]; t1=prog[(g-1)%T][0]
            if t!='L':
                am[g]=1-qp[g]*ap[g-1]; ap[g]=1-qm[g]*am[g-1]   # NOTE python negative index = periodic
            elif t1!='L':
                am[g]=1-(B*ap[g-1]+c*(1-qm[g-1])*ap[g-2]); ap[g]=1-(B*am[g-1]+c*(1-qp[g-1])*am[g-2])
            else:
                am[g]=1-(B*ap[g-1]+c*A*ap[g-2]+c*c*(1-am[g-3])); ap[g]=1-(B*am[g-1]+c*A*am[g-2]+c*c*(1-ap[g-3]))
    return am,ap
def solve_safe(e,prog,vm,vp):
    T=len(prog)
    # make period >=3 to avoid aliasing of g-1,g-2,g-3 in in-place sweeps (in-place Gauss-Seidel still converges to the same fixed point)
    return solve(e,prog,vm,vp)
def mc(e,prog,vm,vp,ngates,rng,direct=False):
    """X^-,X^+ chain of Step 1 with ideal thresholds; also the true chain x of Def two when direct (reads prob vp)."""
    w,A,B,r,lo,hi=consts(e); T=len(prog)
    Xm=[0,0,0,0]; Xp=[1,1,1,1]; ym=[0]*4; yp=[1]*4; kept=[0]*4; x=[0,0,0,0]
    cm=[0]*T; cp=[0]*T; cx=[0]*T; cnt=[0]*T; viol=0
    for g in range(4,ngates):
        t=prog[g%T]; xi=rng.random(); k=0
        if t[0]=='L':
            k = xi>=e; a=b=int(xi<e*w)
        elif t[0]=='N':
            a=b=int(xi<e*w)
        else:
            a=int(xi<(1-e)*(t[1]+sum(wt*vm[c] for c,wt in t[3]))+e*w)
            b=int(xi<(1-e)*(t[1]+sum(wt*vp[c] for c,wt in t[3]))+e*w)
        if not k:
            nm=int(not(Xp[-1] or b)); np_=int(not(Xm[-1] or a))
        elif not kept[-1]:
            nm=(1-Xp[-2])*ym[-1]; np_=(1-Xm[-2])*yp[-1]
        else:
            nm=Xm[-3]; np_=Xp[-3]
        if direct:
            # true chain: read value = b (upper) ; L kept reads x_{g-2}; N kept reads x_{g-1}
            if t[0]=='L' and k: rd=x[-2]
            elif t[0]=='N' and xi>=e: rd=x[-1]
            else: rd=b
            nx=int(not(x[-1] or rd)); x=x[1:]+[nx]
        if nm>np_: viol+=1
        Xm=Xm[1:]+[nm]; Xp=Xp[1:]+[np_]; ym=ym[1:]+[a]; yp=yp[1:]+[b]; kept=kept[1:]+[k]
        if g>300:
            ph=g%T; cnt[ph]+=1; cm[ph]+=1-nm; cp[ph]+=1-np_
            if direct: cx[ph]+=1-nx
    return [a/b for a,b in zip(cm,cnt)],[a/b for a,b in zip(cp,cnt)],[a/b for a,b in zip(cx,cnt)],viol
def randprog(rng,T,C,pL=0.4,pN=0.15):
    prog=[]
    for _ in range(T):
        u=rng.random()
        if u<pL: prog.append(('L',))
        elif u<pL+pN: prog.append(('N',))
        else:
            k=rng.randint(1,3); ws=[rng.random() for _ in range(k+2)]
            if rng.random()<0.5: ws[0]=0
            if rng.random()<0.5: ws[1]=0
            if rng.random()<0.3: ws=[0,0]+[1]+[0]*(k-1)
            s=sum(ws); ws=[z/s for z in ws]
            prog.append(('M',ws[0],ws[1],[(rng.randrange(C),ws[2+i]) for i in range(k)]))
    if all(g[0]=='L' for g in prog): prog[0]=('M',0,0,[(0,1.0)])
    return prog
def pstr(prog): return ''.join(g[0] for g in prog)
if __name__=='__main__':
    mode=sys.argv[1]; rng=random.Random(int(sys.argv[2]))
    if mode=='mc':
        n=int(sys.argv[3]); worst=0; worstd=0
        fixed=[ [('M',0,0,[(0,1.0)]),('L',)], [('M',0,0,[(0,1.0)]),('L',),('L',)], [('N',),('L',),('M',0.1,0.2,[(0,0.3),(1,0.4)])],
                [('M',0,0,[(0,1.0)]),('N',),('L',),('L',),('L',)], [('N',),('L',)], [('M',0,0,[(1,1.0)]),('L',),('L',),('L',),('L',),('L',),('L',)],
                [('L',),('N',),('L',),('M',0,0.5,[(0,0.5)]),('L',),('L',)] ]
        progs=fixed+[randprog(rng,rng.randint(2,7),3) for _ in range(14)]
        for it,prog in enumerate(progs):
            for diag in (0,1):
                e=rng.choice([0.3,0.455,0.46,0.5,0.7,0.9])
                vm=[rng.random()*0.7 for _ in range(3)]; vp=[min(1,a+rng.random()*0.6) for a in vm]
                if diag: vp=list(vm)
                am,ap=solve(e,prog,vm,vp); sm,sp,sx,viol=mc(e,prog,vm,vp,n,rng,direct=bool(diag))
                dev=max(max(abs(a-b) for a,b in zip(am,sm)),max(abs(a-b) for a,b in zip(ap,sp)))
                worst=max(worst,dev)
                dd=max(abs(a-b) for a,b in zip(am,sx)) if diag else 0; worstd=max(worstd,dd)
                print(e,pstr(prog),'diag' if diag else 'off ','dev %.4f'%dev,'direct-chain dev %.4f'%dd,'X-<=X+ violations',viol, 'am-ap diag gap %.1e'%(max(abs(a-b) for a,b in zip(am,ap)) if diag else 0))
        print('worst dev',worst,'worst diag',worstd,'sampling sd about',0.5/math.sqrt(n/7))
