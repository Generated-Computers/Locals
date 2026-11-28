# tracemap_test.py -- numerical test of the trace recursion (T1)-(T3) (not a proof).
# Ideal model: a long chain runs a periodic program with the letters E (far read of a class given
# by its phase index, written E0, E1, ...), N (type L_1) and L (type L_2).  The lower and upper
# values of a read of class c are 1 with probabilities lam[c] <= ups[c] (coupled by one uniform
# number); a read is resampled with probability eps to a bit that is 1 with probability varpi; a
# gate L keeps its pointer with probability 1-eps.  The traces follow the rules of the definition
# of the traces for pointers two back (the rule (l'',u'') at kept gates L).  The script compares
# the measured frequencies of l = 0 and u = 0 at every phase with the periodic solution of the
# trace recursion, and on the diagonal lam = ups with the density of the chain itself.
# usage: python3 -I tracemap_test.py seed number_of_programs gates_per_program
import sys, random
def varpi(e):
    lo,hi=0.0,1.0
    for _ in range(200):
        m=(lo+hi)/2
        if e*(1-m)**2-m>0: lo=m
        else: hi=m
    return lo
def solve(eps,prog,lam,ups,iters=3000):
    w=varpi(eps);A=eps*w;B=eps*(1-w);c=1-eps;T=len(prog)
    def q(g,v):
        t=prog[g%T]
        return B+c*(1-v[int(t[1:])]) if t[0]=='E' else B+c
    am=[1.0]*T; ap=[0.0]*T
    for it in range(iters):
        for g in range(T):
            t=prog[g]; t1=prog[(g-1)%T]
            if t!='L':
                am[g]=1-q(g,ups)*ap[g-1]; ap[g]=1-q(g,lam)*am[g-1]
            elif t1!='L':
                am[g]=1-(B*ap[g-1]+c*(1-q(g-1,lam))*ap[g-2]); ap[g]=1-(B*am[g-1]+c*(1-q(g-1,ups))*am[g-2])
            else:
                am[g]=1-(B*ap[g-1]+c*A*ap[g-2]+c*c*(1-am[g-3])); ap[g]=1-(B*am[g-1]+c*A*am[g-2]+c*c*(1-ap[g-3]))
    return am,ap
def simulate(eps,prog,lam,ups,n,rng):
    w=varpi(eps);T=len(prog)
    l=[0,0,0]; u=[1,1,1]; x=[0,0,0]; yl=[0,0,0]; yu=[0,0,0]; yx=[0,0,0]; kept=[0,0,0]
    cl=[0]*T; cu=[0]*T; cx=[0]*T; cnt=[0]*T
    for g in range(3,n):
        t=prog[g%T]; k=0
        if t=='L' and rng.random()>=eps: k=1
        if t=='L' and not k: b=int(rng.random()<w); rl=ru=rx=b
        elif t=='N' : 
            if rng.random()<eps: b=int(rng.random()<w); rl=ru=rx=b
            else: rl=ru=rx=0                      # effective read of a NOT gate: a determined zero
        elif t[0]=='E':
            if rng.random()<eps: b=int(rng.random()<w); rl=ru=rx=b
            else:
                c=int(t[1:]); z=rng.random(); rl=int(z<lam[c]); ru=int(z<ups[c]); rx=ru   # the chain itself reads the upper value
        else: rl=ru=rx=0
        if not k:
            nl=int(not(u[-1] or ru)); nu=int(not(l[-1] or rl)); nx=int(not(x[-1] or rx))
        elif kept[-1]:
            nl,nu,nx=l[-3],u[-3],x[-3]
        else:
            nl=(1-u[-2])&yl[-1]; nu=(1-l[-2])&yu[-1]; nx=(1-x[-2])&yx[-1]
        if t=='N' and k==0 and not(rl or ru): pass
        l=l[1:]+[nl]; u=u[1:]+[nu]; x=x[1:]+[nx]; yl=yl[1:]+[rl]; yu=yu[1:]+[ru]; yx=yx[1:]+[rx]; kept=kept[1:]+[k]
        assert nl<=nu
        if g>200:
            ph=g%T; cnt[ph]+=1; cl[ph]+=1-nl; cu[ph]+=1-nu; cx[ph]+=1-nx
    return [a/b for a,b in zip(cl,cnt)],[a/b for a,b in zip(cu,cnt)],[a/b for a,b in zip(cx,cnt)]
rng=random.Random(int(sys.argv[1])); P=int(sys.argv[2]); n=int(sys.argv[3]); worst=0; wd=0
for it in range(P):
    eps=rng.choice([0.3,0.455,0.5,0.7]); T=rng.randint(2,6)
    prog=[('L' if rng.random()<0.45 else 'N' if rng.random()<0.15 else 'E%d'%rng.randrange(T)) for _ in range(T)]
    if all(t=='L' for t in prog): prog[0]='E0'
    lam=[rng.random()*0.6 for _ in range(T)]; ups=[min(1.0,a+rng.random()*0.5) for a in lam]
    if it%3==0: ups=list(lam)
    am,ap=solve(eps,prog,lam,ups); sl,su,sx=simulate(eps,prog,lam,ups,n,rng)
    dev=max(max(abs(a-b) for a,b in zip(am,sl)),max(abs(a-b) for a,b in zip(ap,su)))
    worst=max(worst,dev)
    if it%3==0: wd=max(wd,max(abs(a-b) for a,b in zip(am,sx)))
    print(eps,','.join(prog),'largest deviation %.4f'%dev)
print('largest deviation between simulation and recursion: %.4f; on the diagonal, against the chain itself: %.4f; sampling error about %.4f'%(worst,wd,1.5/((n/6)**0.5)))
