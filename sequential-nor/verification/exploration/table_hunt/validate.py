# validate.py: independent re-computation (from the definitions) of the numbers hunt prints,
# on small instances dumped by hunt (HUNT_DUMP=file, one seed).  usage: python3 -I validate.py
import subprocess, os, itertools
def check(n,eps,R,pf,m,spec,seed):
    env=dict(os.environ,HUNT_DUMP='dump.txt')
    subprocess.run(['./hunt',str(n),str(eps),str(R),str(pf),str(m),'1',spec,'100000','1',str(seed)],env=env,capture_output=True,check=True)
    L=open('dump.txt').read().split(); r,mD,sv=map(int,L[:3]); j=[0]+list(map(int,L[3:]))
    assert len(j)==n+1 and all(0<=j[i]<i for i in range(1,n+1))
    x=[1]+[0]*n
    for i in range(1,n+1): x[i]=int(not(x[i-1] or x[j[i]]))
    # table-Jacobi from the definition
    y=[1]+[0]*n; s=0
    while y!=x:
        z=[1]+[0]*n
        for i in range(1,n+1):
            rd=z[j[i]] if i-j[i]<=R else y[j[i]]
            z[i]=int(not(z[i-1] or rd))
        y=z; s+=1
    assert s==r,(s,r)
    # set-valued from the definition, windows as tuples
    K={0}; rounds=0
    while len(K)<n+1:
        rounds+=1; S={tuple([0]*(R-1)+[1])}; K2={0}
        for i in range(1,n+1):
            T=set(); d=i-j[i]
            for sg in S:
                if d<=R: B=[sg[R-d]]
                elif j[i] in K: B=[x[j[i]]]
                else: B=[0,1]
                for b in B: T.add(sg[1:]+(int(not(sg[-1] or b)),))
            S=T
            if len({t[-1] for t in S})==1:
                assert next(iter(S))[-1]==x[i]; K2.add(i)
        assert K<=K2; K=K2
    assert rounds==sv,(rounds,sv)
    # relevance-pruned iteration from its definition (rounds until all known)
    known={0}; t=0
    while len(known)<n+1:
        t+=1; new=set(known)
        resolved=[False]*(n+1); rreset=[False]*(n+1)
        for k in range(1,n+1):
            if j[k]==k-1: resolved[k]=True
            elif j[k] in known: resolved[k]=True; rreset[k]=(x[j[k]]==1)
        for i in range(1,n+1):
            ri=0
            for k in range(i,0,-1):
                if rreset[k]: ri=k;break
            v=1 if ri==0 else 0; ok=True; a=v
            for k in range(ri+1,i+1):
                if a==0 and not resolved[k]: ok=False;break
                a=1-a
            if ok:
                assert a==x[i]; new.add(i)
        known=new
    assert t==mD,(t,mD)
    return r,sv,mD
c=0
for spec,m in [('A',20),('L2,E0',20),('L6,E0',24),('L3,L7,U2-9,H1,F500',0),('U1-12',0),('L5,Q4,L9',30),('S3,E1,Z,O',16)]:
    for eps in (0.1,0.3,0.46,0.7):
        for R in (1,2,3,5,8):
            for seed in (1,2):
                print(spec,eps,R,check(150,eps,R,0.3,m,spec,seed)); c+=1
print('all',c,'cases agree')
