# Offset-stack evaluator for Sequential NOR (my own code).
# Work memory = a stack of offsets o_s = P_{s-1} - k_s (self-delimiting code), plus O(1) index registers.
# Positions are never stored on the stack: they are recomputed from (q; o_1..o_s) by walking the input.
import random, math, sys
def trace(j):
    n=len(j)-1; x=[0]*(n+1); x[0]=1
    for i in range(1,n+1): x[i]=(1^x[i-1])&(1^x[j[i]])
    return x
def gamma_bits(o): return 2*int(math.log2(o+1))+1      # Elias-gamma length of o+1
def eval_pos(j,q):
    """returns (x_q, calls, max stack bits, max depth). Stack holds offsets only."""
    offs=[0]; calls=1; maxbits=0; maxdepth=1
    def positions():                    # recompute current positions from q and offsets (as the machine would)
        P=q; out=[P]
        for o in offs[:-1]:
            P=j[P-o]; out.append(P)
        return out
    Ps=positions()                      # cached copy, used only for speed; equality is asserted below
    ret=None
    while True:
        P=Ps[-1]; o=offs[-1]
        if ret is not None:             # a child returned the value of x at j[k]
            if ret==1:                  # k is a reset: x_k = 0, alternation gives x_P
                val=o&1; offs.pop(); Ps.pop()
                if not offs: return val,calls,maxbits,maxdepth
                ret=val; continue
            ret=None; offs[-1]=o=o+1    # not a reset: keep scanning downwards
        k=P-o
        if k==0:                        # reached position 0 (x_0 = 1) with no reset in (0,P]
            val=1^(P&1); offs.pop(); Ps.pop()
            if not offs: return val,calls,maxbits,maxdepth
            ret=val; continue
        if j[k]==k-1:                   # NOT gate: nothing to resolve
            offs[-1]=o+1; continue
        t=j[k]
        if t==0: ret=1; continue        # pointer to x_0 = 1
        offs.append(0); Ps.append(t); calls+=1
        b=sum(gamma_bits(v) for v in offs)
        if b>maxbits: maxbits=b
        if len(offs)>maxdepth: maxdepth=len(offs)
        if calls%200000==0: assert Ps==positions()
def smooth(base,eps,rng):
    return [0]+[rng.randrange(i) if rng.random()<eps else base[i] for i in range(1,len(base))]
rng=random.Random(int(sys.argv[1]) if len(sys.argv)>1 else 3)
print("%-22s %6s %5s | %9s %8s %7s %9s"%("instance","n","eps","calls(q=n)","maxbits","depth","bits/log2n"))
bases={"uniform":lambda n:[0]*(n+1),
       "j=i-2":lambda n:[0]+[max(0,i-2) for i in range(1,n+1)],
       "j=0":lambda n:[0]*(n+1),
       "j=i//2":lambda n:[0]+[i//2 for i in range(1,n+1)]}
for name,eps in (("uniform",1.0),("j=i-2",0.9),("j=0",0.9),("j=i//2",0.9),("j=i-2",0.8),("j=i//2",0.8)):
    for k in (8,10,12):
        n=2**k; worst=(0,0,0)
        for rep in range(3):
            j=smooth(bases[name](n),eps,rng); xs=trace(j)
            for q in [n, n-1, rng.randrange(n//2,n), rng.randrange(n//2,n)]:
                v,c,b,d=eval_pos(j,q); assert v==xs[q]
                worst=max(worst,(b,c,d))
            if k<=8:
                for q in range(1,n+1): assert eval_pos(j,q)[0]==xs[q]
        b,c,d=worst
        print("%-22s %6d %5.2f | %9d %8d %7d %9.2f"%(name,n,eps,c,b,d,b/k))
