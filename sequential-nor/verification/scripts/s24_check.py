# s24_check.py: exact check of admissible parameters for the embedding of Section "Above 4/9"
# (program S24, pairs of stages). Every number is an integer k standing for k/2^P; lower bounds
# are rounded down and upper bounds up. The noise eps is rational and varpi(eps) is enclosed
# between rationals. Coordinates are the transformed ones: t_c = d_c for even c, 1-d_c for odd c.
# usage: python3 -I s24_check.py EPS_NUM EPS_DEN ETA_NUM ETA_DEN GAM_NUM GAM_DEN H BETA_NUM BETA_DEN RAD J
# The boxes B0, B1 for stage J are placed around the ideal two-cycle (computed in floating point;
# that computation only chooses the boxes, the check itself is exact).
import sys, math
from fractions import Fraction as Fr
KAPPA=[22,9,22,9,22,9,22,9,22,9,None,3,8,9,8,9,8,9,8,11,8,11,8,9]   # None: position 1
T=24; LIST=[3,8,9,11,22]
assert all(k is None or k%2==ph%2 for ph,k in enumerate(KAPPA))
P=96; ONE=1<<P
def fl(x): return (x.numerator<<P)//x.denominator
def ce(x): return -((-(x.numerator<<P))//x.denominator)
def ideal_cycle(e,h=6,iters=20000):
    vp=((2*e+1)-math.sqrt(4*e+1))/(2*e)
    def Pc(d,c):
        o=(h-1)*T+c+1; p=0.0
        for r in range(1,o+1):
            k=KAPPA[(r-1)%T]; dd=0.0 if k is None else d[k]
            p=(1-p)*(1-((1-e)*dd+e*vp))
        return p
    def Phi_t(t):
        d={c:(t[c] if c%2==0 else 1-t[c]) for c in LIST}
        return {c:(Pc(d,c) if c%2==0 else 1-Pc(d,c)) for c in LIST}
    t={c:0.5 for c in LIST}
    for i in range(iters): t=Phi_t(t)
    a,b=t,Phi_t(t)
    return (a,b) if a[8]<b[8] else (b,a)
def main(a):
    eps=Fr(int(a[0]),int(a[1])); eta=Fr(int(a[2]),int(a[3])); gam=Fr(int(a[4]),int(a[5]))
    h=int(a[6]); beta=Fr(int(a[7]),int(a[8])); rad=float(a[9]); J=int(a[10])
    assert J%2==0 and beta.denominator&(beta.denominator-1)==0 and 0<beta<1
    x=4*eps+1; p,q=x.numerator,x.denominator; K=200
    s=math.isqrt(p*q*4**K); sq_lo=Fr(s,q*2**K); sq_hi=Fr(s+1,q*2**K)
    assert sq_lo**2<=x<=sq_hi**2
    vp_lo=((2*eps+1)-sq_hi)/(2*eps); vp_hi=((2*eps+1)-sq_lo)/(2*eps)
    psim=vp_lo-eta; psip=vp_hi+eta
    assert 0<eta<vp_lo and psip<1
    G=ce(gam); c1=1-eps
    def chain(D,c,lower):
        """bound on P^sigma_c(d) (sigma=- if lower): start and reads chosen by Lemma progchain."""
        o=(h-1)*T+c+1
        inc0=(o%2==0)
        lo=hi=((0 if inc0 else ONE) if lower else (ONE if inc0 else 0))
        for r in range(1,o+1):
            k=KAPPA[(r-1)%T]; Dk=0 if k is None else D[k]
            inc=((o-r)%2==1)
            psi=psim if (inc==lower) else psip
            v=c1*Fr(Dk,ONE)+eps*psi
            rdn,rup=fl(v),ce(v)
            lo,hi=((ONE-hi)*(ONE-rup))>>P, -((-(ONE-lo)*(ONE-rdn))>>P)
        return lo if lower else hi
    def env(t,lower):
        """lower (Psi^-) or upper (Psi^+) envelope at t, rounded outward; antitone in t."""
        D={c:(t[c] if c%2==0 else ONE-t[c]) for c in LIST}
        out={}
        for c in LIST:
            if c%2==0: v=chain(D,c,lower)+(-G if lower else G)
            else:      v=ONE-chain(D,c,not lower)+(-G if lower else G)
            assert 0<=v<=ONE
            out[c]=v
        return out
    aa,bb=ideal_cycle(float(eps))
    qz=lambda v:int(round(v*ONE))
    B0=({c:qz(aa[c]-rad) for c in LIST},{c:qz(aa[c]+rad) for c in LIST})
    B1=({c:qz(bb[c]-rad) for c in LIST},{c:qz(bb[c]+rad) for c in LIST})
    w=(1-beta)/2
    def M(A,B,lower):
        out={}
        for c in LIST:
            v=w*Fr(A[c]+B[c],ONE)+beta
            out[c]=fl(v) if lower else ce(v)
        return out
    L01=env(M(B0[1],B0[1],False),True)
    U01=env(M(B0[0],B1[0],True),False); U11=env(M(B1[0],B1[0],True),False)
    U00={c:max(U01[c],U11[c]) for c in LIST}
    TOP={c:ONE for c in LIST}; BOT={c:0 for c in LIST}
    I={1:(L01,TOP),0:(BOT,U00)}
    for k in range(1,J+1):
        I={v:(env(I[v][1],True),env(I[v][0],False)) for v in (0,1)}
    inside=lambda Iv,B: all(B[0][c]<=Iv[0][c] and Iv[1][c]<=B[1][c] for c in LIST)
    ok1=inside(I[1],B1); ok0=inside(I[0],B0); sep=I[1][0][8]>I[0][1][8]
    f=lambda v:" ".join(f"{c}:{v[c]/ONE:.6f}" for c in LIST)
    print(f"eps={eps} eta={eta} gamma={gam} h={h} beta={beta} J={J} (P={P} bits)")
    print(f"varpi in [{float(vp_lo):.15f}, {float(vp_hi):.15f}]")
    print("B0 (value 0, stage J):",f(B0[0]),"|",f(B0[1]))
    print("B1 (value 1, stage J):",f(B1[0]),"|",f(B1[1]))
    print("L_0^1 (gate, 00->1):  ",f(L01)); print("U_0^0 (gate, 01,11->0):",f(U00))
    print("[L_J^1,U_J^1]:",f(I[1][0]),"|",f(I[1][1]))
    print("[L_J^0,U_J^0]:",f(I[0][0]),"|",f(I[0][1]))
    print(f"(L_J^1)_8={I[1][0][8]/ONE:.6f}  (U_J^0)_8={I[0][1][8]/ONE:.6f}")
    print("inside B1:",ok1," inside B0:",ok0," (V):",sep)
    print("ADMISSIBLE" if (ok1 and ok0 and sep) else "NOT ADMISSIBLE")
if __name__=="__main__": main(sys.argv[1:])
