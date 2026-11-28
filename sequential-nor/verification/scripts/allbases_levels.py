# allbases_levels.py: the noise level of the section "All bases: the relevance-pruned iteration below 3/4".
#   varpi = root in (0,1/2) of varpi = eps (1-varpi)^2,  r = 1 - eps*varpi,  eta2 = eps^2 varpi (1-varpi)
#   chi(eps)    = (1-eps)(1-eta2) - eps*varpi          (decreasing; its root is eps_1)
#   kappa2(eps) = r^2 + (1-eps)(1 - eps/2 + r)         (decreasing)
# The script encloses varpi(eps) between rationals and checks, in exact rational arithmetic,
#   chi(0.7287) > 0 > chi(0.7288)   and   kappa2(0.7287) < 0.96.
# It also prints floating-point values: eps_1, kappa2 there, and the level 0.7394 of the remark
# (the root of (r+(1-eps)(1-eta2))(r+1-eps) = 1), which the same proof gives without the inequality
# for pointers of reach 2.
import math
from fractions import Fraction as Fr
def varpi_enc(e):
    x=4*e+1; p,q=x.numerator,x.denominator; K=120
    s=math.isqrt(p*q*4**K); lo=Fr(s,q*2**K); hi=Fr(s+1,q*2**K)     # lo <= sqrt(4e+1) <= hi
    assert lo*lo<=x<=hi*hi
    return ((2*e+1)-hi)/(2*e), ((2*e+1)-lo)/(2*e)
def enc(e):
    wl,wh=varpi_enc(e)                               # varpi in [wl,wh], inside (0,1/2)
    assert 0<wl<=wh<Fr(1,2)
    r_lo,r_hi=1-e*wh,1-e*wl
    eta_lo,eta_hi=e*e*wl*(1-wl),e*e*wh*(1-wh)        # w(1-w) increases on (0,1/2)
    chi_lo=(1-e)*(1-eta_hi)-e*wh; chi_hi=(1-e)*(1-eta_lo)-e*wl
    k_lo=r_lo*r_lo+(1-e)*(1-e/2+r_lo); k_hi=r_hi*r_hi+(1-e)*(1-e/2+r_hi)
    old_lo=(r_lo+(1-e)*(1-eta_hi))*(r_lo+1-e)-1; old_hi=(r_hi+(1-e)*(1-eta_lo))*(r_hi+1-e)-1
    return chi_lo,chi_hi,k_lo,k_hi,old_lo,old_hi
a=enc(Fr(7287,10000)); b=enc(Fr(7288,10000))
assert a[0]>0 and b[1]<0, "chi"
assert a[3]<Fr(96,100), "kappa2"
print("chi(0.7287) in [%.3e, %.3e] > 0,  chi(0.7288) in [%.3e, %.3e] < 0"%(a[0],a[1],b[0],b[1]))
print("kappa2(0.7287) in [%.6f, %.6f] < 0.96"%(a[2],a[3]))
c=enc(Fr(7393,10000)); d=enc(Fr(7394,10000))
assert c[4]>0 and d[5]<0, "old level"
print("without the reach-2 inequality: level between 0.7393 and 0.7394 (exact)")
def vp(e): return ((2*e+1)-math.sqrt(4*e+1))/(2*e)
def root(f,lo,hi):
    for _ in range(80):
        m=(lo+hi)/2
        if f(lo)*f(m)<=0: hi=m
        else: lo=m
    return (lo+hi)/2
def chi(e):
    w=vp(e); return (1-e)*(1-e*e*w*(1-w))-e*w
def kappa2(e):
    w=vp(e); r=1-e*w; return r*r+(1-e)*(1-e/2+r)
e1=root(chi,0.6,0.8)
print("floating point: eps_1 = %.6f, varpi = %.5f, eta2 = %.5f, kappa2(eps_1) = %.5f"%(e1,vp(e1),e1*e1*vp(e1)*(1-vp(e1)),kappa2(e1)))
print("floating point: kappa2 = 1 at eps = %.4f"%root(lambda e:kappa2(e)-1,0.5,0.8))
