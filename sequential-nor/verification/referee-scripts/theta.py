import math
from fractions import Fraction

def varpi(eps):
    return ((2*eps+1) - math.sqrt(4*eps+1))/(2*eps)

def Tm(H, m, eps, c, beta):
    r = 1-eps*c; e0 = eps*(1-c)
    q = min(m, H-1)
    t0 = t1 = r**(m-q)
    for i in range(q-1, -1, -1):
        n1 = r*t0
        n0 = max(a*e0*t1 + (1-a)*r*t0 for a in (eps*beta, r))
        t0, t1 = n0, n1
    return t0, t1

def Theta(H, eps, c, beta, exact=True):
    r = 1-eps*c
    if exact:
        # T_m = r^{m-(H-1)} T_{H-1} for m >= H-1 (positive homogeneity)
        s = sum(max(Tm(H, m, eps, c, beta)) for m in range(H-1))
        s += max(Tm(H, H-1, eps, c, beta))/(1-r)
        return (1-eps)*s
    s = 0.0; m = 0
    while r**m > 1e-18:
        s += max(Tm(H, m, eps, c, beta)); m += 1
    return (1-eps)*s

print("three claims (c=varpi-1e-4, beta=0.4999):")
for H, eps, bound in ((3,0.74,0.985),(8,0.72,0.971),(16,0.71,0.994)):
    c = varpi(eps)-1e-4
    a = Theta(H, eps, c, 0.4999); b = Theta(H, eps, c, 0.4999, exact=False)
    print(H, eps, "varpi=%.6f"%varpi(eps), "Theta=%.6f (brute %.6f)"%(a,b), "claimed <=", bound, a <= bound)

print("table (c=varpi, beta=1/2):")
for H in (2,3,4,6,8,16,64):
    row = [Theta(H, e, varpi(e), 0.5) for e in (0.71,0.72,0.73,0.74)]
    lo, hi = 0.6, 0.8
    for _ in range(60):
        mid = (lo+hi)/2
        if Theta(H, mid, varpi(mid), 0.5) > 1: lo = mid
        else: hi = mid
    print(H, " ".join("%.4f"%v for v in row), "root %.4f"%lo)

# monotonicity in H of every T_m, and T_m <= r^m, homogeneity
bad = 0
for eps in (0.3,0.5,0.6,0.7,0.72,0.74,0.9):
    for c in (0.05, varpi(eps)-1e-4, 0.2):
        for beta in (0.0, 0.25, 0.4999):
            r = 1-eps*c
            for H in range(2, 40):
                for m in range(0, 80):
                    a = max(Tm(H, m, eps, c, beta)); b = max(Tm(H+1, m, eps, c, beta))
                    if b > a*(1+1e-12) or a > r**m*(1+1e-12): bad += 1
                    if m >= H-1:
                        hgen = r**(m-H+1)*max(Tm(H, H-1, eps, c, beta))
                        if abs(hgen-a) > 1e-12*max(a,1e-300): bad += 1
print("violations of T_m^{H+1}<=T_m^H, T_m<=r^m, homogeneity:", bad)
# which alpha attains the max? and is T_m = t_0(1) always?
for H,eps in ((3,0.74),(8,0.72),(16,0.71)):
    c = varpi(eps)-1e-4
    print(H, eps, [ (round(Tm(H,m,eps,c,0.4999)[0],5), round(Tm(H,m,eps,c,0.4999)[1],5)) for m in range(0,6)])
# H=2: T_m = r^m, Theta_2 = (1-eps)/(eps c)
for eps in (0.71,0.74,0.75):
    c=varpi(eps); print("H=2", eps, Theta(2,eps,c,0.5), (1-eps)/(eps*c))
