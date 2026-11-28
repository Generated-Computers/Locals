# Density of ones on the eps-smoothing of the fixed-distance base j_i = max(i-H,0):
# the bounds l_s <= psi <= mu_s of the note (l_{s+1} = Lam(mu_s), mu_{s+1} = U(l_{s+1}), mu_0 = 1/2)
# against simulation.  Standard library only.
import random
def shift_density(e, s=40):
    mu = 0.5; l = None
    for _ in range(s):
        l = (1-e*mu)/(3-e-e*mu); mu = (1-e*l)/(2-e*l)
    return l, mu          # l_s increases to l*, mu_s decreases to mu*
def simulate(n, H, eps, seed):
    rng = random.Random(seed); x = [1]*(n+1); ones = 1; lo = 1.0; hi = 0.0
    for i in range(1, n+1):
        j = rng.randrange(i) if rng.random() < eps else max(i-H, 0)
        x[i] = 1 - (x[i-1] | x[j]); ones += x[i]
        if i >= 20000:
            psi = ones/(i+1); lo = min(lo, psi); hi = max(hi, psi)
    return lo, hi
if __name__ == "__main__":
    for H in (2, 3, 8, 32):
        for eps in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            l, mu = shift_density(eps); lo, hi = simulate(400000, H, eps, 3)
            print("H=%2d eps=%.1f  bounds [%.4f, %.4f]  prefix densities for g >= 20000: [%.4f, %.4f]" % (H, eps, l, mu, lo, hi))
            assert l - 0.01 <= lo and hi <= mu + 0.01
    print("all densities inside the bounds")
