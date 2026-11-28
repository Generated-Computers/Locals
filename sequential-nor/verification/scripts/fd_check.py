# Sanity check of the one-step bound for fixed-distance bases j_i = max(i-L,0):
# empirical mean number of gates k <= P that keep their base pointer with no reset in (k,P],
# split by the value x_{k-L} that the chain would land on, against Theta_L from fd_bound.py.
import random, sys
sys.path.insert(0, '.')
from fd_bound import theta, varpi
def run(n, L, eps, seed):
    rng = random.Random(seed)
    x = [1]*(n+1); base = [False]*(n+1); reset = [False]*(n+1)
    for i in range(1, n+1):
        if rng.random() < eps: j = rng.randrange(i)
        else: j = max(i-L, 0); base[i] = True
        x[i] = 1 - (x[i-1] | x[j]); reset[i] = (j != i-1 and x[j] == 1)
    # for each k (base), the number of P >= k with no reset in (k,P] = run length above k
    nxt = n+1; tot = [0.0, 0.0]; cnt = [0, 0]; ones = sum(x)/(n+1)
    for k in range(n, n//2, -1):
        if base[k]:
            v = x[k-L]; tot[v] += nxt - k; cnt[v] += 1
        if reset[k]: nxt = k
    # mean over P of #{k base, no reset in (k,P]} = sum_k (run length above k) / #positions
    N = n - n//2
    return ones, (tot[0]+tot[1])/N, tot[0]/max(cnt[0],1)*(1-eps), tot[1]/max(cnt[1],1)*(1-eps)
if __name__ == "__main__":
    for L in (3, 8):
        for eps in (0.60, 0.70, 0.74):
            ones, mean, c0, c1 = run(1000000, L, eps, 5)
            th = theta(eps, L)
            print("L=%d eps=%.2f ones=%.3f (varpi %.3f)  measured: all %.3f, landing on 0: %.3f, on 1: %.3f   bound: %.3f / %.3f"
                  % (L, eps, ones, varpi(eps), mean, c0, c1, th[0], th[1]))
