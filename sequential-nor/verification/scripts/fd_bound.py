# Growth factor of the chain count for the fixed-distance base j_i = i-L, when a window gate that
# keeps its base pointer is charged for reading a zero at a fresh position above the chain's target.
import math
def varpi(e):   # smaller root of w = e(1-w)^2
    return ((2*e+1) - math.sqrt(4*e+1))/(2*e)
def theta(e, L, c=None, mu=0.5, M=4000):
    if c is None: c = varpi(e)
    s = 1-e; r = 1-e*c; a_lo = e*(1-mu); a_hi = 1-e*c; ez = e*(1-c)
    best = []
    for v in (0,1):
        tot = 0.0
        # T^{(m)}: weight of m window gates above k; offsets 1..min(m,L-1) are charged by the DP
        for m in range(0, M):
            q = min(m, L-1)
            T0 = T1 = r**(m-q)            # gates beyond offset L-1: factor r each
            for i in range(q-1, -1, -1):  # T_i from T_{i+1}
                n1 = r*T0
                cand = [a*ez*T1 + (1-a)*r*T0 for a in (a_lo, a_hi)]
                T0, T1 = max(cand), n1
            tot += (T0 if v==0 else T1)
            if r**(m-q) < 1e-16 and m>q: break
        best.append(s*tot)
    return best
if __name__ == "__main__":
  for L in (2,3,4,6,8,16,64):
    lo, hi = 0.5, 0.75
    for _ in range(50):
        mid=(lo+hi)/2
        if max(theta(mid,L))<1: hi=mid
        else: lo=mid
    print("L=%2d threshold %.4f   Theta at 0.71: %.4f  0.72: %.4f  0.73: %.4f  0.74: %.4f" % (L,hi,max(theta(0.71,L)),max(theta(0.72,L)),max(theta(0.73,L)),max(theta(0.74,L))))
  # the three claims of the theorem, with c and beta strictly inside their ranges
  for (L, e, stated) in ((3, 0.74, 0.985), (8, 0.72, 0.972), (16, 0.71, 0.994)):
      val = max(theta(e, L, c=varpi(e)-1e-4, mu=1-0.4999))
      print("claim H>=%d at eps=%.2f: Theta = %.6f (stated bound %.3f)" % (L, e, val, stated)); assert val <= stated
  # Theta_H is nonincreasing in H (spot check of the induction in the note)
  for e in (0.6, 0.7, 0.74):
      vals = [max(theta(e, L)) for L in range(2, 30)]
      assert all(vals[i+1] <= vals[i] + 1e-12 for i in range(len(vals)-1))
      assert vals[0] <= (1-e)/(e*varpi(e)) + 1e-9
  print("monotone in H: ok")
  # the same factor with the density bounds proved for fixed-distance bases:
  # c < l*(eps) and beta < 1 - mu*(eps); l_s < l* and mu_s > mu* for every s
  def shift_density(e, s=40):
      mu = 0.5; l = None
      for _ in range(s):
          l = (1-e*mu)/(3-e-e*mu); mu = (1-e*l)/(2-e*l)
      return l, mu
  def theta_fd(e, L, margin=0.0):
      l, mu = shift_density(e)
      return max(theta(e, L, c=l-margin, mu=mu+margin))
  for L in (2,3,4,6,8,16,64):
      lo, hi = 0.5, 0.75
      for _ in range(50):
          mid=(lo+hi)/2
          if theta_fd(mid,L)<1: hi=mid
          else: lo=mid
      print("fixed-distance density, H=%2d: threshold %.4f   Theta at 0.69: %.4f  0.70: %.4f  0.71: %.4f  0.72: %.4f  0.73: %.4f" % ((L,hi)+tuple(theta_fd(e,L) for e in (0.69,0.70,0.71,0.72,0.73))))
  for (L, e, stated) in ((3, 0.73, 0.964), (8, 0.70, 0.977), (16, 0.69, 0.995)):
      val = theta_fd(e, L, margin=1e-4)
      print("claim (fixed-distance density) H>=%d at eps=%.2f: Theta = %.6f (stated bound %.3f)" % (L, e, val, stated)); assert val <= stated
