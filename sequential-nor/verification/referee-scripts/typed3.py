import sys
from fractions import Fraction as F
sys.path.insert(0, sys.argv[1])
import exact_typed as et
# length-3 typed skeletons, and report the tightest case that has at least one base step with a nonempty window
for (H, eps, g0, beta, c, n) in ((3, F(74,100), 2, F(1,4), F(1,10), 13), (2, F(1,2), 2, F(1,4), F(1,5), 12), (4, F(6,10), 3, F(1,3), F(1,4), 14)):
    w, cnt, wsk = et.run(H, eps, c, beta, g0, n, 3)
    print("typed l<=3 ok H=%d eps=%s g0=%d c=%s n=%d cases=%d" % (H, eps, g0, c, n, cnt), flush=True)
