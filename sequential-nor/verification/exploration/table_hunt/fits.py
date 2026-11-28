# fits.py: for the long fixed-reach runs, fit TJ rounds T(n) by  a*log2(n)+b,  (a*(log2 n - c))^2,  a*n^alpha
# (least squares; relative RMS error of each), and print the increments per factor 4 in n.
import re, math, sys, collections
def fit(xs,ys):
    k=len(xs);mx=sum(xs)/k;my=sum(ys)/k
    b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs);return b,my-b*mx
G=collections.OrderedDict()
for f in sys.argv[1:]:
    for l in open(f):
        g=re.match(r'(\S+) n=(\d+) eps=([\d.]+) R=(\d+).*TJ mean ([\d.]+) max (\d+).*maxD mean ([\d.]+) max (\d+)',l)
        if g: G.setdefault((g[1],g[3],g[4]),[]).append((math.log2(int(g[2])),float(g[5]),int(g[6]),float(g[7]),int(g[8])))
for (spec,eps,R),rows in G.items():
    rows.sort(); xs=[r[0] for r in rows]; ys=[r[1] for r in rows]
    if len(xs)<5: continue
    def rms(pred): return math.sqrt(sum(((p-y)/y)**2 for p,y in zip(pred,ys))/len(ys))
    a,b=fit(xs,ys); e1=rms([a*x+b for x in xs])
    a2,b2=fit(xs,[math.sqrt(y) for y in ys]); e2=rms([(a2*x+b2)**2 for x in xs])
    a3,b3=fit(xs,[math.log2(y) for y in ys]); e3=rms([2**(a3*x+b3) for x in xs])
    print(f'{spec} eps={eps} R={R} log2n={int(xs[0])}..{int(xs[-1])}')
    print('   TJ mean:',' '.join(f'{y:.1f}' for y in ys),'| TJ max:',' '.join(str(r[2]) for r in rows),'| maxD mean:',' '.join(f'{r[3]:.1f}' for r in rows))
    print('   increments per x4:',' '.join(f'{ys[i+1]-ys[i]:.0f}' for i in range(len(ys)-1)),'| ratios:',' '.join(f'{ys[i+1]/ys[i]:.2f}' for i in range(len(ys)-1)))
    print(f'   fits: log {a:.1f}*log2n{b:+.0f} (rel.rms {e1:.3f}); log^2 ({a2:.2f}*(log2n-{-b2/a2:.1f}))^2 (rel.rms {e2:.3f}); power n^{a3:.3f} (rel.rms {e3:.3f}); local exponent at the last step {math.log2(ys[-1]/ys[-2])/(xs[-1]-xs[-2]):.3f}')
