# run_families.py: growth of the table-Jacobi rounds with n on many families, R = C*log2(n).
# usage: python3 -I run_families.py C out.tsv [family-substring]
import subprocess, sys, math, re
C=int(sys.argv[1]); out=open(sys.argv[2],'a'); filt=sys.argv[3] if len(sys.argv)>3 else ''
NS=[13,15,17,19,21]; EPS=[0.46,0.50,0.60]
def fams(n,R):
    q=int(math.isqrt(n)); e=R+(R%2)  # e even >= R
    F=[
     ('fixed H=2 (in reach)',f'L2',0,0), ('fixed H=R+1',f'L{R+1}',0,0), ('fixed H=2R',f'L{2*R}',0,0), ('fixed H=sqrt n',f'L{q}',0,0),
     ('two dist R+1,R+2',f'L{R+1},L{R+2}',0,0), ('two dist R+1,2R+1',f'L{R+1},L{2*R+1}',0,0), ('three dist',f'L{R+2},L{2*R+3},L{3*R+1}',0,0),
     ('uniform dist R+1..2R',f'U{R+1}-{2*R}',0,0), ('uniform dist R+1..4R',f'U{R+1}-{4*R}',0,0), ('uniform dist R+1..sqrt n',f'U{R+1}-{q+R+1}',0,0),
     ('i/2',f'F500',0,0), ('0.9 i',f'F900',0,0), ('0.99 i',f'F990',0,0), ('i/2 and R+1 alternating',f'F500,L{R+1}',0,0),
     ('block m=R/2 (in reach) pf=0','A',0,R//2), ('block m=R+1 pf=0','A',0,R+1), ('block m=4R pf=0','A',0,4*R), ('block m=600 pf=.5','A',0.5,600), ('block m=4R pf=.5','A',0.5,4*R),
     ('barrier 2,3 (in reach) pf=.5','L2,L3',0.5,0), ('barrier 2e,3e pf=.5',f'L{2*e},L{3*e}',0.5,0), ('barrier 2e+2,3e+3 pf=.5',f'L{2*e+2},L{3*e+3}',0.5,0), ('barrier 2e+2,3e+3 pf=0',f'L{2*e+2},L{3*e+3}',0,0),
     ('mixed L2,E0 (in reach) m=600','L2,E0',0.5,600), ('mixed L(e+2),E0 m=600',f'L{e+2},E0',0.5,600), ('mixed L(e+2),E0 m=8e',f'L{e+2},E0',0.5,8*e), ('mixed L(e+4),E0 m=600',f'L{e+4},E0',0.5,600),
     ('mixed L(e+2),E0 m=600 pf=.1',f'L{e+2},E0',0.1,600),
     ('twin L(e+4),Q(e) m=600',f'L{e+4},Q{e}-0',0.5,600), ('twin L(e+2),Q(e) m=600',f'L{e+2},Q{e}-0',0.5,600), ('twin L(2e+2),Q(2e) m=1200',f'L{2*e+2},Q{2*e}-0',0.5,1200),
     ('hier (R+1)*2^v2(i)',f'H{R+1}',0,0), ('hier 2^v2(i) + R+1 alt',f'H1,L{R+1}',0,0), ('trailing ones',f'S{R+1}',0,0),
    ]
    return F
def run(n,eps,R,pf,m,spec,seeds):
    o=subprocess.run(['./hunt',str(n),str(eps),str(R),str(pf),str(m),str(seeds),spec,'20000'],capture_output=True,text=True)
    g=re.search(r'TJ mean ([\d.]+) max (\d+)(\+?) \| maxD mean ([\d.]+) max (\d+)',o.stdout)
    return float(g[1]),int(g[2]),float(g[4]),int(g[5]),o.stderr.strip()
def fit(xs,ys):
    k=len(xs);mx=sum(xs)/k;my=sum(ys)/k
    b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs);a=my-b*mx
    return b,a,sum((y-a-b*x)**2 for x,y in zip(xs,ys))
nf=len(fams(2**13,C*13))
for fi in range(nf):
    name=fams(2**13,C*13)[fi][0]
    if filt not in name: continue
    for eps in EPS:
        rows=[]
        for ln in NS:
            n=2**ln;R=C*ln;_,spec,pf,m=fams(n,R)[fi]
            seeds=3 if ln<21 else 2
            tm,tx,dm,dx,err=run(n,eps,R,pf,m,spec,seeds)
            rows.append((ln,tm,tx,dm,dx))
            out.write(f'{name}\t{spec}\t{pf}\t{m}\t{eps}\t{n}\t{R}\t{tm}\t{tx}\t{dm}\t{dx}\t{err}\n');out.flush()
        xs=[r[0] for r in rows];ys=[r[1] for r in rows]
        sl,_,_=fit(xs,ys); al,_,_=fit(xs,[math.log2(max(y,1)) for y in ys])
        ds=[r[3] for r in rows]; ad,_,_=fit(xs,[math.log2(max(y,1)) for y in ds])
        print(f'{name:34s} eps={eps:.2f} C={C} TJ:'+' '.join(f'{y:7.1f}' for y in ys)+f' | slope/log2n {sl:6.2f} alpha {al:5.2f} | maxD:'+' '.join(f'{y:7.1f}' for y in ds)+f' alphaD {ad:5.2f}',flush=True)
