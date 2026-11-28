# climb2.py: hill climbing over plain periodic pointer patterns (no layers, no preamble): period P,
# distances from a menu beyond the reach (several scales), maximising table-Jacobi rounds at n=2^17.
# usage: python3 -I climb2.py eps R P evals seed out.log
import subprocess, sys, random, re
eps=float(sys.argv[1]);R=int(sys.argv[2]);P=int(sys.argv[3]);EV=int(sys.argv[4]);rs=int(sys.argv[5]);out=open(sys.argv[6],'a')
rng=random.Random(rs); d=R+1
M=[f'L{d+k}' for k in range(0,P+1)]+[f'L{c*d+k}' for c in (2,3,4,6,8,16,32,64,256) for k in (0,1)]+[f'H{d}',f'H{d+1}',f'S{d}',f'S{d+1}',f'U{d}-{4*R}',f'U{d}-{64*R}','F900','F990','F999']
def score(des,ln=17,seeds=2):
    o=subprocess.run(['./hunt',str(2**ln),str(eps),str(R),'0','0',str(seeds),','.join(des),'20000'],capture_output=True,text=True)
    g=re.search(r'TJ mean ([\d.]+) max (\d+)(\+?) \| maxD mean ([\d.]+)',o.stdout)
    return float(g[1]),float(g[4])
des=[rng.choice(M) for _ in range(P)]; sc,dd=score(des)
for it in range(EV):
    nd=list(des); nd[rng.randrange(P)]=rng.choice(M)
    if rng.random()<0.3: nd[rng.randrange(P)]=rng.choice(M)
    s2,d2=score(nd)
    if s2>=sc: des,sc,dd=nd,s2,d2
    out.write(f'{it}\t{s2}\t{d2}\t{",".join(nd)}\n');out.flush()
res=[score(des,ln,2) for ln in (13,15,17,19,21)]
print(f'eps={eps} R={R} P={P} seed={rs} best {",".join(des)} : TJ at n=2^13,15,17,19,21: '+' '.join(f'{a:.1f}' for a,_ in res)+' | maxD: '+' '.join(f'{b:.1f}' for _,b in res),flush=True)
