# climb.py: hill climbing over periodic designs (layered, or plain periodic when m=0) that
# maximises the table-Jacobi rounds at a given noise, reach R fixed.
# usage: python3 -I climb.py eps R P layered(0/1) evals seed out.log
import subprocess, sys, random, re
eps=float(sys.argv[1]);R=int(sys.argv[2]);P=int(sys.argv[3]);lay=int(sys.argv[4]);EV=int(sys.argv[5]);rs=int(sys.argv[6]);out=open(sys.argv[7],'a')
rng=random.Random(rs)
w=P*((R+P)//P)           # twin period: a multiple of P above the reach
def menu():
    M=[f'L{d}' for d in range(R+1,R+2*P+2)]+[f'L{2*R+k}' for k in range(1,P+1)]+[f'L{4*R+k}' for k in range(1,P+1)]
    M+=['L1','L2','L3','Z','O','P1','P1','P2',f'H{R+1}',f'S{R+1}']
    if lay:
        M+=[f'E{c}' for c in range(P)]*2+['A']+[f'Q{w}-{c}' for c in range(P)]+[f'L{w+k}' for k in range(1,7)]+[f'L{2*w+k}' for k in range(1,4)]
    return M
M=menu()
def score(des,L=160,seeds=2,cap=3000):
    m=(480//P)*P if lay else 0
    n=(L+1)*2*480
    o=subprocess.run(['./hunt',str(n),str(eps),str(R),'0.5',str(m),str(seeds),','.join(des),str(cap)],capture_output=True,text=True)
    g=re.search(r'TJ mean ([\d.]+) max (\d+)(\+?) \| maxD mean ([\d.]+)',o.stdout)
    return float(g[1]),float(g[4])
best=None
des=[rng.choice(M) for _ in range(P)]; sc,d=score(des)
for it in range(EV):
    nd=list(des)
    k=rng.randrange(P); nd[k]=rng.choice(M)
    if rng.random()<0.3: nd[rng.randrange(P)]=rng.choice(M)
    s2,d2=score(nd)
    if s2>=sc: des,sc,d=nd,s2,d2
    out.write(f'{it}\t{s2}\t{d2}\t{",".join(nd)}\n');out.flush()
# growth of the best
res=[]
for L in (40,160,640,2560):
    res.append(score(des,L,2,20000))
print(f'eps={eps} R={R} P={P} lay={lay} seed={rs} best {",".join(des)} : TJ at L=40,160,640,2560: '+' '.join(f'{a:.1f}' for a,_ in res)+' | maxD: '+' '.join(f'{b:.1f}' for _,b in res),flush=True)
