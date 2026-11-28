#!/usr/bin/env python3 -I
"""search.py -- referee's random search with successive re-measurement over designs with dependent reads.
Uses ./refsim2.  Every design must contain at least one dependent rule (R, A, G, F, P, L>=2).
Ranking is by the per-generation estimate R0hat (equal to the layer growth when all reads go one layer down).

usage: python3 -I search.py random  SEED N TMIN TMAX OUT      (stage 1: N random designs, 2 runs each)
       python3 -I search.py remeasure IN TOP RUNS SEED0 OUT   (take the TOP best of IN, RUNS more runs each)
       python3 -I search.py list FILE RUNS SEED0 OUT          (measure the designs listed in FILE)
env: WORKERS (6), M (2000000), DELTA (0.003), L (40)
"""
import sys, os, random, subprocess, re
from concurrent.futures import ThreadPoolExecutor

WORKERS=int(os.environ.get('WORKERS','6')); M=os.environ.get('M','2000000'); L=os.environ.get('L','40')
HERE=os.path.dirname(os.path.abspath(__file__))
pat=re.compile(r"layer ([0-9.]+) \+- ([0-9.]+) \| pergen: R0hat ([0-9.]+) \+- ([0-9.]+)")

def measure(args):
    design,runs,seed0=args
    out=subprocess.run([os.path.join(HERE,'refsim2'),'0.46',design,M,L,str(runs),str(seed0)],capture_output=True,text=True)
    m=pat.search(out.stdout)
    if not m: return design,None
    return design,(float(m.group(1)),float(m.group(2)),float(m.group(3)),float(m.group(4)),runs)

def rand_rule_onelayer(rng,T):
    r=rng.random()
    if r<0.35: return 'U'
    if r<0.45: return 'I%d'%rng.randrange(T)
    if r<0.50: return 'X'
    if r<0.70: return 'R%d'%rng.choice([-8,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,8,0])
    if r<0.80: return 'A%d'%rng.choice([-6,-4,-3,-2,-1,1,2,3,4,6])
    if r<0.88: return 'F%d'%rng.randrange(-6,7)
    if r<0.93: return 'P%d'%rng.randrange(1,6)
    if r<0.97: return 'L%d'%rng.randrange(2,6)
    return 'O'

def rand_rule(rng,T):
    if os.environ.get('RULESET')=='onelayer': return rand_rule_onelayer(rng,T)
    r=rng.random()
    if r<0.30: return 'U'
    if r<0.40: return 'I%d'%rng.randrange(T)
    if r<0.45: return 'X'
    if r<0.48: return 'Y%d'%rng.randrange(T)
    if r<0.60: return 'R%d'%rng.choice([-4,-3,-2,-1,1,2,3,4,0])
    if r<0.66: return 'A%d'%rng.choice([-4,-3,-2,-1,1,2,3,4])
    if r<0.76: return 'G%d'%rng.choice([-2,-1,0,1,2])
    if r<0.79: return 'Q%d'%rng.choice([-1,0,1])
    if r<0.84: return 'F%d'%rng.randrange(-4,5)
    if r<0.88: return 'H%d'%rng.randrange(-4,5)
    if r<0.90: return 'V'
    if r<0.93: return 'L%d'%rng.randrange(2,5)
    if r<0.95: return 'N'
    if r<0.96: return 'Z'
    if r<0.98: return 'O'
    return 'P%d'%rng.randrange(1,5)

def rand_design(rng,tmin,tmax):
    while True:
        T=rng.randrange(tmin,tmax+1)
        rules=[rand_rule(rng,T) for _ in range(T)]
        s=''.join(x[0] for x in rules)
        if not any(c in s for c in 'UIF'): continue
        if not any(c in s for c in 'RAGFPL'): continue
        if s.count('Z')+s.count('O')+s.count('N')>T//3: continue
        return ','.join(rules)

def run_all(jobs,outpath):
    res=[]
    with ThreadPoolExecutor(WORKERS) as ex, open(outpath,'w') as f:
        for design,r in ex.map(measure,jobs):
            if r is None: continue
            res.append((design,r))
            f.write("%.5f %.5f %.5f %.5f %d %s\n"%(r[2],r[3],r[0],r[1],r[4],design)); f.flush()
    return res

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='random':
        seed=int(sys.argv[2]); N=int(sys.argv[3]); tmin=int(sys.argv[4]); tmax=int(sys.argv[5]); out=sys.argv[6]
        rng=random.Random(seed); seen=set(); jobs=[]
        while len(jobs)<N:
            d=rand_design(rng,tmin,tmax)
            if d in seen: continue
            seen.add(d); jobs.append((d,2,seed*100000+len(jobs)*2))
        run_all(jobs,out)
    elif mode=='remeasure':
        inp=sys.argv[2]; top=int(sys.argv[3]); runs=int(sys.argv[4]); seed0=int(sys.argv[5]); out=sys.argv[6]
        rows=[l.split() for l in open(inp) if l.strip()]
        rows.sort(key=lambda r:-float(r[0]))
        jobs=[(r[5],runs,seed0+i*runs) for i,r in enumerate(rows[:top])]
        run_all(jobs,out)
    elif mode=='list':
        inp=sys.argv[2]; runs=int(sys.argv[3]); seed0=int(sys.argv[4]); out=sys.argv[5]
        ds=[l.strip() for l in open(inp) if l.strip() and not l.startswith('#')]
        jobs=[(d,runs,seed0+i*runs) for i,d in enumerate(ds)]
        run_all(jobs,out)
