#!/usr/bin/env python3 -I
"""Referee: compare TU/TD and the table nx of rows_check.c (dumped by cert_rows_tu) with exact
rational operators built from the definition in the paper.  Checks TD <= exact min <= exact max <= TU,
the tightness, and that the table equals the one from the definition."""
import sys, random
from fractions import Fraction as F
from itertools import product
def f(v,o,b,w):
    if v==0 and b==1: return w
    if v==1 and b==1: return o&w
    if v==0 and b==0: return o|w
    return o
def read(p,q): return (1-(p[0]|q[0]), f(p[0],p[1],q[0],q[1]))
def idx(s): return s[0]+2*s[1]+4*s[2]+8*s[3]
ST=list(product((0,1),repeat=4)); ST.sort(key=idx)
def table(y,s):
    r=[]
    for i in range(3):
        low=read((s[0],s[1]),((y&1,y>>1),(1,0),(0,0))[i])
        r.append([idx(low+read((s[2],s[3]),(low,(1,0),(0,0))[k])) for k in range(3)])
    return r
def mix(sel,E,D,a,b,c): return sel((1-e)*a+e*(d*b+(1-d)*c) for e in E for d in D)
def T(sel,E,D,y,g):
    out=[]
    for s in ST:
        t=table(y,s); out.append(mix(sel,E,D,*[mix(sel,E,D,g[t[i][0]],g[t[i][1]],g[t[i][2]]) for i in range(3)]))
    return out
def single(e,d1,d2,y,g):   # operator of the paper for one eps, d2 = d2[i] depending on the lower outcome
    out=[]
    for s in ST:
        t=table(y,s); p1=(1-e,e*d1,e*(1-d1)); v=0
        for i in range(3):
            p2=(1-e,e*d2[i],e*(1-d2[i])); v+=p1[i]*sum(p2[k]*g[t[i][k]] for k in range(3))
        out.append(v)
    return out
random.seed(7); n=0; worst=0.0; nt=0; bad=0
for line in open(sys.argv[1]):
    t=line.split()
    if t[0]=='NX':
        y,s=int(t[1]),int(t[2]); mine=[v for row in table(y,ST[s]) for v in row]
        if mine!=[int(x) for x in t[3:]]: bad+=1; print('TABLE DIFFERS',y,s)
        nt+=1; continue
    v=[F(float.fromhex(x)) for x in t[:4]]; E=(v[0],v[1]); D=(v[2],v[3]); y=int(t[4])
    g=[F(float.fromhex(x)) for x in t[5:21]]; u=[F(float.fromhex(x)) for x in t[21:37]]; l=[F(float.fromhex(x)) for x in t[37:53]]
    hi=T(max,E,D,y,g); lo=T(min,E,D,y,g)
    for s in range(16):
        if not (l[s]<=lo[s]<=hi[s]<=u[s]): bad+=1; print('ROUNDING VIOLATION line',n,'state',s)
        worst=max(worst,float(u[s]-hi[s]),float(lo[s]-l[s]))
    for _ in range(5):       # single eps, densities inside the box, d2 depending on the outcome
        e=E[0]+(E[1]-E[0])*F(random.random()); d1=D[0]+(D[1]-D[0])*F(random.random()); d2=[D[0]+(D[1]-D[0])*F(random.random()) for _ in range(3)]
        sg=single(e,d1,d2,y,g)
        for s in range(16):
            if not (lo[s]<=sg[s]<=hi[s]): bad+=1; print('BOX OPERATOR NOT A BOUND line',n,'state',s)
    n+=1
print('%d operator samples, %d table rows compared, %d violations, largest slack of TU/TD over the exact box operator %.3e'%(n,nt,bad,worst))
