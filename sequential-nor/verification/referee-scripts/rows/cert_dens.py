#!/usr/bin/env python3 -I
"""Referee: exact l_4, mu_4 at sample noise levels against the intervals of d printed by rows_check
(lines "eps [a,b] d [lo,hi]"): needs lo <= l_4 - 2 delta and mu_4 + 2 delta <= hi on [a,b]
(printed with 5-6 digits, so a tolerance of 1e-5 in d and eps is used)."""
import sys,re
from fractions import Fraction as F
def lm(e):
    mu=F(1,2)
    for s in range(4):
        l=(1-e*mu)/(3-e-e*mu); mu=(1-e*l)/(2-e*l)
    return l,mu
dl=F(2,100000); n=0; bad=0; ms=1
for fn in sys.argv[1:]:
    for line in open(fn):
        m=re.match(r'\s*eps \[([0-9.]+),([0-9.]+)\] d \[([0-9.]+),([0-9.]+)\]',line)
        if not m: continue
        a,b,lo,hi=[F(x) for x in m.groups()]
        for k in range(21):
            e=a+(b-a)*k/20; l,mu=lm(e)
            s=min(l-2*dl-lo, hi-mu-2*dl); ms=min(ms,s)
            if s<-F(1,100000): bad+=1; print('NOT ENCLOSED',fn,line.strip(),float(e))
        n+=1
print('%d intervals, %d violations, least slack %.2e'%(n,bad,float(ms)))
for e in (F(455,1000),F(1,2),F(7,10),F(9,10),F(1)): print('eps %.3f l4 %.6f mu4 %.6f'%(e,*map(float,lm(e))))
