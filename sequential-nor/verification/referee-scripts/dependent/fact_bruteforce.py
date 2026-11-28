#!/usr/bin/env python3 -I
"""Brute-force check of the 'one fact' of the remark on dependent reads.
For every parent array j (0<=j_i<i, 1<=i<=n) and every pair of assignments x, x' in {0,1}^{n+1}:
gate k (k>=1) is regular if x_k = NOR(x_{k-1}, x_{j_k}) in both assignments.  For every p with x_p != x'_p:
 (A)  a regular gate k with k-1 == p or j_k == p does not hold 1 in both;
 (A*) the wider reading: no regular gate k > p holds 1 in both          (expected to FAIL);
 (B)  if gate p is regular, neither p-1 nor j_p holds 1 in both;
 (C)  consequence used in the text: a regular-or-not gate g whose read is p+1 (regular), a regular gate
      listing p, p-1 or j_p (p regular) reads (0,0) or a differing value.
Also counts how often a differing read ends a difference in a chain (the case the text leaves open).
"""
import itertools, sys

def nor(a,b): return 0 if (a or b) else 1

def run(n):
    cnt=dict(A=0,Astar=0,B=0,C=0); viol=dict(A=None,Astar=None,B=None,C=None)
    ends_by_diff_read=0; ends_by_one_both=0
    ranges=[range(i) for i in range(1,n+1)]
    for jj in itertools.product(*ranges):
        j=(None,)+jj
        for xb in range(1<<(n+1)):
            x=[(xb>>i)&1 for i in range(n+1)]
            for yb in range(xb+1,1<<(n+1)):
                y=[(yb>>i)&1 for i in range(n+1)]
                reg=[False]+[x[k]==nor(x[k-1],x[j[k]]) and y[k]==nor(y[k-1],y[j[k]]) for k in range(1,n+1)]
                for p in range(n+1):
                    if x[p]==y[p]: continue
                    nb=set()
                    for k in range(1,n+1):
                        if reg[k] and (k-1==p or j[k]==p):
                            cnt['A']+=1; nb.add(k)
                            if x[k]==1 and y[k]==1 and viol['A'] is None: viol['A']=(jj,x,y,p,k)
                        if reg[k] and k>p:
                            cnt['Astar']+=1
                            if x[k]==1 and y[k]==1 and viol['Astar'] is None: viol['Astar']=(jj,x,y,p,k)
                    if p>=1 and reg[p]:
                        cnt['B']+=1; nb.add(p-1); nb.add(j[p])
                        if (x[p-1]==1 and y[p-1]==1) or (x[j[p]]==1 and y[j[p]]==1):
                            if viol['B'] is None: viol['B']=(jj,x,y,p)
                    for q in nb:
                        cnt['C']+=1
                        if x[q]==1 and y[q]==1 and viol['C'] is None: viol['C']=(jj,x,y,p,q)
                # how a difference in the chain ends at a regular gate
                for k in range(2,n+1):
                    if reg[k] and x[k-1]!=y[k-1] and x[k]==y[k]:
                        if x[j[k]]==1 and y[j[k]]==1: ends_by_one_both+=1
                        elif x[j[k]]!=y[j[k]]: ends_by_diff_read+=1
                        else: raise SystemExit("impossible: difference ended with read 0 in both")
    return cnt,viol,ends_by_one_both,ends_by_diff_read

if __name__=='__main__':
    nmax=int(sys.argv[1]) if len(sys.argv)>1 else 5
    for n in range(2,nmax+1):
        cnt,viol,e1,e2=run(n)
        print("n=%d checks %s"%(n,cnt))
        for key in ('A','Astar','B','C'):
            print("   %-5s %s"%(key,"holds" if viol[key] is None else "FAILS, e.g. j=%s x=%s x'=%s p=%s%s"%(
                viol[key][0],viol[key][1],viol[key][2],viol[key][3],(" k/q=%s"%viol[key][4]) if len(viol[key])>4 else "")))
        print("   a difference in the chain ended at a regular gate: by a one in both %d times, by a differing read %d times"%(e1,e2))
