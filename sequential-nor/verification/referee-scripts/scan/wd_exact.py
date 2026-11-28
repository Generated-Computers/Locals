# Referee's independent exact evaluation of G_2 for "window designs" (Remark rem:scan(iii)).
# design: widths list, gates list of ('P',i,off) / ('T',i,off)
import sys, itertools, math
def run(eps, widths, gates, verbose=False):
    vp=((2*eps+1)-math.sqrt(4*eps+1))/(2*eps); A=eps*vp; B=eps*(1-vp); c=1-eps
    T=len(gates)
    stat=[(c,'k'),(A,'1'),(B,'0')]
    def pgate(x,s,t):
        y = t if s=='k' else (1 if s=='1' else 0)
        return 0 if (x or y) else 1
    def win(w, start, d, d0):
        # start: dict (xa,xb)->mass; returns dict (xa tuple, ta tuple, xb tuple, tb tuple)->mass
        cur={(k,(),(),(),()):m for k,m in start.items()}
        for pos in range(w):
            nxt={}
            for ((xa,xb),XA,TA,XB,TB),m in cur.items():
                for ps,s in stat:
                    if pos==d0:
                        na=pgate(xa,s,0); nb=pgate(xb,s,1)
                        key=((na,nb),XA+(na,),TA+(0,),XB+(nb,),TB+(1,)); nxt[key]=nxt.get(key,0)+m*ps
                    else:
                        for t in (0,1):
                            na=pgate(xa,s,t); nb=pgate(xb,s,t)
                            key=((na,nb),XA+(na,),TA+(t,),XB+(nb,),TB+(t,)); nxt[key]=nxt.get(key,0)+m*ps*(d if t else 1-d)
            cur=nxt
        out={}
        for (_,XA,TA,XB,TB),m in cur.items():
            key=(XA,TA,XB,TB); out[key]=out.get(key,0)+m
        return out
    def rd(g,sel,copy):
        kind,i,off=g; XA,TA,XB,TB=sel[i]
        if kind=='P': return (XA,XB)[copy][off]
        return (TA,TB)[copy][off]
    def estep(st,va,vb):
        n={}
        for (a,b),m in st.items():
            for ps,s in stat:
                ya = va if s=='k' else (1 if s=='1' else 0); yb = vb if s=='k' else (1 if s=='1' else 0)
                k=(0 if (a or ya) else 1, 0 if (b or yb) else 1); n[k]=n.get(k,0)+m*ps
        return n
    def clean(d):
        q=B+c*(1-d); al=1/(1+q)
        ws=[win(w,{(0,0):al,(1,1):1-al},d,None) for w in widths]
        # affine out map and ones, Zk
        res={}
        Z=[0.0]*(T+1)
        f0=[0,0]; ones=[0,0]
        for combo in itertools.product(*[list(x.items()) for x in ws]):
            p=1.0; sel=[]
            for k,m in combo: p*=m; sel.append(k)
            for s in (0,1):
                st={(s,s):1.0}; o=0
                for g in gates:
                    v=rd(g,sel,0); st=estep(st,v,v); o+=sum(m for (a,b),m in st.items() if a==1)
                f0[s]+=p*sum(m for (a,b),m in st.items() if a==0); ones[s]+=p*o
            z=1.0
            for i,g in enumerate(gates):
                v=rd(g,sel,0); z*=B+c*(1-v); Z[i+1]+=p*z
        # stationary P[x=0] before the period: p0 = p0 f0[0] + (1-p0) f0[1]
        p0=f0[1]/(1-f0[0]+f0[1])
        dens=(p0*ones[0]+(1-p0)*ones[1])/T
        Cst=sum(Z[1:])/(1-Z[T])
        return dens,p0,Cst,ws,q,al
    # fixed point by bisection on dens(d)-d (scan for all roots)
    roots=[]; grid=40; x0=0.0; f0=clean(0.0)[0]-0.0
    for i in range(1,grid+1):
        x1=i/grid; f1=clean(x1)[0]-x1
        if f0*f1<0:
            a,b,fa=x0,x1,f0
            for _ in range(50):
                m=(a+b)/2; fm=clean(m)[0]-m
                if fa*fm<=0: b=m
                else: a,fa=m,fm
            roots.append((a+b)/2)
        x0,f0=x1,f1
    out=[]
    for d in roots:
        dens,p0,Cst,ws,q,al=clean(d)
        tot=0.0
        for iw,w in enumerate(widths):
            for d0 in ['neg']+list(range(w)):
                if d0=='neg':
                    # explicit sum over the differing gate at positions j=-1,-2,...: pair state after position -1
                    m10=m01=0.0; cur=(c*al,0.0)  # (mass (1,0), mass (0,1)) right after the differing gate
                    for j in range(1,400):
                        m10+=cur[0]; m01+=cur[1]; cur=(cur[1]*q,cur[0]*q)
                    dw=win(w,{(1,0):m10,(0,1):m01},d,None)
                else:
                    dw=win(w,{(0,0):al,(1,1):1-al},d,d0)
                others=[list(ws[i].items()) if i!=iw else list(dw.items()) for i in range(len(widths))]
                sub=0.0
                for combo in itertools.product(*others):
                    p=1.0; sel=[]
                    for k,m in combo: p*=m; sel.append(k)
                    if sel[iw][0]==sel[iw][2] and sel[iw][1]==sel[iw][3]: continue
                    st={(0,0):p0,(1,1):1-p0}; cnt=0.0
                    for g in gates:
                        st=estep(st,rd(g,sel,0),rd(g,sel,1)); cnt+=sum(m for (a,b),m in st.items() if a!=b)
                    sub+=p*(cnt+Cst*sum(m for (a,b),m in st.items() if a!=b))
                if verbose: print('  window',iw,'dirty at',d0,'%.6f'%sub)
                tot+=sub
        out.append((d,tot/T))
    return out
def parse(s):
    w,g=s.split(':'); widths=[int(x) for x in w.split(',')]
    gates=[(t[0],int(t[1]),int(t[2])) for t in g.split(',')]
    return widths,gates
if __name__=='__main__':
    eps=float(sys.argv[1])
    for s in sys.argv[2:]:
        w,g=parse(s)
        for d,G in run(eps,w,g): print(s,'eps',eps,'d %.8f'%d,'G2 %.8f'%G)
