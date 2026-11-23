import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
from a05data import *
names,P,plants,D=load(); n,m=D.shape
def stats(A,label):
    tot,cpt,k,T=system_cost(A,P,D); c=RHO*D[np.arange(n),A]+unit_cost(T)[A]
    g=np.abs(c[:,None]-c[None,:]).sum()/(2*n*c.sum())
    dev=0
    for i in range(n):
        j0=A[i]; Tm=T.copy(); Tm[j0]-=P[i]; cc=RHO*D[i]+unit_cost(Tm+P[i])
        dev+= cc.min()<cc[j0]-1e-9
    print(f"{label:22s} cost {cpt:.4f}  plants {k}  Cmax {c.max():.2f}  Gini {g:.3f}  tkm {(P*D[np.arange(n),A]).sum():,.0f}  would-switch {dev}  below5kt {int(((T>0)&(T<T0)).sum())}")
    return set(np.where(T>0)[0]),T
it=np.load(OUTPUT_DIR+"/iter_assign.npy"); mi=np.load(OUTPUT_DIR+"/milp_assign_K40.npy")
s1,T1=stats(it,"equilibrium (paper)"); s2,T2=stats(mi,"planner (MILP)")
print("open only in planner:",[(plants[j],round(T2[j])) for j in s2-s1]); print("open only in equilibrium:",[(plants[j],round(T1[j])) for j in s1-s2])
print("municipalities assigned differently:",int((it!=mi).sum()))
import csv
rows=[]
for label,A in (("fixed_point_equilibrium",it),("planner_milp",mi)):
    tot,cpt,k,T=system_cost(A,P,D); c=RHO*D[np.arange(n),A]+unit_cost(T)[A]
    g=np.abs(c[:,None]-c[None,:]).sum()/(2*n*c.sum())
    dev=sum(1 for i in range(n) if (lambda Tm:(RHO*D[i]+unit_cost(Tm+P[i])).min()<(RHO*D[i]+unit_cost(Tm+P[i]))[A[i]]-1e-9)(T-np.eye(m)[A[i]]*P[i]))
    rows.append(dict(method=label,cost_per_t=round(cpt,4),total_cost=round(tot),plants=k,c_max=round(c.max(),2),gini=round(g,3),tkm=round((P*D[np.arange(n),A]).sum()),would_switch=dev))
with open(OUTPUT_DIR+"/method_comparison.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("written",OUTPUT_DIR+"/method_comparison.csv")
