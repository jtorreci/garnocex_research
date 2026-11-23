import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
from a05data import *
names,P,plants,D=load(); n,m=D.shape
from fixed_point import fixed_point
def paper_procedure(A_init):
    return fixed_point(A_init, P, D)[0]
def best_response(A, rng, max_rounds=500):
    """Sequential: each municipality moves if its own cost (counting its own tonnage at the new plant) falls."""
    A=A.copy(); T=np.bincount(A,weights=P,minlength=m)
    for r in range(max_rounds):
        moved=0
        for i in rng.permutation(n):
            j0=A[i]; Tm=T.copy(); Tm[j0]-=P[i]
            c=RHO*D[i]+unit_cost(Tm+P[i])   # cost at each plant if i joins it
            j=int(np.argmin(c))
            if c[j]<c[j0]-1e-9: T[j0]-=P[i]; T[j]+=P[i]; A[i]=j; moved+=1
        if moved==0: return A,True
    return A,False
def is_nash(A):
    T=np.bincount(A,weights=P,minlength=m); bad=0
    for i in range(n):
        j0=A[i]; Tm=T.copy(); Tm[j0]-=P[i]; c=RHO*D[i]+unit_cost(Tm+P[i])
        if c.min()<c[j0]-1e-9: bad+=1
    return bad
rng=np.random.default_rng(2026)
it=np.load(OUTPUT_DIR+"/iter_assign.npy")
print("paper solution: cost %.4f plants %d; municipalities that would gain by switching: %d"%(system_cost(it,P,D)[1],system_cost(it,P,D)[2],is_nash(it)))
res=[]
for s in range(50):
    A_rand=rng.integers(0,m,n)
    Ap=paper_procedure(A_rand); _,cp,np_,_=system_cost(Ap,P,D)
    Ab,ok=best_response(A_rand,rng); _,cb,nb,_=system_cost(Ab,P,D)
    res.append((cp,np_,cb,nb,ok))
r=np.array([x[:4] for x in res])
print("paper procedure, random seeds: cost min %.4f median %.4f max %.4f; plants %s"%(r[:,0].min(),np.median(r[:,0]),r[:,0].max(),sorted(set(r[:,1].astype(int)))))
print("sequential best response (Nash), random seeds: cost min %.4f median %.4f max %.4f; plants %s; converged %d/50"%(r[:,2].min(),np.median(r[:,2]),r[:,2].max(),sorted(set(r[:,3].astype(int))),sum(x[4] for x in res)))
A0=np.argmin(D,axis=1); Ab,ok=best_response(A0,rng); print("best response from nearest seed: cost %.4f plants %d"%system_cost(Ab,P,D)[1:3])
np.save(OUTPUT_DIR+"/seeds.npy",r)
