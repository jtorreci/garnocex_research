"""System-optimal (planner) concave-cost location-assignment for A05.
Secant underestimation of C_fix on log-spaced intervals -> valid lower bound; the MILP solution is then
evaluated with the exact cost -> upper bound. Single sourcing, all 32 consolidated groups as candidates."""
import sys, time; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np, highspy
from a05data import *
names,P,plants,D=load(); n,m=D.shape; Tmax=P.sum()
K=int(sys.argv[1]) if len(sys.argv)>1 else 40; tl=float(sys.argv[2]) if len(sys.argv)>2 else 600
bp=np.geomspace(T0,Tmax,K+1)
segs=[(1.0,T0,C0,0.0)]  # (L,U,alpha,beta): constant C0 on (0,T0]
for a,b in zip(bp[:-1],bp[1:]):
    fa,fb=fix(a),fix(b); beta=(fb-fa)/(b-a); segs.append((a,b,fa-beta*a,beta))
h=highspy.Highs(); h.setOptionValue("output_flag",False); h.setOptionValue("time_limit",tl); h.setOptionValue("mip_rel_gap",1e-5)
inf=highspy.kHighsInf
x=[[h.addBinary(obj=RHO*P[i]*D[i,j]) for j in range(m)] for i in range(n)]
z=[[h.addBinary(obj=s[2]) for s in segs] for j in range(m)]
t=[[h.addVariable(lb=0,ub=s[1],obj=s[3]) for s in segs] for j in range(m)]
for i in range(n): h.addConstr(sum(x[i][j] for j in range(m))==1)
for j in range(m):
    h.addConstr(sum(z[j])<=1)
    for k,s in enumerate(segs):
        h.addConstr(t[j][k]-s[1]*z[j][k]<=0); h.addConstr(t[j][k]-s[0]*z[j][k]>=0)
    h.addConstr(sum(t[j])-sum(P[i]*x[i][j] for i in range(n))==0)
t0=time.time(); h.run(); el=time.time()-t0
info=h.getInfo(); sol=h.getSolution().col_value
X=np.array(sol[:n*m]).reshape(n,m); A=X.argmax(axis=1)
tot,cpt,nop,T=system_cost(A,P,D)
lb=(info.mip_dual_bound+V*P.sum())/P.sum()
print(f"K={K} status={h.modelStatusToString(h.getModelStatus())} time={el:.0f}s gap={info.mip_gap:.2e}")
print(f"planner MILP solution: exact cost {cpt:.4f} EUR/t, plants {nop}, total {tot:,.0f}")
print(f"certified lower bound on any single-sourcing network: {lb:.4f} EUR/t")
np.save(OUTPUT_DIR+f"/milp_assign_K{K}.npy",A)
