"""Rebuild A05 inputs (production vector, 32-group road distance matrix) exactly as scripts/iterative_cost_assignment.py does."""
# Inputs and cost model are those of scripts/iterative_cost_assignment.py (A01 cost model, 32 consolidated groups).
import math, os, re
import numpy as np, pandas as pd
TABLAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "tablas")
C0, T0, V, RHO, DOT = 40_000, 5_000, 0.35, 0.35, 0.5
CONS = {2:1,12:1,17:3,15:6,40:19,32:20,23:22,26:22,28:22,34:27,35:27,37:27,38:36,43:36}
def norm(s):
    if pd.isna(s): return ""
    s = str(s).strip().replace("\n"," ").encode("ascii",errors="ignore").decode("ascii")
    s = re.sub(r"[.,]\s*(La|Los|Las|El)\s*$","",s,flags=re.IGNORECASE)
    s = re.sub(r"^(La|Los|Las|El|Les)\s+","",s,flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9 ]","",s.lower()); return re.sub(r"\s+"," ",s).strip()
def load():
    dd = pd.read_csv(os.path.join(TABLAS,"distancias_reales_plantas_municipios.csv"))
    mr = pd.read_csv(os.path.join(TABLAS,"datos_voronoi_municipios.csv"),encoding="utf-8",on_bad_lines="warn")
    pop = {norm(r.get("NOMBRE","")):int(r["HBT"]) for _,r in mr.iterrows() if norm(r.get("NOMBRE","")) and not pd.isna(r.get("HBT"))}
    asig = pd.read_csv(os.path.join(TABLAS,"asignacion_municipios_real.csv"))
    muns=[]
    for _,r in asig.iterrows():
        k=norm(str(r["municipio"]).strip()); p=pop.get(k,0)
        if p>0: muns.append((str(r["municipio"]).strip(),k,p*DOT))
    dl={}
    for _,r in dd.iterrows():
        pid=int(r["origin_id"]); mk=norm(str(r["destination_id"])); d=float(r["total_cost"])/1000
        if mk and d>0: dl[(pid,mk)]=d
    plants=sorted({CONS.get(p,p) for p,_ in dl})
    dc={}
    for (p,mk),d in dl.items():
        c=CONS.get(p,p)
        if dc.get((c,mk)) is None or d<dc[(c,mk)]: dc[(c,mk)]=d
    names=[m[0] for m in muns]; P=np.array([m[2] for m in muns])
    D=np.full((len(muns),len(plants)),np.inf)
    for i,m in enumerate(muns):
        for j,p in enumerate(plants):
            if (p,m[1]) in dc: D[i,j]=dc[(p,m[1])]
    return names,P,plants,D
def fix(T):
    T=np.asarray(T,float); out=np.where(T<=0,0.0,np.where(T<T0,C0,C0*(np.log2(np.maximum(T,1)/T0)+1))); return out
def system_cost(assign,P,D):
    T=np.bincount(assign,weights=P,minlength=D.shape[1])
    total=fix(T).sum()+V*P.sum()+RHO*(P*D[np.arange(len(P)),assign]).sum()
    return total, total/P.sum(), int((T>0).sum()), T
def unit_cost(T):
    T=np.asarray(T,float); return np.where(T<=0,999.0,fix(T)/np.maximum(T,1e-9)+V)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs", "equilibrium_planner")
os.makedirs(OUTPUT_DIR, exist_ok=True)
