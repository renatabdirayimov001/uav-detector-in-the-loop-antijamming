"""Robustness study (paper Sections 6.5-6.6): fading-only packet losses (noise power sweep),
AFT margin m, and run length T. Reproduces results/robustness/robustness_raw.csv.
Usage: python scripts/run_robustness_study.py   (about 6 min on 2 CPU cores)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import pandas as pd
from joblib import Parallel, delayed
from emulator_and_policies import Env, run, load, Oracle, DetHard, FCCS
from feedback_policies import NackHop, AFT
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'robustness'); os.makedirs(OUT, exist_ok=True)
D = {b: load(b) for b in ['5', '2.4']}

def one(kind, b, dom, jt, sd, pname, par):
    meta, SC, TI, warm = D[b]
    env = Env(b, meta, SC, TI, dom, jt, 10000 + sd, par.get('T', 2000))
    if 'N' in par: env.N = par['N']          # noise power in dBm (default -95)
    mk = {'Oracle-state': lambda: Oracle(), 'Detector hard (0.5)': lambda: DetHard(), 'FCCS (conformal)': lambda: FCCS(warm),
          'Det + NACK-hop': lambda: NackHop(), 'AFT (proposed)': lambda: AFT(warm, margin=par.get('m', 0.3))}
    r = run(env, mk[pname]()); r.update(kind=kind, band=b, domain=dom, jammer=jt, seed=sd, policy=pname, **par); return r

P = ['Oracle-state', 'Detector hard (0.5)', 'FCCS (conformal)', 'Det + NACK-hop', 'AFT (proposed)']
J, S = ['sweep', 'markov', 'reactive'], range(10)
jobs = []
for b, Ns in [('5', [-95, -93, -92, -91, -90, -87, -85]), ('2.4', [-95, -90, -87, -85])]:
    for N in Ns:
        for dom in ['matched', 'pow_shift']:
            jobs += [('noise', b, dom, jt, sd, p, {'N': N}) for jt in J for sd in S for p in P]
for m in [0.0, 0.3, 0.6, 1.0]:
    for dom in ['matched', 'pow_shift']:
        jobs += [('margin', '5', dom, jt, sd, 'AFT (proposed)', {'m': m}) for jt in J for sd in S]
for T in [500, 1000, 5000]:
    jobs += [('horizon', '5', 'pow_shift', jt, sd, p, {'T': T}) for jt in J for sd in S for p in P if p != 'FCCS (conformal)']
R = pd.DataFrame(Parallel(n_jobs=-1, batch_size=8)(delayed(one)(*j) for j in jobs))
R.to_csv(f'{OUT}/robustness_raw.csv', index=False)
n = R[R.kind == 'noise']
print(n.groupby(['band', 'domain', 'N', 'policy']).PSR.mean().unstack('policy').round(4).to_string())
print(R[R.kind == 'margin'].groupby(['domain', 'm']).PSR.mean().unstack(0).round(4).to_string())
print(R[R.kind == 'horizon'].groupby(['T', 'policy']).PSR.mean().unstack(1).round(4).to_string())
