"""Offline closed-loop policy study on real WiFiSpectralJam detector scores (reproduces results/policy_study).
Usage: python scripts/run_policy_study.py   (about 5-10 min on a laptop CPU)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from emulator_and_policies import Env, run, load, Oracle, DetHard, FBR, FCCS
from feedback_policies import NackHop, AFT
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'policy_study'); os.makedirs(OUT, exist_ok=True)
rows = []
for b in ['5', '2.4']:
    meta, SC, TI, warm = load(b)
    for dom in ['matched', 'pow_shift', 'loc_shift', 'loc_pow']:
        for jt in ['sweep', 'markov', 'reactive']:
            for sd in range(10):
                for p in [Oracle(), DetHard(), FBR(), FCCS(warm), NackHop(), AFT(warm, nack_hop=False), AFT(warm)]:
                    r = run(Env(b, meta, SC, TI, dom, jt, 10000 + sd, 2000), p)
                    r.update(band=b, domain=dom, jammer=jt, seed=sd, policy=p.name); rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/policy_study_raw.csv', index=False)
print(R.groupby(['band', 'domain', 'policy']).PSR.mean().unstack(1).round(4).to_string())
st = []
for (b, dom), g in R.groupby(['band', 'domain']):
    piv = g.pivot_table(index=['jammer', 'seed'], columns='policy', values='PSR')
    for p in piv.columns:
        if p == 'AFT (proposed)': continue
        d = piv['AFT (proposed)'] - piv[p]
        try: pv = wilcoxon(d).pvalue
        except Exception: pv = np.nan
        st.append(dict(band=b, domain=dom, vs=p, diff_pp=100 * d.mean(), wins=int((d > 0).sum()), ties=int((d == 0).sum()), n=len(d), p=pv,
                       oracle_gap_pp=100 * (piv['Oracle-state'] - piv[p]).mean()))
pd.DataFrame(st).to_csv(f'{OUT}/paired_stats_aft.csv', index=False)
# sensitivity of AFT (5 GHz)
meta, SC, TI, warm = load('5'); rows = []
for eps in [1e-4, 1e-3, 1e-2]:
    for down in [0.25, 0.5, 1.0]:
        for up in [5e-4, 2e-3, 1e-2]:
            for dom in ['pow_shift', 'matched']:
                ps = [run(Env('5', meta, SC, TI, dom, jt, 10000 + sd, 2000), AFT(warm, eps=eps, down=down, up=up))['PSR']
                      for jt in ['sweep', 'markov', 'reactive'] for sd in range(5)]
                rows.append(dict(eps=eps, down=down, up=up, domain=dom, PSR=np.mean(ps)))
pd.DataFrame(rows).pivot_table(index=['eps', 'down', 'up'], columns='domain', values='PSR').to_csv(f'{OUT}/aft_sensitivity_5GHz.csv')
print('done')
