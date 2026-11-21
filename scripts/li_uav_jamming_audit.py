"""Leakage-aware audit of the public UAV jamming dataset of Li et al. (IEEE Access 2022).
Get the data:  git clone https://github.com/michaelevol/uavs_jamming_detection data/uavs_jamming_detection
Usage: python scripts/li_uav_jamming_audit.py"""
import os, re, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
X = os.path.join(BASE, 'data', 'uavs_jamming_detection', 'Feature-based Approach', 'The Sorted Dataset Based on Separation and Location.xlsx')
OUT = os.path.join(BASE, 'results', 'li_audit'); os.makedirs(OUT, exist_ok=True)
xl = pd.ExcelFile(X); rows = []
for s in xl.sheet_names:
    if s.startswith('Summary'): continue
    e = xl.parse(s)
    if s == 'Clean Jamming': e['type'], e['sep'], e['loc'] = 'Clean', np.nan, 'none'
    else:
        m = re.match(r'([\d.]+)m - (.+) (J\d)', s); e['sep'], e['type'], e['loc'] = float(m.group(1)), m.group(2), m.group(3)
    e['sheet'] = s; rows.append(e)
D = pd.concat(rows, ignore_index=True); D.columns = [c.strip() for c in D.columns]
D = D.rename(columns={'Subcarriers (FFT)': 'FFT', 'Average Recieved Power (mW)': 'RxPow_mW', 'Average Signal Power 2': 'SigPow',
                      'Average Noise Power': 'NoisePow', 'Subcarrier Spacing': 'SCS', 'Symbol time': 'Tsym', 'CP length': 'CP'})
F = ['SCS', 'Tsym', 'FFT', 'CP', 'RxPow_mW', 'Threshold', 'SNR', 'SigPow', 'NoisePow']
D['idx'] = D.groupby('sheet').cumcount(); D['n'] = D.groupby('sheet')['idx'].transform('max') + 1
J = D[D.type != 'Clean'].copy()
MODELS = {'RF': lambda s: RandomForestClassifier(300, n_jobs=-1, random_state=s), 'HistGB': lambda s: HistGradientBoostingClassifier(random_state=s),
          'LogReg': lambda s: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))}
res = []
def ev(proto, model, seed, tr, te):
    m = MODELS[model](seed).fit(tr[F], tr.type); p = m.predict(te[F])
    res.append(dict(protocol=proto, model=model, seed=seed, acc=accuracy_score(te.type, p), macroF1=f1_score(te.type, p, average='macro'), n_test=len(te)))
for seed in [0, 1, 2]:
    for model in MODELS:
        tr, te = train_test_split(J, test_size=0.3, random_state=seed, stratify=J.type); ev('random 70/30', model, seed, tr, te)
        ev('temporal block (first 70% / last 30% of each capture)', model, seed, J[J.idx < 0.7 * J.n], J[J.idx >= 0.7 * J.n])
        for l in sorted(J['loc'].unique()): ev('leave-one-jammer-location-out', model, seed, J[J['loc'] != l], J[J['loc'] == l])
        for s in sorted(J.sep.unique()): ev('leave-one-separation-out', model, seed, J[J.sep != s], J[J.sep == s])
R = pd.DataFrame(res); R.to_csv(f'{OUT}/li_audit_raw.csv', index=False)
T = R.groupby(['protocol', 'model'])[['acc', 'macroF1']].agg(['mean', 'std']); T.to_csv(f'{OUT}/li_audit_summary.csv'); print(T.round(4))
C = D[D.type == 'Clean']; print('Energy-threshold flag: detection', (J.Threshold > 0.5).mean(), 'false alarm', (C.Threshold > 0.5).mean())
os_rows = []
for t in sorted(J.type.unique()):
    tr = D[(D.type != t) & (D.idx < 0.7 * D.n)]; val = D[(D.type != t) & (D.idx >= 0.7 * D.n)]; te = D[D.type == t]
    m = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(tr[F], tr.type)
    pk = m.predict_proba(val[F]).max(1); pu = m.predict_proba(te[F]).max(1); pred = pd.Series(m.predict(te[F]))
    os_rows.append(dict(unseen=t, flagged_as_jamming=float((pred != 'Clean').mean()), pred_dist=pred.value_counts(normalize=True).round(3).to_dict(),
                        unknown_AUROC_maxprob=roc_auc_score(np.r_[np.zeros(len(pk)), np.ones(len(pu))], np.r_[-pk, -pu])))
pd.DataFrame(os_rows).to_csv(f'{OUT}/li_openset.csv', index=False); print(pd.DataFrame(os_rows))
