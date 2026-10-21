import pandas as pd, numpy as np
from scipy.stats import wilcoxon
d = pd.read_csv('results/comparison/phasefinal_seed1/per_slice_metrics.csv')
SHORT = {'Uformer-lite (L1)':'L1',
         'Uformer + matched base (no pathology) [CONTROL]':'MatchedBase',
         'Uformer + PathologyLoss (current, nested)':'PathologyLoss',
         'Uformer + FlatROI [ROIRecNet-inspired]':'FlatROI',
         'Uformer + masked L1 w=0.5 [BraSyn 2023]':'BraSyn',
         'Noisy input (no restoration) [FLOOR]':'FLOOR',
         'Clean image (oracle) [CEILING]':'CEILING'}
d['arm'] = d.method.map(SHORT)

def subj(metric):
    return d.groupby(['arm','subject'])[metric].mean().unstack(0)

for metric in ['dice_et','dice_tc','psnr_et','psnr']:
    t = subj(metric)
    print(f'===== {metric} : subject-level means =====')
    print(f'{"arm":<15}{"n":>4}{"mean":>9}{"sd":>8}')
    for a in ['FLOOR','L1','MatchedBase','FlatROI','BraSyn','PathologyLoss','CEILING']:
        if a not in t: continue
        s = t[a].dropna()
        print(f'{a:<15}{len(s):>4}{s.mean():>9.4f}{s.std():>8.4f}')
    print()

print('===== PAIRED subject-level contrasts on dice_et (common subjects only) =====')
t = subj('dice_et')
pairs = [('PathologyLoss','MatchedBase'),('PathologyLoss','L1'),
         ('PathologyLoss','FlatROI'),('PathologyLoss','BraSyn'),
         ('MatchedBase','L1'),('MatchedBase','BraSyn'),('MatchedBase','FlatROI')]
print(f'{"contrast":<32}{"n":>4}{"delta":>9}{"p":>9}{"win":>6}')
for a,b in pairs:
    p = t[[a,b]].dropna()
    diff = p[a]-p[b]
    try: _,pv = wilcoxon(p[a],p[b])
    except Exception: pv=float('nan')
    print(f'{a+" - "+b:<32}{len(p):>4}{diff.mean():>+9.4f}{pv:>9.4f}{(diff>0).sum():>4}/{len(p)}')

print()
print('===== PAIRED on psnr_et =====')
t2 = subj('psnr_et')
print(f'{"contrast":<32}{"n":>4}{"delta":>9}{"p":>9}{"win":>6}')
for a,b in pairs:
    p = t2[[a,b]].dropna()
    diff = p[a]-p[b]
    try: _,pv = wilcoxon(p[a],p[b])
    except Exception: pv=float('nan')
    print(f'{a+" - "+b:<32}{len(p):>4}{diff.mean():>+9.4f}{pv:>9.4f}{(diff>0).sum():>4}/{len(p)}')
